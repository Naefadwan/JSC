"""Tests for the deterministic company detector."""

from __future__ import annotations

from jsc_review.companies.autolink import (
    decide_autolink,
    iter_pages_from_extracted,
    iter_pages_from_ocr,
    merge_pages_with_ocr,
    normalize_arabic,
    normalize_latin,
    rank_candidates,
    rank_candidates_with_fallback,
)
from jsc_review.companies.registry import Company, CompanyRegistry


def _registry() -> CompanyRegistry:
    return CompanyRegistry(
        companies=(
            Company(
                name_ar="البنك الإسلامي الأردني",
                short_name_ar="بنك الإسلامي",
                ticker="JOIB",
                numeric_code="111001",
                sector_ar="البنوك",
            ),
            Company(
                name_ar="بنك الاردن",
                short_name_ar="بنك الاردن",
                ticker="BOJX",
                numeric_code="111022",
                sector_ar="البنوك",
            ),
            Company(
                name_ar="التأمين الاردنية",
                short_name_ar="التأمين الاردنية",
                ticker="JOIN",
                numeric_code="121004",
                sector_ar="التأمين",
            ),
        )
    )


# ---------------------------------------------------- normalization --------


def test_normalize_arabic_strips_diacritics_and_folds_variants() -> None:
    raw = "البَنْكُ الإسلامي الأُرْدُنِي"
    normalized = normalize_arabic(raw)
    assert "َ" not in normalized  # no fatha
    assert "ُ" not in normalized  # no damma
    # alef variants collapsed to bare alef
    assert "إ" not in normalized
    assert "أ" not in normalized


def test_normalize_arabic_folds_ta_marbuta_and_yaa() -> None:
    assert normalize_arabic("شركة") == normalize_arabic("شركه")
    assert normalize_arabic("شركات تجاري") == normalize_arabic("شركات تجاري")
    assert normalize_arabic("ليلى") == normalize_arabic("ليلي")


def test_normalize_arabic_collapses_whitespace_and_removes_zwj() -> None:
    assert normalize_arabic("بنك\u200bالاردن") == normalize_arabic("بنكالاردن")
    assert normalize_arabic("  بنك  الاردن  ") == normalize_arabic("بنك الاردن")


def test_normalize_latin_uppercase_and_whitespace_strip() -> None:
    assert normalize_latin("  joib  ") == "JOIB"
    assert normalize_latin("JO IB") == "JOIB"


# ---------------------------------------------------- rank_candidates ------


def test_rank_candidates_returns_empty_without_text() -> None:
    assert rank_candidates([(1, "")], _registry()) == []


def test_rank_candidates_returns_empty_with_empty_registry() -> None:
    assert rank_candidates(
        [(1, "البنك الإسلامي الأردني")],
        CompanyRegistry(companies=()),
    ) == []


def test_rank_candidates_prefers_full_name_hit() -> None:
    # Issuer-page style: the standalone name on a cover line.
    pages = [(1, "البنك الإسلامي الأردني\nالتقرير السنوي 2026")]
    matches = rank_candidates(pages, _registry())
    assert matches, "expected at least one match"
    assert matches[0].company.ticker == "JOIB"
    assert matches[0].hits >= 1
    assert "البنك الإسلامي الأردني" in matches[0].matched_terms


def test_rank_candidates_matches_ticker_in_latin_text() -> None:
    pages = [(1, "Issuer: JOIB — annual filing")]
    matches = rank_candidates(pages, _registry())
    assert matches
    assert matches[0].company.ticker == "JOIB"


def test_rank_candidates_does_not_match_ticker_substring_inside_word() -> None:
    # 'JOIBE' should NOT trigger 'JOIB' because of the word-boundary guard.
    pages = [(1, "The company name is JOIBESS and this is irrelevant.")]
    matches = rank_candidates(pages, _registry())
    assert matches == []


def test_rank_candidates_scores_full_name_above_short_only() -> None:
    pages = [
        (1, "البنك الإسلامي الأردني"),
        (2, "بنك الإسلامي"),
    ]
    matches = rank_candidates(pages, _registry(), scan_pages=2)
    assert matches[0].company.ticker == "JOIB"
    # The ticker JOIB should not show up at all here (no latin text).
    for match in matches:
        assert "JOIB" not in [match.company.name_ar]  # sanity: not confused


def test_rank_candidates_handles_normalization_variants() -> None:
    # Diacritics plus alternate alef should still match the registry entry.
    pages = [(1, "البَنْكُ الإِسلاميُ الأُردني")]
    matches = rank_candidates(pages, _registry())
    assert matches
    assert matches[0].company.ticker == "JOIB"


def test_rank_candidates_respects_scan_pages_window() -> None:
    """``scan_pages`` caps the number of non-empty pages the scanner reads."""

    pages = [
        (1, "lorem ipsum dolor sit amet consectetur adipiscing elit"),
        (2, "sed do eiusmod tempor incididunt ut labore et dolore"),
        (3, "ut enim ad minim veniam quis nostrud exercitation"),
        (4, "البنك الإسلامي الأردني التقرير السنوي"),
    ]
    # Only the first 3 non-empty pages count -> the issuer name is skipped.
    assert rank_candidates(pages, _registry(), scan_pages=3) == []
    # With a 4-page budget the issuer name on page 4 enters the window.
    assert rank_candidates(pages, _registry(), scan_pages=4)


# ---------------------------------------------------- decide_autolink ------


def test_decide_autolink_accepts_clear_leader() -> None:
    candidates = rank_candidates(
        [(1, "التقرير السنوي للبنك الإسلامي الأردني")], _registry()
    )
    decision = decide_autolink(candidates, threshold=0.5, min_lead_margin=0.25)
    assert decision.reason == "accepted"
    assert decision.accepted is not None
    assert decision.accepted.company.ticker == "JOIB"


def test_decide_autolink_rejects_when_below_threshold() -> None:
    # single weak hit against a very high threshold
    candidates = rank_candidates(
        [(1, "برنامج التمويل في بنك الإسلامي")], _registry()
    )
    decision = decide_autolink(candidates, threshold=100.0)
    assert decision.reason == "below_threshold"
    assert decision.accepted is None
    assert decision.candidates  # suggestions still surfaced


def test_decide_autolink_rejects_ambiguous_pair() -> None:
    # Two companies with the same two-word mention tie up for the top spot.
    custom = CompanyRegistry(
        companies=(
            Company(name_ar="شركة الأمل", ticker="A", numeric_code="1"),
            Company(name_ar="شركة الأمل", ticker="B", numeric_code="2"),
        )
    )
    candidates = rank_candidates([(1, "شركة الأمل")], custom)
    decision = decide_autolink(candidates, threshold=0.5, min_lead_margin=0.25)
    assert decision.reason == "ambiguous"
    assert decision.accepted is None
    assert {match.company.ticker for match in decision.candidates} == {"A", "B"}


def test_decide_autolink_no_candidates_is_no_match() -> None:
    decision = decide_autolink([], threshold=0.5)
    assert decision.reason == "no_match"
    assert decision.accepted is None
    assert decision.candidates == ()


# ---------------------------------------------------- iter_pages_from_extracted


def test_iter_pages_from_extracted_handles_well_formed_payload() -> None:
    payload = {
        "pages": [
            {"page": 1, "text": "first"},
            {"page": 2, "text": "second"},
            {"page": 3, "text": "third"},
        ]
    }
    result = list(iter_pages_from_extracted(payload))
    assert result == [(1, "first"), (2, "second"), (3, "third")]


def test_iter_pages_from_extracted_tolerates_malformed_entries() -> None:
    payload = {
        "pages": [
            {"page": 1, "text": "ok"},
            "not a dict",
            {"page": "oops", "text": "no"},  # page is not int
            {"page": 2},  # no text
            {"page": 3, "text": "fine"},
        ]
    }
    result = list(iter_pages_from_extracted(payload))
    assert result == [(1, "ok"), (3, "fine")]


def test_iter_pages_from_extracted_handles_missing_pages_field() -> None:
    assert list(iter_pages_from_extracted({})) == []
    assert list(iter_pages_from_extracted({"pages": "no"})) == []


# ---------------------------------------------------- OCR + merge ---------


def test_iter_pages_from_ocr_handles_the_native_shape() -> None:
    payload = {
        "pages": [
            {"page_number": 1, "text": "first ocr"},
            {"page_number": 2, "text": "second ocr"},
        ]
    }
    assert list(iter_pages_from_ocr(payload)) == [(1, "first ocr"), (2, "second ocr")]


def test_iter_pages_from_ocr_also_tolerates_page_key() -> None:
    payload = {"pages": [{"page": 3, "text": "legacy"}]}
    assert list(iter_pages_from_ocr(payload)) == [(3, "legacy")]


def test_merge_pages_with_ocr_prefers_native_over_ocr() -> None:
    native = [(1, "native page"), (2, "")]
    ocr = [(1, "ocr page"), (2, "scanned cover"), (3, "appendix ocr")]
    merged = merge_pages_with_ocr(native, ocr)
    # Page 1 keeps native; page 2 filled from OCR; page 3 added from OCR.
    assert merged == [(1, "native page"), (2, "scanned cover"), (3, "appendix ocr")]


def test_merge_pages_with_ocr_skips_pages_blank_in_both_sources() -> None:
    native = [(1, "   "), (2, "")]
    ocr = [(1, ""), (2, "   ")]
    assert merge_pages_with_ocr(native, ocr) == []


# ---------------------------------------------------- deep fallback ------


def test_rank_candidates_with_fallback_uses_cheap_scan_when_it_matches() -> None:
    pages = [
        (1, "البنك الإسلامي الأردني التقرير السنوي"),
        (2, "محتوى آخر"),
    ]
    result = rank_candidates_with_fallback(
        pages, _registry(), scan_pages=1, deep_scan_pages=20
    )
    assert result.candidates, "expected a candidate from the quick scan"
    assert result.candidates[0].company.ticker == "JOIB"
    assert result.deep is False  # quick scan was enough


def test_rank_candidates_with_fallback_widens_when_quick_scan_misses() -> None:
    pages = [
        (1, "lorem ipsum dolor sit amet consectetur adipiscing"),
        (2, "sed do eiusmod tempor incididunt ut labore"),
        (3, "ut enim ad minim veniam quis nostrud"),
        (4, "duis aute irure dolor in reprehenderit"),
        (5, "البنك الإسلامي الأردني التقرير السنوي 2026"),
    ]
    quick = rank_candidates_with_fallback(
        pages, _registry(), scan_pages=3, deep_scan_pages=3
    )
    assert quick.is_empty
    assert quick.deep is False  # no fallback attempted when budgets equal
    deep = rank_candidates_with_fallback(
        pages, _registry(), scan_pages=3, deep_scan_pages=10
    )
    assert deep.candidates
    assert deep.candidates[0].company.ticker == "JOIB"
    assert deep.deep is True


def test_decide_autolink_refuses_to_accept_deep_scan_matches() -> None:
    candidates = rank_candidates(
        [(1, "البنك الإسلامي الأردني التقرير السنوي")], _registry()
    )
    assert candidates, "fixture must produce at least one candidate"
    decision = decide_autolink(
        candidates, threshold=0.5, min_lead_margin=0.25, from_deep_scan=True
    )
    assert decision.reason == "deep_suggestion"
    assert decision.accepted is None
    assert decision.candidates  # suggestions still surfaced for the reviewer
