"""Detect the company a document belongs to from its extracted text.

The detector is deterministic and purely local. It scans a prefix of the
document (typically the first few pages, where the cover and header carry
the issuer name) and scores every company in the registry by how often its
Arabic name, short name, ticker, or numeric code appears. The reviewer sees
the ranked candidates and either accepts the top one, picks one of the
others, or ignores the suggestion.

The scorer favours precise matches and short-but-specific names over long
descriptions. It is intentionally conservative: when two companies tie or
the leader has only a tiny margin, it refuses to auto-link and surfaces the
tie so a human picks.

Nothing in this module writes to disk, calls any network, or depends on an
LLM. Autolink hooks in the pipeline call this and then decide whether to
write a link via :func:`jsc_review.companies.mapping.set_link`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from jsc_review.companies.registry import Company, CompanyRegistry

# Arabic text normalisation ---------------------------------------------------
#
# Issuer names on Jordanian cover pages are typed in a dozen stylistically
# different ways: with or without diacritics, with أ/إ/آ instead of ا, with
# ي rendered as ى, with ة rendered as ه, with embedded zero-width chars from
# PDF text extraction, and with newlines splitting the name. The normalizer
# collapses every one of those into a canonical form before matching so the
# scorer sees a stable string.

_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # harakat + tatweel
_ALEF_VARIANTS = str.maketrans({
    "\u0623": "\u0627",  # أ
    "\u0625": "\u0627",  # إ
    "\u0622": "\u0627",  # آ
    "\u0671": "\u0627",  # ٱ
    "\u0649": "\u064A",  # ى -> ي
    "\u0629": "\u0647",  # ة -> ه
    "\u0624": "\u0648",  # ؤ -> و
    "\u0626": "\u064A",  # ئ -> ي
})
_PUNCT_TRANSLATE = str.maketrans({
    char: " "
    for char in (
        ".,;:!?\u060c\u061b\u061f\"'()[]{}<>/\\|-_+=*&^%$#@~`\u00b7"
    )
})
_ZERO_WIDTH = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")
_WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Return a canonical form that collapses the common Arabic spelling noise."""

    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    folded = _ZERO_WIDTH.sub("", folded)
    folded = _ARABIC_DIACRITICS.sub("", folded)
    folded = folded.translate(_ALEF_VARIANTS)
    folded = folded.translate(_PUNCT_TRANSLATE)
    folded = _WHITESPACE.sub(" ", folded).strip()
    return folded.lower()


def normalize_latin(text: str) -> str:
    """Return a stripped, upper-cased form for ticker and numeric-code matching."""

    return re.sub(r"\s+", "", (text or "")).upper()


# Match + score ---------------------------------------------------------------


@dataclass(frozen=True)
class CompanyMatch:
    """One candidate company for a document, with evidence of why it matched."""

    company: Company
    score: float
    hits: int
    matched_terms: tuple[str, ...]


# Weighting rationale:
# * ticker: ~1.0 — tickers are rare and specific enough that one hit is
#   near-conclusive, but we still require a text occurrence (not just the
#   filename) so random noise does not win.
# * short_name_ar: ~0.9 — short name is the most frequent reference in body
#   text and headers.
# * name_ar: ~1.0 — the full legal name is the strongest signal.
# * numeric_code: ~0.6 — the 6-digit code occasionally appears but can also
#   collide with other numbers in financial tables.
_TICKER_WEIGHT = 1.0
_NUMERIC_WEIGHT = 0.6
_SHORT_NAME_WEIGHT = 0.9
_FULL_NAME_WEIGHT = 1.0

# When the top candidate's score exceeds the threshold AND leads the runner
# up by at least ``_MIN_LEAD_MARGIN`` of its own score, we auto-link.
# Otherwise the reviewer picks. Tunable from Settings.
_MIN_LEAD_MARGIN_DEFAULT = 0.25


def _count_occurrences(haystack: str, needle: str) -> int:
    """Return how many times ``needle`` appears in ``haystack`` (0 when empty)."""

    if not needle or not haystack:
        return 0
    # str.count is O(n*m) worst-case; names are small enough that this is fine.
    return haystack.count(needle)


# Arabic prefix letters that fuse directly onto nouns on the left-hand side
# (ل for, ب with/in, ك as, ف then, و and, س future, ال the). Up to two of
# them may stack (e.g. ``وبال`` = "and with the"). We allow this zero-to-two
# character prefix in word-boundary matches so "للبنك" counts as a hit for
# "البنك".
_ARABIC_PREFIX_CHARS = "\u0644\u0628\u0643\u0641\u0648\u0633\u0627"  # ل ب ك ف و س ا
_ARABIC_LETTER = "[\u0600-\u06FF]"


def _arabic_word_count_pattern(needle: str) -> str:
    """Return a regex that matches ``needle`` as a word with an optional prefix."""

    return (
        rf"(?<!{_ARABIC_LETTER})"  # left word boundary: no Arabic letter before
        rf"[{_ARABIC_PREFIX_CHARS}]{{0,2}}"  # optional stacked prefix
        + re.escape(needle)
        + rf"(?!{_ARABIC_LETTER})"  # right word boundary
    )


def _count_arabic_word(haystack: str, needle: str) -> int:
    """Return occurrences of ``needle`` with Arabic word-boundary semantics."""

    if not needle or not haystack:
        return 0
    return len(re.findall(_arabic_word_count_pattern(needle), haystack))


def _is_specific_enough(normalized_term: str) -> bool:
    """Return True when the term is long or multi-word enough to avoid noise.

    Short common words like "أمل" (hope) collide with random Arabic text.
    A term earns its score when it has at least two words OR is at least
    six characters long.
    """

    if not normalized_term:
        return False
    if " " in normalized_term:
        return True
    return len(normalized_term) >= 6


def _score_one(company: Company, text_ar: str, text_latin: str) -> CompanyMatch:
    """Return the match score for one company against the normalized text."""

    hits = 0
    weighted = 0.0
    terms: list[str] = []

    full_name = normalize_arabic(company.name_ar)
    short_name = normalize_arabic(company.short_name_ar)
    ticker = normalize_latin(company.ticker)
    numeric = normalize_latin(company.numeric_code)

    if full_name and _is_specific_enough(full_name):
        count = _count_arabic_word(text_ar, full_name)
        if count:
            hits += count
            weighted += count * _FULL_NAME_WEIGHT
            terms.append(company.name_ar)
    if short_name and short_name != full_name and _is_specific_enough(short_name):
        count = _count_arabic_word(text_ar, short_name)
        if count:
            hits += count
            weighted += count * _SHORT_NAME_WEIGHT
            terms.append(company.short_name_ar)
    if ticker:
        # Require a word-boundary-like context so three-letter sequences inside
        # other words do not count. ``normalize_latin`` keeps only letters and
        # digits so we check surrounding whitespace or digits via a regex.
        pattern = rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])"
        count = len(re.findall(pattern, text_latin))
        if count:
            hits += count
            weighted += count * _TICKER_WEIGHT
            terms.append(company.ticker)
    if numeric:
        count = len(re.findall(rf"(?<!\d){re.escape(numeric)}(?!\d)", text_latin))
        if count:
            hits += count
            weighted += count * _NUMERIC_WEIGHT
            terms.append(company.numeric_code)

    return CompanyMatch(
        company=company,
        score=weighted,
        hits=hits,
        matched_terms=tuple(terms),
    )


_MIN_TEXT_CHARS_PER_PAGE = 20


def _extract_scan_text(
    pages: Sequence[tuple[int, str]], scan_pages: int
) -> tuple[str, str]:
    """Return (normalized_arabic_text, normalized_latin_text) for the scan window.

    ``pages`` is a sequence of ``(page_number, raw_text)`` tuples. The
    scanner concatenates the first ``scan_pages`` *non-empty* pages (lowest
    numbers first) so a blank or image-only cover page does not consume the
    scan budget. Latin-script content is extracted separately so ticker
    matching does not require Arabic normalisation.
    """

    ordered = sorted(pages, key=lambda item: item[0])
    chosen: list[tuple[int, str]] = []
    for item in ordered:
        text = (item[1] or "").strip()
        if len(text) < _MIN_TEXT_CHARS_PER_PAGE:
            continue
        chosen.append(item)
        if len(chosen) >= scan_pages:
            break
    # Fall back to the very first N pages if everything was below the threshold
    # (e.g. a dense table of numbers only), so we still try rather than give up.
    if not chosen:
        chosen = ordered[:scan_pages]
    raw = "\n".join(text for _, text in chosen if text)
    return normalize_arabic(raw), normalize_latin(raw)


def rank_candidates(
    pages: Sequence[tuple[int, str]],
    registry: CompanyRegistry,
    *,
    scan_pages: int = 3,
    top_k: int = 5,
) -> list[CompanyMatch]:
    """Return the top ``top_k`` candidates for a document, strongest first.

    Only matches with a positive score appear. The list is sorted by score
    descending then by hit count descending. When no company matches the
    text, an empty list comes back.
    """

    if not registry.companies:
        return []
    text_ar, text_latin = _extract_scan_text(pages, scan_pages)
    if not text_ar and not text_latin:
        return []
    scored = [
        _score_one(company, text_ar, text_latin) for company in registry.companies
    ]
    scored = [match for match in scored if match.score > 0]
    scored.sort(key=lambda match: (-match.score, -match.hits, match.company.name_ar))
    return scored[:top_k]


@dataclass(frozen=True)
class ScanResult:
    """A ranked candidate list plus the scan depth that produced it.

    ``deep`` is ``True`` when the quick scan missed and we widened the
    window. Deep results carry more noise (reports reference other
    companies as partners or competitors) so the policy treats them as
    reviewer suggestions and refuses to auto-link from them.
    """

    candidates: tuple[CompanyMatch, ...]
    deep: bool

    @property
    def is_empty(self) -> bool:
        return not self.candidates


def rank_candidates_with_fallback(
    pages: Sequence[tuple[int, str]],
    registry: CompanyRegistry,
    *,
    scan_pages: int,
    deep_scan_pages: int,
    top_k: int = 5,
) -> ScanResult:
    """Try a cheap scan first; widen the window when nothing matches.

    Reviewer-friendly: the quick scan handles the common case in a few
    milliseconds, and only hard docs pay the price of scanning dozens of
    pages. ``deep_scan_pages`` is a soft cap -- when the cheap scan returns
    at least one match, the function skips the deeper pass entirely.
    """

    quick = rank_candidates(pages, registry, scan_pages=scan_pages, top_k=top_k)
    if quick:
        return ScanResult(candidates=tuple(quick), deep=False)
    if deep_scan_pages <= scan_pages:
        return ScanResult(candidates=(), deep=False)
    deep = rank_candidates(pages, registry, scan_pages=deep_scan_pages, top_k=top_k)
    return ScanResult(candidates=tuple(deep), deep=True)


def merge_pages_with_ocr(
    native_pages: Sequence[tuple[int, str]],
    ocr_pages: Sequence[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Return pages where OCR text supplements empty native extraction.

    The native extraction wins when it has usable text. OCR fills in for
    scanned covers that would otherwise yield blank pages to the scorer.
    Pages present in neither source are omitted. Output is ordered by page
    number so the scanner sees the natural reading order.
    """

    merged: dict[int, str] = {}
    for page_number, text in native_pages:
        if text and text.strip():
            merged[page_number] = text
    for page_number, text in ocr_pages:
        if page_number in merged:
            continue
        if text and text.strip():
            merged[page_number] = text
    return sorted(merged.items(), key=lambda item: item[0])


def iter_pages_from_ocr(payload: Mapping[str, object]) -> Iterable[tuple[int, str]]:
    """Yield ``(page_number, text)`` tuples from an OCR JSON payload.

    The OCR writer stores one record per OCR'd page under ``pages``. The
    shape mirrors the extracted file so the same iterator pattern works.
    """

    pages_field = payload.get("pages", [])
    if not isinstance(pages_field, list):
        return
    for page in pages_field:
        if not isinstance(page, dict):
            continue
        if "text" not in page:
            continue
        page_number = page.get("page_number", page.get("page"))
        text = page.get("text")
        if isinstance(page_number, int) and isinstance(text, str):
            yield page_number, text


@dataclass(frozen=True)
class AutoLinkDecision:
    """What the autolink policy decided for one document."""

    accepted: CompanyMatch | None
    candidates: tuple[CompanyMatch, ...]
    reason: str  # "accepted", "below_threshold", "ambiguous", "no_match"


def decide_autolink(
    candidates: Sequence[CompanyMatch],
    *,
    threshold: float,
    min_lead_margin: float = _MIN_LEAD_MARGIN_DEFAULT,
    from_deep_scan: bool = False,
) -> AutoLinkDecision:
    """Apply the strict-with-suggestions policy to a ranked candidate list.

    * no candidates -> ``no_match``
    * top score below threshold -> ``below_threshold`` (surface suggestions)
    * runner-up within ``min_lead_margin`` of top -> ``ambiguous`` (surface)
    * otherwise -> ``accepted`` and the top candidate becomes the auto-link

    When ``from_deep_scan`` is True the policy never accepts: deep-scan
    matches carry too much noise (reports reference partner and competitor
    names), so they appear as suggestions the reviewer confirms by hand.
    """

    if not candidates:
        return AutoLinkDecision(
            accepted=None, candidates=(), reason="no_match"
        )
    top = candidates[0]
    if top.score < threshold:
        return AutoLinkDecision(
            accepted=None, candidates=tuple(candidates), reason="below_threshold"
        )
    if len(candidates) >= 2:
        runner_up = candidates[1]
        if top.score - runner_up.score < min_lead_margin * top.score:
            return AutoLinkDecision(
                accepted=None, candidates=tuple(candidates), reason="ambiguous"
            )
    if from_deep_scan:
        return AutoLinkDecision(
            accepted=None,
            candidates=tuple(candidates),
            reason="deep_suggestion",
        )
    return AutoLinkDecision(
        accepted=top, candidates=tuple(candidates), reason="accepted"
    )


def iter_pages_from_extracted(payload: Mapping[str, object]) -> Iterable[tuple[int, str]]:
    """Yield ``(page_number, text)`` tuples from an extracted JSON payload.

    Isolated here so the detector can run against on-disk files without
    importing the full pipeline module. Entries without both ``page`` (int)
    and ``text`` (str) fields are skipped so a malformed row cannot pollute
    the scanner input.
    """

    pages_field = payload.get("pages", [])
    if not isinstance(pages_field, list):
        return
    for page in pages_field:
        if not isinstance(page, dict):
            continue
        if "text" not in page:
            continue
        page_number = page.get("page")
        text = page.get("text")
        if isinstance(page_number, int) and isinstance(text, str):
            yield page_number, text
