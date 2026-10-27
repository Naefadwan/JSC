"""Tests for the Phase A/B/C deterministic rules.

Covers:
- Arabic affix tolerance in :func:`_anchor_pattern` / :func:`_find_variant`
- The new HEADING_VARIANTS entries (Phase B strong + presence-only)
- The multi-anchor rule (DIS-067)
- The early-pages rule (GOV-006)
- The table-structure rule (DIS-004)
"""

from __future__ import annotations

from jsc_review.review.findings import (
    HEADING_VARIANTS,
    TABLE_STRUCTURE_RULES,
    Requirement,
    _anchor_pattern,
    _find_variant,
    _find_variant_within,
    _find_variants_on_same_page,
    _has_table_structure,
    review_document,
)

# --------------------------------------------------------- affix tolerance


def test_anchor_pattern_matches_bare_arabic() -> None:
    pattern = _anchor_pattern("الميزانية")
    assert pattern.search("هذه الميزانية العمومية") is not None


def test_anchor_pattern_matches_prefixed_arabic() -> None:
    pattern = _anchor_pattern("الميزانية")
    # Common prefixed forms: lam (ل), ba (ب), waw (و), kaf (ك), fa (ف)
    for text in [
        "للميزانية العامة",
        "بالميزانية العمومية",
        "والميزانية الموحدة",
        "كالميزانية السنوية",
        "فالميزانية مرفقة",
    ]:
        assert pattern.search(text) is not None, text


def test_anchor_pattern_does_not_cross_letter_boundary() -> None:
    pattern = _anchor_pattern("الميزانية")
    # A longer word that merely contains the anchor's letters as a suffix
    # must not match (there is no such Arabic word in practice but the guard
    # should still hold).
    assert pattern.search("شيءلميزانيةرى") is None


def test_anchor_pattern_latin_uses_word_boundary() -> None:
    pattern = _anchor_pattern("balance sheet")
    assert pattern.search("A balance sheet appears here") is not None
    assert pattern.search("balance sheet.") is not None
    assert pattern.search("rebalance sheetings") is None  # suffix must not match


def test_find_variant_matches_despite_dropped_hamza() -> None:
    # OCR frequently emits ``ادارة`` (no hamza) for ``إدارة`` (with hamza).
    # The normalized view folds both into the same letter so the anchor hits.
    from jsc_review.review.findings import _find_variant as find_variant  # local to avoid rebinding
    pages = {4: "سياسة ادارة المخاطر معتمدة من مجلس الادارة"}
    hit = find_variant(pages, ("سياسة إدارة المخاطر",))
    assert hit is not None
    page, variant, excerpt = hit
    assert page == 4
    assert variant == "سياسة إدارة المخاطر"
    assert "ادارة" in excerpt  # the raw text keeps the OCR'd form


def test_find_variant_matches_despite_dropped_alef_madda() -> None:
    # ``آ`` (U+0622, alef with madda) normalizes to the bare alef ``ا``.
    pages = {1: "القرارات الاستراتيجية هي أساس العمل"}
    hit = _find_variant(pages, ("آساس",))  # anchor uses alef madda
    assert hit is not None
    page, variant, excerpt = hit
    assert page == 1
    assert "اساس" in excerpt or "أساس" in excerpt


def test_find_variant_returns_first_hit_with_window() -> None:
    pages = {1: "لا شيء هنا", 3: "تحت عنوان للميزانية العامة الموحدة"}
    found = _find_variant(pages, ("الميزانية",))
    assert found is not None
    page_number, variant, excerpt = found
    assert page_number == 3
    assert variant == "الميزانية"
    # The excerpt captures surrounding words so a reviewer can audit the hit.
    assert "للميزانية" in excerpt
    assert "العامة" in excerpt


# ------------------------------------------- HEADING_VARIANTS coverage --


def _req(req_id: str) -> Requirement:
    return Requirement(
        requirement_id=req_id,
        section="s",
        text_ar="",
        text_en="",
        evaluation_method="deterministic",
    )


def test_new_heading_variants_cover_every_converted_requirement() -> None:
    converted = {
        "GOV-001",
        "GOV-002",
        "GOV-007",
        "GOV-011",
        "GOV-015",
        "DIS-001",
        "DIS-026",
        "DIS-034",
        "DIS-039",
        "DIS-056",
        "DIS-058",
    }
    assert converted.issubset(HEADING_VARIANTS)


def test_heading_rule_meets_on_code_of_conduct_anchor() -> None:
    pages = {
        45: (
            "اعتمد مجلس الإدارة مدونة السلوك المهني للشركة بما يضمن "
            "المعايير المهنية السليمة للتعامل."
        )
    }
    finding = review_document("doc1", _req("GOV-002"), pages, {})
    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"


def test_dis058_meets_on_bank_variant() -> None:
    # Banks declare continuity for "the bank", not "the company".
    pages = {
        88: (
            "يقر مجلس إدارة البنك الإسلامي الأردني بعدم وجود أي أمور جوهرية "
            "قد تؤثر على استمرارية البنك خلال السنة المالية التالية."
        )
    }
    finding = review_document("doc1", _req("DIS-058"), pages, {})
    assert finding.result.status == "Met"


def test_dis058_meets_when_word_boundary_is_short_stopped() -> None:
    # OCR sometimes glues the next word onto "تؤثر", producing "تؤثرعلى".
    # The short-stop anchor "بعدم وجود أي أمور جوهرية" ends at a letter
    # boundary and matches regardless of what follows.
    pages = {
        28: (
            "يقر مجلس إدارة الشركة بعدم وجود أي أمور جوهرية قد تؤثرعلى "
            "استمرار الشركة خلال السنة المالية التالية."
        )
    }
    finding = review_document("doc1", _req("DIS-058"), pages, {})
    assert finding.result.status == "Met"


def test_dis056_meets_on_related_parties_balance_summary() -> None:
    pages = {
        56: (
            "فيما يلي ملخص الأرصدة مع الجهات ذات علاقة الظاهرة في قائمة "
            "الأرباح أو الخسائر الموحدة."
        )
    }
    finding = review_document("doc1", _req("DIS-056"), pages, {})
    assert finding.result.status == "Met"


def test_dis017_falls_through_to_threshold_path_on_pct_disclosure() -> None:
    # When the heading anchors do not fire, DIS-017 falls through to the
    # threshold path so a report that mentions "موردين" plus a
    # concentration figure of 10% or more is still Met.
    page = "تعاونت الشركة مع الموردين الذين يشكلون 12% من إجمالي المشتريات."
    finding = review_document("doc1", _req("DIS-017"), {6: page}, {})
    assert finding.result.status == "Met"


def test_dis017_meets_on_no_major_customer_disclosure() -> None:
    # A disclosure stating that no customers exceed the 10% threshold is
    # itself a valid DIS-017 disclosure. The heading-anchor path catches
    # it without needing a percentage figure.
    pages = {
        14: (
            "درجة الاعتماد على موردين محددين أو عملاء رئيسيين محليا "
            "وخارجيا يشكلون 10% فأكثر من إجمالي المشتريات: "
            "لا يوجد موردين محددين أو عملاء رئيسيين."
        )
    }
    finding = review_document("doc1", _req("DIS-017"), pages, {})
    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"


def test_heading_rule_meets_on_board_members_anchor() -> None:
    pages = {5: "تشكيلة مجلس الإدارة الحالية مكونة من سبعة أعضاء"}
    finding = review_document("doc1", _req("GOV-007"), pages, {})
    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"
    assert finding.result.evidence is not None and finding.result.evidence.verified


def test_heading_rule_meets_on_going_concern_anchor() -> None:
    pages = {
        12: (
            "يقر مجلس الإدارة بعدم وجود أية أمور جوهرية قد تؤثر على "
            "استمرارية الشركة خلال السنة المالية التالية."
        )
    }
    finding = review_document("doc1", _req("DIS-058"), pages, {})
    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"


def test_heading_rule_needs_review_when_no_anchor_found() -> None:
    pages = {1: "محتوى غير مرتبط بالمتطلب"}
    finding = review_document("doc1", _req("GOV-011"), pages, {})
    assert finding.result.status == "Needs Review"


# ----------------------------------------- DIS-067 auditor's report ------


def test_dis067_meets_on_auditor_report_heading() -> None:
    # Presence of the auditor's report heading is the deterministic signal.
    # Earlier multi-anchor logic was wrong: real auditor's reports cite
    # auditing standards (ISA), not the regulator's disclosure instructions.
    assert "DIS-067" in HEADING_VARIANTS
    pages = {
        3: (
            "تقرير مدققي الحسابات المستقلين إلى مساهمي الشركة. "
            "تقرير حول تدقيق القوائم المالية الموحدة."
        )
    }
    finding = review_document("doc1", _req("DIS-067"), pages, {})
    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"


def test_dis067_meets_on_english_independent_auditor_heading() -> None:
    pages = {229: "INDEPENDENT AUDITOR'S REPORT TO THE SHAREHOLDERS OF ..."}
    finding = review_document("doc1", _req("DIS-067"), pages, {})
    assert finding.result.status == "Met"


def test_dis067_needs_review_when_no_auditor_heading_present() -> None:
    pages = {3: "لا توجد عناوين تدقيق على هذه الصفحة."}
    finding = review_document("doc1", _req("DIS-067"), pages, {})
    assert finding.result.status == "Needs Review"


def test_find_variants_on_same_page_does_not_cross_pages() -> None:
    # The helper is still used by MULTI_ANCHOR_RULES if any entry is added.
    anchors = (("x",), ("y",))
    pages = {1: "x only here", 2: "y only here"}
    assert _find_variants_on_same_page(pages, anchors) is None
    pages_same = {1: "x and y on one page"}
    hit = _find_variants_on_same_page(pages_same, anchors)
    assert hit is not None and hit[0] == 1


# ----------------------------------------- GOV-006 governance chapter ---


def test_gov006_meets_on_anchor_anywhere_in_report() -> None:
    assert "GOV-006" in HEADING_VARIANTS
    pages = {72: "دليل قواعد حوكمة الشركات المساهمة المدرجة في بورصة عمان"}
    finding = review_document("doc1", _req("GOV-006"), pages, {})
    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"


def test_gov006_needs_review_when_no_anchor_present() -> None:
    pages = {5: "نص لا يذكر الحوكمة"}
    finding = review_document("doc1", _req("GOV-006"), pages, {})
    assert finding.result.status == "Needs Review"


def test_find_variant_within_respects_max_page() -> None:
    pages = {1: "noise", 10: "الميزانية العامة"}
    assert _find_variant_within(pages, ("الميزانية",), max_page=5) is None
    assert _find_variant_within(pages, ("الميزانية",), max_page=15) is not None


# --------------------------------------- table structure (DIS-004) ------


def test_has_table_structure_detects_tabular_block() -> None:
    text = (
        "عدد الموظفين\n"
        "النشاط      الموقع      العدد\n"
        "البيع       عمان        120\n"
        "الإنتاج     الزرقاء     85\n"
        "الإدارة     عمان        30\n"
    )
    assert _has_table_structure(text, min_cols=2, min_rows=3) is True


def test_has_table_structure_rejects_plain_prose() -> None:
    text = (
        "عدد الموظفين في الشركة اثنان وخمسون موظفًا موزعين على "
        "مختلف الإدارات والفروع."
    )
    assert _has_table_structure(text, min_cols=2, min_rows=3) is False


def test_has_table_structure_detects_dense_digit_cluster() -> None:
    # Arabic PDFs often lose inter-column whitespace during extraction,
    # leaving a dense run of digits. That digit cluster is the deterministic
    # signal that the page is a table of counts.
    text = (
        "عدد الموظفين عدد الفروع والمكاتب 97 108 111 105 111 100 109 105 108 111 "
        "2236 2440 2440 2335 2434 2437 2405 2439 2445 2414 2018 2021 2016 2019 "
        "2022 2017 2020 2023 2024 2025"
    )
    assert _has_table_structure(text, min_cols=2, min_rows=3) is True


def test_has_table_structure_detects_arabic_indic_digit_cluster() -> None:
    # Same signal using Arabic-Indic digits ٠-٩ (U+0660 .. U+0669).
    text = (
        "عدد الموظفين ٩٧ ١٠٨ ١١١ ١٠٥ ١١١ ١٠٠ ١٠٩ ١٠٥ ١٠٨ ١١١ ٢٠٢٠ ٢٠٢١ ٢٠٢٢ ٢٠٢٣"
    )
    assert _has_table_structure(text, min_cols=2, min_rows=3) is True


def test_dis004_meets_on_anchor_with_table() -> None:
    assert "DIS-004" in TABLE_STRUCTURE_RULES
    pages = {
        17: (
            "عدد الموظفين حسب النشاط والموقع\n"
            "النشاط      الموقع      العدد\n"
            "البيع       عمان        120\n"
            "الإنتاج     الزرقاء     85\n"
            "الإدارة     عمان        30\n"
        )
    }
    finding = review_document("doc1", _req("DIS-004"), pages, {})
    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"


def test_dis004_needs_review_on_anchor_without_table_or_digits() -> None:
    pages = {
        17: (
            "يبلغ عدد الموظفين في الشركة اثنان وخمسون موظفًا موزعين على "
            "مختلف الإدارات والفروع."
        )
    }
    finding = review_document("doc1", _req("DIS-004"), pages, {})
    assert finding.result.status == "Needs Review"
    # Anchor is present so we still cite it as evidence; the finding stays
    # Needs Review because neither the whitespace signal nor the digit
    # signal met the table threshold.
    assert finding.result.evidence is not None
    assert finding.result.evidence.page == 17


def test_dis004_meets_on_anchor_with_dense_digit_cluster() -> None:
    # Real reports often render employee tables as a flat run of digits
    # after PDF extraction. The anchor + digit heuristic rescues this case.
    pages = {
        26: (
            "رابعًا: تطوير الموارد البشرية "
            "عدد الموظفين عدد الفروع والمكاتب "
            "97 108 111 105 111 100 109 105 108 111 "
            "2236 2440 2440 2335 2434 2437 2405 2439 2445 2414 "
            "2016 2017 2018 2019 2020 2021 2022 2023 2024 2025"
        )
    }
    finding = review_document("doc1", _req("DIS-004"), pages, {})
    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"
