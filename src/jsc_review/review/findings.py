"""Requirement findings from local report text.

Language-model requirements are retrieved here. A separate local-model step
judges those passages. Ownership and customer thresholds are checked from
embedded page text.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jsc_review.core.models import Evidence, ReviewResult
from jsc_review.documents.cer_measurement import MeasurementStore
from jsc_review.documents.normalization import normalize_arabic, quote_matches_page
from jsc_review.retrieval.bm25 import Bm25Index
from jsc_review.retrieval.dense import DenseIndex
from jsc_review.retrieval.reranker import CrossEncoder
from jsc_review.retrieval.search import HybridConfig, hybrid_search
from jsc_review.review.thresholds import match_customer_disclosure, match_owner_disclosure

# A judge callable has the same shape as :func:`judge_passage`. It is passed
# as an argument so tests and the regression runner can inject deterministic
# stand-ins instead of calling the local model. ``Any`` avoids a circular
# import on ``ModelJudgment`` because ``ollama`` imports ``Requirement`` from
# this module; the lazy import inside :func:`judge_requirement` breaks the
# cycle at runtime.
JudgeCallable = Callable[..., Any]

_WHITESPACE = re.compile(r"\s+")

# Arabic prefix letters that commonly fuse to the start of a content word
# (lam, ba, kaf, fa, waw, sin, alef). Allowing 0-2 of them before each anchor
# lets us match ``للميزانية`` / ``وبيان`` / ``بالميزانية`` without listing
# every surface form in HEADING_VARIANTS.
_ARABIC_PREFIX_CHARS = "\u0644\u0628\u0643\u0641\u0648\u0633\u0627"
_ARABIC_RANGE = r"\u0600-\u06FF"


def _is_arabic(text: str) -> bool:
    """True when the string contains any Arabic script character."""

    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _anchor_pattern(anchor: str) -> re.Pattern[str]:
    """Build a case-folded regex that tolerates Arabic affix letters.

    Arabic anchors get a ``[ل ب ك ف و س ا]{0,2}`` prefix allowance and a
    non-Arabic-letter guard on both sides so the match is a word boundary
    even when the anchor fuses with surrounding words. Non-Arabic anchors
    use a plain word-boundary form that works for mixed Latin/Arabic pages.

    Special case: when the stem starts with ``ال`` (definite article), the
    leading ``ا`` orthographically drops after a ``لـ`` or ``بـ`` prefix,
    so ``لـ + الميزانية`` is written ``للميزانية`` (one alef, two lams).
    The pattern therefore also matches the stem with the article dropped
    when preceded by one of the valid prefix letters.
    """

    folded = anchor.casefold()
    escaped = re.escape(folded)
    full_form = rf"[{_ARABIC_PREFIX_CHARS}]{{0,2}}{escaped}"
    stem_form = ""
    # ``ال`` is alef (U+0627) followed by lam (U+0644).
    if _is_arabic(anchor) and anchor.startswith("\u0627\u0644"):
        stem_without_article = anchor[1:]  # drop the leading alef
        stem_escaped = re.escape(stem_without_article.casefold())
        # At least one prefix letter is required when the article is dropped,
        # otherwise we would match bare "لميزانية" which is not Arabic.
        stem_form = rf"|[{_ARABIC_PREFIX_CHARS}]{{1,2}}{stem_escaped}"
    if _is_arabic(anchor):
        pattern = (
            rf"(?:(?<![{_ARABIC_RANGE}])|^)"
            rf"(?:{full_form}{stem_form})"
            rf"(?:(?![{_ARABIC_RANGE}])|$)"
        )
    else:
        pattern = rf"\b{escaped}\b"
    return re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)


# -------------------------------------------------- deterministic rules ---
#
# Each requirement in HEADING_VARIANTS is Met when any of its anchor phrases
# appears on an embedded page (and, when CER measurements allow, on an OCR
# page). Entries live here so the UI register can flip a row to
# ``evaluation_method=deterministic`` and the review loop picks it up without
# a code change in production code paths.
HEADING_VARIANTS: dict[str, tuple[str, ...]] = {
    "DIS-062": (
        "statement of financial position",
        "consolidated statement of financial position",
        "balance sheet",
        "قائمة المركز المالي",
        "قائمة المركز المالي الموحدة",
        "بيان المركز المالي",
        "الميزانية العمومية",
        "الميزانية العامة",
    ),
    "DIS-064": (
        "statement of cash flows",
        "consolidated statement of cash flows",
        "cash flow statement",
        "cash flows",
        "قائمة التدفقات النقدية",
        "قائمة التدفقات النقدية الموحدة",
        "قائمة التدفق النقدي",
        "بيان التدفقات النقدية",
        "بيان التدفق النقدي",
    ),
    # GOV-002: adoption of a professional code of conduct for the board
    # and executives. The regulatory text asks the report to show the code
    # has been adopted and is kept current. Deterministic coverage is any
    # of the recognized headings for the code or an ethics charter.
    "GOV-002": (
        "مدونة السلوك المهني",
        "مدونة السلوك",
        "مدونة قواعد السلوك",
        "ميثاق الأخلاقيات",
        "ميثاق السلوك المهني",
        "ميثاق سلوك",
        "ميثاق أخلاقيات العمل",
        "قواعد السلوك المهني",
        "code of conduct",
        "code of ethics",
        "code of professional conduct",
        "professional conduct code",
        "business ethics",
    ),
    # Phase C: move GOV-006 (governance chapter) from EARLY_PAGE_RULES to
    # a presence-only check — diagnostic shows the governance chapter in
    # live reports starts around page 70+, so a page restriction is wrong.
    "GOV-006": (
        "قواعد حوكمة الشركات",
        "دليل قواعد حوكمة الشركات",
        "حوكمة الشركات المساهمة",
        "تعليمات حوكمة الشركات",
        "نظام الحوكمة",
        "إطار حوكمة الشركة",
        "corporate governance rules",
        "corporate governance code",
        "corporate governance framework",
    ),
    # Phase B strong conversions.
    "GOV-007": (
        "board of directors",
        "members of the board",
        "board composition",
        "chairman and members of the board",
        "أعضاء مجلس الإدارة",
        "تشكيلة مجلس الإدارة",
        "تشكيل مجلس الإدارة",
        "تركيبة مجلس الإدارة",
        "أعضاء المجلس الحاليين",
        "رئيس وأعضاء مجلس الإدارة",
        "رئيس مجلس الإدارة ونائب رئيس مجلس الإدارة",
        "رئيس مجلس إدارة الشركة",
        "مجلس إدارة الشركة",
    ),
    "GOV-011": (
        "governance liaison",
        "governance liaison officer",
        "corporate governance officer",
        "ضابط ارتباط الحوكمة",
        "ضابط ارتباط حوكمة",
        "مسؤول ارتباط الحوكمة",
    ),
    "GOV-015": (
        "committee meetings",
        "number of committee meetings",
        "meetings of the committees",
        "اجتماعات اللجان",
        "عدد اجتماعات اللجان",
        "اجتماعات لجنة التدقيق",
        "اجتماعات لجنة الترشيحات",
        "اجتماعات لجنة المكافآت",
        "اجتماعات لجنة الحوكمة",
    ),
    "DIS-039": (
        "securities owned by board members",
        "shares owned by board members",
        "directors' shareholdings",
        "أسهم أعضاء مجلس الإدارة",
        "ملكية أعضاء مجلس الإدارة للأوراق المالية",
        "ملكية أعضاء المجلس للأوراق المالية",
        "ملكية أعضاء مجلس الإدارة من الأوراق المالية",
        # Phrasings observed in live reports (diagnostic 2026-10-27):
        "عدد الأوراق المالية المملوكة",
        "الأوراق المالية المملوكة من قبل أعضاء مجلس الإدارة",
        "عدد الأسهم التي يمتلكها",
        "عدد الأسهم التي يمتلكها رئيس وأعضاء مجلس",
        "األوراق المالية المملوكة",
        "األسهم التي يمتلكها",
    ),
    "DIS-058": (
        "going concern",
        "no material matters",
        "استمرارية الشركة",
        "استمرارية البنك",
        "استمرار الشركة",
        "لا توجد أي أمور جوهرية",
        "عدم وجود أمور جوهرية",
        "لا توجد أية أمور جوهرية",
        "أمور جوهرية قد تؤثر على استمرارية الشركة",
        # Live-report phrasings that short-stop before "تؤثر" so the trailing
        # letter boundary still matches when the extraction glues the next
        # word (e.g. "تؤثرعلى") against it.
        "بعدم وجود أي أمور جوهرية",
        "بعدم وجود أية أمور جوهرية",
        "أي أمور جوهرية قد تؤثر",
        "أية أمور جوهرية قد تؤثر",
    ),
    # Phase B presence-only conversions.
    "GOV-001": (
        "risk management policy",
        "policy for risk management",
        "سياسة إدارة المخاطر",
        "سياسة إدارة مخاطر",
        "اعتماد سياسة إدارة المخاطر",
    ),
    "DIS-001": (
        "principal activities",
        "main activities",
        "nature of business",
        "company activities",
        "أنشطة الشركة الرئيسية",
        "الأنشطة الرئيسية للشركة",
        "النشاطات الرئيسية",
        "طبيعة أعمال الشركة",
        "نشاط الشركة",
        "أعمال الشركة",
        "نبذة عن الشركة",
        "رؤية الشركة ومهمتها",
        "رؤية الشركة",
    ),
    "DIS-026": (
        "principal risks",
        "risk factors",
        "risks facing the company",
        "risk management",
        "المخاطر التي تواجه الشركة",
        "المخاطر التي تتعرض لها الشركة",
        "وصف المخاطر",
        "أنواع المخاطر",
        "إدارة المخاطر",
        "لجنة إدارة المخاطر",
        "سياسة إدارة المخاطر",
        "مخاطر الشركة",
    ),
    "DIS-034": (
        "financial position analysis",
        "operating results",
        "management discussion and analysis",
        "analysis of financial position",
        "تحليل المركز المالي",
        "تحليل المركز المالي ونتائج الأعمال",
        "نتائج أعمال الشركة",
        "تحليل نتائج الأعمال",
        "تحليل نتائج أعمال الشركة",
        "نتائج أعمال الشركة ومركزها المالي",
        "تحليل نتائج أعمال الشركة ومركزها المالي",
    ),
    "DIS-056": (
        "related party transactions",
        "transactions with related parties",
        "related parties",
        "related party",
        "عقود مع الأطراف ذات العلاقة",
        "معاملات الأطراف ذات العلاقة",
        "العقود والمشاريع والارتباطات",
        "صفقات الأطراف ذات العلاقة",
        "الأطراف ذات العلاقة",
        "ارتباطات الشركة مع رئيس وأعضاء مجلس",
        # Live-report phrasings observed in financial-note sections:
        "الجهات ذات علاقة",
        "الجهات ذات العلاقة",
        "جهات ذات علاقة",
        "جهات ذات العلاقة",
        "الأرصدة مع الجهات ذات علاقة",
        "ملخص الأرصدة مع الجهات",
        "المعاملات مع الأطراف ذات العلاقة",
    ),
    # DIS-017 (customer / supplier concentration disclosure). The threshold
    # rule in review/thresholds.py requires an explicit percentage figure,
    # which fails on the common "no customers or suppliers above the 10%
    # threshold" disclosure — that is still a valid disclosure and should
    # count as Met. The heading-presence anchors catch both the "yes, we
    # have X major customers" phrasing and the "no major concentration"
    # phrasing. The dispatch runs HEADING_VARIANTS before THRESHOLD_RULES,
    # so DIS-017 is handled here now.
    "DIS-017": (
        "درجة الاعتماد على موردين",
        "درجة الاعتماد على عملاء",
        "موردين محددين أو عملاء رئيسيين",
        "موردين رئيسيين أو عملاء",
        "لا يوجد موردين محددين أو عملاء",
        "لا يوجد موردين محددين",
        "لا يوجد عملاء رئيسيين",
        "اعتماد الشركة على موردين",
        "اعتماد الشركة على عملاء",
        "تركز العملاء",
        "تركز الموردين",
        "customer concentration",
        "supplier concentration",
        "major customers",
        "major suppliers",
        "dependence on customers",
        "dependence on suppliers",
    ),
    # DIS-067 (auditor's report present): the regulatory requirement is that
    # an auditor's report accompanies the filing. Earlier multi-anchor logic
    # also required the report to cite the disclosure instructions, which
    # real auditor's reports do not: they cite auditing standards (ISA).
    # Presence of the auditor's report heading is the sound deterministic
    # test; its contents are reviewed elsewhere.
    "DIS-067": (
        "تقرير مدققي الحسابات",
        "تقرير مدقق الحسابات",
        "تقرير مدققي الحسابات المستقلين",
        "تقرير حول تدقيق القوائم المالية",
        "independent auditor's report",
        "independent auditors' report",
        "independent auditor",
        "report of the independent auditor",
    ),
}

# Multi-anchor rule: Met requires every listed anchor to appear on the
# same page. Reserved for genuinely conjunctive requirements where any
# single anchor would be too weak to prove compliance.
MULTI_ANCHOR_RULES: dict[str, tuple[tuple[str, ...], ...]] = {}

# Early-pages rule: Met requires the anchor within the first N pages of the
# report. Reserved for truly front-matter requirements. GOV-006 was moved
# to HEADING_VARIANTS because the governance chapter in live reports sits
# past page 70 — a page restriction was the wrong constraint.
EARLY_PAGE_RULES: dict[str, tuple[tuple[str, ...], int]] = {}

# Table-structure rule: Met requires the anchor plus a nearby tabular
# structure (DIS-004 asks for employees per activity and location, which
# almost always appears as a table rather than prose).
TABLE_STRUCTURE_RULES: dict[str, tuple[tuple[str, ...], int, int]] = {
    "DIS-004": (
        (
            "عدد الموظفين",
            "number of employees",
            "عدد موظفي الشركة",
            "staff count",
        ),
        2,   # minimum columns (counted as runs of 2+ spaces or tab-like gaps)
        3,   # minimum rows that look tabular
    ),
}


THRESHOLD_RULES = frozenset({"DIS-010", "DIS-011", "DIS-017"})


@dataclass(frozen=True)
class Requirement:
    """One selected checklist requirement."""

    requirement_id: str
    section: str
    text_ar: str
    text_en: str
    evaluation_method: str


@dataclass(frozen=True)
class DocumentFinding:
    """A schema-valid recommendation for one requirement in one report."""

    document_id: str
    result: ReviewResult


@dataclass(frozen=True)
class LocatedPassage:
    """The passage that will be shown or sent to the local model."""

    page: int
    section: str
    text: str
    extraction_method: str
    chunk_id: str
    passage_is_ocr: bool


def review_document(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    index: Bm25Index | None = None,
    *,
    dense: DenseIndex | None = None,
    reranker: CrossEncoder | None = None,
    cer_store: MeasurementStore | None = None,
) -> DocumentFinding:
    """Review one requirement against one report's extracted pages.

    When ``dense`` is given, retrieval fuses BM25 and dense rankings with
    reciprocal rank fusion. When ``dense`` is ``None`` behavior is identical
    to the BM25-only path. When ``reranker`` is given, the top candidates
    from fusion are re-scored by the cross-encoder; production leaves this
    ``None`` because the measured gate shows the reranker lowers top-k
    recall on the current label set. When ``cer_store`` is given, OCR
    evidence on pages whose measured CER is at or below the policy
    threshold may support ``Met``.
    """

    if requirement.requirement_id in HEADING_VARIANTS:
        result = _review_heading(document_id, requirement, pages, ocr_pages, cer_store=cer_store)
        # Fallthrough: when a requirement sits in both HEADING_VARIANTS and
        # THRESHOLD_RULES (e.g. DIS-017), the heading check catches the
        # "no major concentration" disclosure and the threshold check
        # catches the "customer X at 12% of sales" disclosure. If the
        # heading pass does not reach Met, try the threshold path too.
        if (
            result.status != "Met"
            and requirement.requirement_id in THRESHOLD_RULES
        ):
            threshold_result = _review_threshold(
                document_id, requirement, pages, ocr_pages, cer_store=cer_store
            )
            if threshold_result.status == "Met":
                result = threshold_result
    elif requirement.requirement_id in THRESHOLD_RULES:
        result = _review_threshold(document_id, requirement, pages, ocr_pages, cer_store=cer_store)
    elif requirement.requirement_id in MULTI_ANCHOR_RULES:
        result = _review_multi_anchor(
            document_id, requirement, pages, ocr_pages, cer_store=cer_store
        )
    elif requirement.requirement_id in EARLY_PAGE_RULES:
        result = _review_early_page(
            document_id, requirement, pages, ocr_pages, cer_store=cer_store
        )
    elif requirement.requirement_id in TABLE_STRUCTURE_RULES:
        result = _review_table_structure(
            document_id, requirement, pages, ocr_pages, cer_store=cer_store
        )
    else:
        result = _review_retrieved(
            document_id,
            requirement,
            pages,
            index,
            dense=dense,
            reranker=reranker,
            judged=False,
            reason_found=(
                "يُعرض أقرب مقطع. لا يتوفر نموذج محلي للحكم على ما إذا كان المتطلب مستوفى."
            ),
        )
    return DocumentFinding(document_id=document_id, result=result)


def _review_threshold(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    *,
    cer_store: MeasurementStore | None = None,
) -> ReviewResult:
    if requirement.requirement_id == "DIS-017":
        match = match_customer_disclosure(
            pages, ocr_pages, document_id=document_id, cer_store=cer_store
        )
    else:
        match = match_owner_disclosure(
            pages,
            ocr_pages,
            require_share_count=requirement.requirement_id == "DIS-011",
            document_id=document_id,
            cer_store=cer_store,
        )
    evidence = None
    if match.page is not None and match.excerpt:
        evidence = _evidence(
            document_id,
            match.page,
            requirement.section,
            match.excerpt,
            verified=match.verified,
            chunk_suffix="ocr" if match.from_ocr else "c00",
        )
    return _result(
        requirement,
        status=match.status,
        reason=match.reason,
        evidence=evidence,
        confidence=match.confidence,
        evaluation_method="deterministic",
    )


def _review_heading(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    *,
    cer_store: MeasurementStore | None = None,
) -> ReviewResult:
    variants = HEADING_VARIANTS[requirement.requirement_id]
    found = _find_variant(pages, variants)
    if found is not None:
        page_number, variant, excerpt = found
        return _result(
            requirement,
            status="Met",
            reason=f"تم التعرّف على عنوان القائمة التالي: {variant}.",
            evidence=_evidence(
                document_id,
                page_number,
                requirement.section,
                excerpt,
                verified=True,
            ),
            confidence=0.9,
            evaluation_method="deterministic",
        )
    ocr_found = _find_variant(ocr_pages, variants)
    if ocr_found is not None:
        page_number, variant, excerpt = ocr_found
        if cer_store is not None and cer_store.allows_verified_evidence(document_id, page_number):
            justification = cer_store.verification_reason(document_id, page_number) or ""
            return _result(
                requirement,
                status="Met",
                reason=(
                    f"تم التعرّف على عنوان القائمة التالي من نص OCR: {variant}. "
                    f"{justification}"
                ).strip(),
                evidence=_evidence(
                    document_id,
                    page_number,
                    requirement.section,
                    excerpt,
                    verified=True,
                    chunk_suffix="ocr",
                ),
                confidence=0.85,
                evaluation_method="deterministic",
            )
        return _result(
            requirement,
            status="Needs Review",
            reason=(
                f"ظهر العنوان '{variant}' في نص OCR فقط. "
                "لا يمكن اعتماد نص OCR دليلاً حتى ينخفض معدل الخطأ إلى 5٪ أو أقل."
            ),
            evidence=_evidence(
                document_id,
                page_number,
                requirement.section,
                excerpt,
                verified=False,
                chunk_suffix="ocr",
            ),
            confidence=0.4,
            evaluation_method="deterministic",
        )
    return _result(
        requirement,
        status="Needs Review",
        reason=(
            "لم يُعثر على عنوان قائمة معترف به. تحتاج الصياغة غير المألوفة إلى "
            "مراجعة بشرية بدلاً من اعتبارها عدم استيفاء مؤكد."
        ),
        evidence=None,
        confidence=0.3,
        evaluation_method="deterministic",
    )


def _review_multi_anchor(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    *,
    cer_store: MeasurementStore | None = None,
) -> ReviewResult:
    """Met when every anchor group hits the same page (DIS-067 style)."""

    anchor_groups = MULTI_ANCHOR_RULES[requirement.requirement_id]
    found = _find_variants_on_same_page(pages, anchor_groups)
    if found is not None:
        page_number, excerpt = found
        return _result(
            requirement,
            status="Met",
            reason=(
                "تم التحقق من حضور كل العبارات المطلوبة على الصفحة ذاتها، "
                f"وهي الصفحة رقم {page_number}."
            ),
            evidence=_evidence(
                document_id,
                page_number,
                requirement.section,
                excerpt,
                verified=True,
            ),
            confidence=0.9,
            evaluation_method="deterministic",
        )
    ocr_found = _find_variants_on_same_page(ocr_pages, anchor_groups)
    if ocr_found is not None:
        page_number, excerpt = ocr_found
        if cer_store is not None and cer_store.allows_verified_evidence(document_id, page_number):
            justification = cer_store.verification_reason(document_id, page_number) or ""
            return _result(
                requirement,
                status="Met",
                reason=(
                    "تم التحقق من حضور كل العبارات المطلوبة على الصفحة ذاتها من نص OCR "
                    f"(الصفحة {page_number}). {justification}"
                ).strip(),
                evidence=_evidence(
                    document_id,
                    page_number,
                    requirement.section,
                    excerpt,
                    verified=True,
                    chunk_suffix="ocr",
                ),
                confidence=0.85,
                evaluation_method="deterministic",
            )
        return _result(
            requirement,
            status="Needs Review",
            reason=(
                "جميع العبارات المطلوبة ظهرت على نفس صفحة OCR، ولكن معدل خطأ OCR "
                "لا يسمح باعتماده دليلاً حتى الآن."
            ),
            evidence=_evidence(
                document_id,
                page_number,
                requirement.section,
                excerpt,
                verified=False,
                chunk_suffix="ocr",
            ),
            confidence=0.4,
            evaluation_method="deterministic",
        )
    return _result(
        requirement,
        status="Needs Review",
        reason=(
            "لم تتجمع جميع العبارات المطلوبة على صفحة واحدة. "
            "تحتاج الصياغة إلى مراجعة بشرية."
        ),
        evidence=None,
        confidence=0.3,
        evaluation_method="deterministic",
    )


def _review_early_page(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    *,
    cer_store: MeasurementStore | None = None,
) -> ReviewResult:
    """Met when an anchor is found within the first N embedded pages."""

    variants, max_page = EARLY_PAGE_RULES[requirement.requirement_id]
    found = _find_variant_within(pages, variants, max_page=max_page)
    if found is not None:
        page_number, variant, excerpt = found
        return _result(
            requirement,
            status="Met",
            reason=(
                f"تم التعرّف على عنوان الفصل التالي ضمن الصفحات الأولى: {variant} "
                f"(الصفحة {page_number})."
            ),
            evidence=_evidence(
                document_id,
                page_number,
                requirement.section,
                excerpt,
                verified=True,
            ),
            confidence=0.9,
            evaluation_method="deterministic",
        )
    ocr_found = _find_variant_within(ocr_pages, variants, max_page=max_page)
    if ocr_found is not None:
        page_number, _variant, excerpt = ocr_found
        if cer_store is not None and cer_store.allows_verified_evidence(document_id, page_number):
            justification = cer_store.verification_reason(document_id, page_number) or ""
            return _result(
                requirement,
                status="Met",
                reason=(
                    f"تم التعرّف على عنوان الفصل من نص OCR على الصفحة {page_number}. "
                    f"{justification}"
                ).strip(),
                evidence=_evidence(
                    document_id,
                    page_number,
                    requirement.section,
                    excerpt,
                    verified=True,
                    chunk_suffix="ocr",
                ),
                confidence=0.85,
                evaluation_method="deterministic",
            )
    return _result(
        requirement,
        status="Needs Review",
        reason=(
            "لم يُعثر على فصل مخصص لقواعد الحوكمة ضمن الصفحات الأولى من التقرير. "
            "تحتاج المسألة إلى مراجعة بشرية."
        ),
        evidence=None,
        confidence=0.3,
        evaluation_method="deterministic",
    )


def _review_table_structure(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    *,
    cer_store: MeasurementStore | None = None,
) -> ReviewResult:
    """Met when an anchor is on a page that also contains a tabular block."""

    variants, min_cols, min_rows = TABLE_STRUCTURE_RULES[requirement.requirement_id]
    # Pass 1: embedded pages that have both the anchor and a table-shaped region.
    for page_number in sorted(pages):
        text = pages[page_number]
        if _find_variant({page_number: text}, variants) is None:
            continue
        if _has_table_structure(text, min_cols=min_cols, min_rows=min_rows):
            match = _find_variant({page_number: text}, variants)
            assert match is not None
            _, variant, excerpt = match
            return _result(
                requirement,
                status="Met",
                reason=(
                    f"تم التعرّف على عبارة {variant} مع بنية جدولية على الصفحة {page_number}."
                ),
                evidence=_evidence(
                    document_id,
                    page_number,
                    requirement.section,
                    excerpt,
                    verified=True,
                ),
                confidence=0.88,
                evaluation_method="deterministic",
            )
    # Pass 2: anchor-only match stays Needs Review (presence without the table
    # is not enough for Met, matching DIS-004 wording).
    anchor_only = _find_variant(pages, variants)
    if anchor_only is not None:
        page_number, variant, excerpt = anchor_only
        return _result(
            requirement,
            status="Needs Review",
            reason=(
                f"عُثر على عبارة {variant} على الصفحة {page_number}، لكن لم "
                "تُلتقط بنية جدولية لعدد الموظفين حسب النشاط والموقع. "
                "تحتاج المسألة إلى مراجعة بشرية."
            ),
            evidence=_evidence(
                document_id,
                page_number,
                requirement.section,
                excerpt,
                verified=True,
            ),
            confidence=0.5,
            evaluation_method="deterministic",
        )
    # OCR fallback mirrors the heading check.
    ocr_found = _find_variant(ocr_pages, variants)
    if ocr_found is not None:
        page_number, _variant, excerpt = ocr_found
        if cer_store is not None and cer_store.allows_verified_evidence(document_id, page_number):
            text = ocr_pages[page_number]
            if _has_table_structure(text, min_cols=min_cols, min_rows=min_rows):
                justification = cer_store.verification_reason(document_id, page_number) or ""
                return _result(
                    requirement,
                    status="Met",
                    reason=(
                        f"تم التعرّف على بنية الجدول من نص OCR على الصفحة {page_number}. "
                        f"{justification}"
                    ).strip(),
                    evidence=_evidence(
                        document_id,
                        page_number,
                        requirement.section,
                        excerpt,
                        verified=True,
                        chunk_suffix="ocr",
                    ),
                    confidence=0.82,
                    evaluation_method="deterministic",
                )
    return _result(
        requirement,
        status="Needs Review",
        reason=(
            "لم يُعثر على بيان عدد الموظفين حسب النشاط والموقع في نص الصفحات. "
            "تحتاج المسألة إلى مراجعة بشرية."
        ),
        evidence=None,
        confidence=0.3,
        evaluation_method="deterministic",
    )


def locate_passage(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    index: Bm25Index | None,
    *,
    dense: DenseIndex | None = None,
    reranker: CrossEncoder | None = None,
    search_config: HybridConfig | None = None,
) -> LocatedPassage | None:
    """Find the closest passage in one report.

    Uses the composed hybrid retrieval pipeline (BM25 + dense + RRF fusion +
    optional cross-encoder reranker + page-level deduplication) with
    requirement-driven query augmentation. The pipeline knobs are keyed by
    the requirement, not by the document or the company, so a new upload of
    any company is treated identically to the last.

    An embedded page is used when one ranks near the requirement. OCR text
    is used only when no embedded page is in that ranking.
    """

    if index is None:
        return None
    config = search_config or HybridConfig()
    hits = hybrid_search(
        requirement,
        document_id,
        bm25=index,
        dense=dense,
        reranker=reranker,
        config=config,
    )
    embedded = [hit for hit in hits if hit.extraction_method == "pymupdf-text"]
    chosen = embedded[0] if embedded else (hits[0] if hits else None)
    if chosen is None:
        return None
    page_text = pages.get(chosen.page, "")
    excerpt = _excerpt(page_text, requirement.text_ar) or chosen.text.strip()[:1200]
    if not excerpt:
        return None
    is_ocr = chosen.extraction_method != "pymupdf-text"
    text = excerpt if excerpt in page_text else chosen.text.strip()[:1200]
    return LocatedPassage(
        page=chosen.page,
        section=chosen.section or requirement.section,
        text=text,
        extraction_method=chosen.extraction_method,
        chunk_id=chosen.chunk_id,
        passage_is_ocr=is_ocr,
    )


def apply_model_judgment(
    document_id: str,
    requirement: Requirement,
    passage: LocatedPassage | None,
    *,
    status: str,
    reason: str,
    quote: str,
    confidence: float,
    pages: dict[int, str],
    ocr_pages: dict[int, str] | None = None,
    cer_store: MeasurementStore | None = None,
) -> ReviewResult:
    """Accept a local-model judgment only when its quote is on the cited page.

    When the passage came from OCR, the quote is accepted only if the OCR
    page has a measured CER at or below the policy threshold.
    """

    if passage is None:
        return _missing(requirement)
    page_text = pages.get(passage.page, "")
    quote_on_embedded = (
        bool(quote) and not passage.passage_is_ocr and quote_matches_page(quote, page_text)
    )
    quote_on_ocr = False
    if (
        bool(quote)
        and passage.passage_is_ocr
        and cer_store is not None
        and cer_store.allows_verified_evidence(document_id, passage.page)
    ):
        ocr_text = (ocr_pages or {}).get(passage.page, "")
        quote_on_ocr = quote_matches_page(quote, ocr_text)
    quote_on_page = quote_on_embedded or quote_on_ocr
    usable_status = status if status in {"Met", "Not Met", "Needs Review"} else "Needs Review"
    if usable_status in {"Met", "Not Met"} and not quote_on_page:
        usable_status = "Needs Review"
        if passage.passage_is_ocr and (
            cer_store is None or not cer_store.allows_verified_evidence(document_id, passage.page)
        ):
            reason = (
                "لم يُقبل حكم النموذج لأن المقطع من نص OCR وتعذّر التحقق منه، "
                "ولا يوجد قياس معتمد لمعدل خطأ OCR على هذه الصفحة. " + reason
            )
        else:
            reason = (
                "لم يُقبل حكم النموذج لأن الاقتباس لم يُتحقق منه في النص المضمَّن "
                "لصفحة التقرير. " + reason
            )
    evidence_text = quote if quote_on_page else passage.text
    return _result(
        requirement,
        status=usable_status,
        reason=reason,
        evidence=_evidence(
            document_id,
            passage.page,
            passage.section,
            evidence_text,
            verified=quote_on_page,
            chunk_id=passage.chunk_id,
        ),
        confidence=confidence,
        evaluation_method="llm",
    )


def _review_retrieved(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    index: Bm25Index | None,
    *,
    dense: DenseIndex | None = None,
    reranker: CrossEncoder | None = None,
    judged: bool,
    reason_found: str,
) -> ReviewResult:
    del judged
    passage = locate_passage(document_id, requirement, pages, index, dense=dense, reranker=reranker)
    if passage is None:
        return _missing(requirement)
    verified = not passage.passage_is_ocr and passage.text in pages.get(passage.page, "")
    return _result(
        requirement,
        status="Needs Review",
        reason=reason_found,
        evidence=_evidence(
            document_id,
            passage.page,
            passage.section,
            passage.text,
            verified=verified,
            chunk_id=passage.chunk_id,
        ),
        confidence=0.45,
        evaluation_method=_method(requirement),
    )


def _missing(requirement: Requirement) -> ReviewResult:
    return _result(
        requirement,
        status="Needs Review",
        reason=("لم يُعثر في النص المضمَّن للتقرير على مقطع يرتبط بهذا المتطلب."),
        evidence=None,
        confidence=0.2,
        evaluation_method=_method(requirement),
    )


def _approx_raw_offset(raw_text: str, normalized_text: str, normalized_offset: int) -> int:
    """Scale a normalized character offset back to a raw-text offset.

    Normalization folds alef variants, strips hamza/tashkeel, and collapses
    whitespace, so the raw and normalized strings have slightly different
    lengths. The excerpt window is wide (60 before, 180 after), so an
    approximate offset is enough for the reviewer to see the matched phrase
    with surrounding context.
    """

    if not normalized_text:
        return 0
    ratio = len(raw_text) / len(normalized_text)
    return max(0, min(len(raw_text), int(round(normalized_offset * ratio))))


def _find_variant(
    pages: dict[int, str],
    variants: tuple[str, ...],
) -> tuple[int, str, str] | None:
    """Return the first (page, variant, excerpt) match across pages.

    Variants are matched by the affix-tolerant regex built in
    :func:`_anchor_pattern`, so an Arabic anchor like ``الميزانية`` also
    matches ``للميزانية``, ``بالميزانية``, and ``وبالميزانية`` without us
    needing to enumerate every prefixed form in the variants tuple.

    Both the anchor and the page text are passed through
    :func:`normalize_arabic` before matching, so a page that writes
    ``ادارة`` (without hamza) still matches an anchor written ``إدارة``.
    The returned excerpt is cut from the raw page text so the reviewer
    sees the original diacritics and spacing.
    """

    compiled = [
        (variant, _anchor_pattern(normalize_arabic(variant))) for variant in variants
    ]
    for page_number in sorted(pages):
        raw_text = pages[page_number]
        normalized_text = normalize_arabic(raw_text)
        for variant, pattern in compiled:
            match = pattern.search(normalized_text)
            if match is not None:
                raw_offset = _approx_raw_offset(raw_text, normalized_text, match.start())
                length = max(len(variant), match.end() - match.start())
                return page_number, variant, _window(raw_text, raw_offset, length)
    return None


def _find_variants_on_same_page(
    pages: dict[int, str],
    anchor_groups: tuple[tuple[str, ...], ...],
) -> tuple[int, str] | None:
    """Return a (page, excerpt) when every anchor group hits the same page.

    Each inner tuple is a set of synonyms; one hit per group is enough. The
    excerpt is pulled from around the first-group match on that page.
    Matching uses the normalized Arabic view just like :func:`_find_variant`.
    """

    compiled: list[list[re.Pattern[str]]] = [
        [_anchor_pattern(normalize_arabic(v)) for v in group] for group in anchor_groups
    ]
    for page_number in sorted(pages):
        raw_text = pages[page_number]
        normalized_text = normalize_arabic(raw_text)
        first_match: re.Match[str] | None = None
        all_present = True
        for patterns in compiled:
            match = next(
                (m for pattern in patterns if (m := pattern.search(normalized_text))), None
            )
            if match is None:
                all_present = False
                break
            if first_match is None:
                first_match = match
        if all_present and first_match is not None:
            length = max(1, first_match.end() - first_match.start())
            raw_offset = _approx_raw_offset(raw_text, normalized_text, first_match.start())
            return page_number, _window(raw_text, raw_offset, length)
    return None


def _find_variant_within(
    pages: dict[int, str],
    variants: tuple[str, ...],
    *,
    max_page: int,
) -> tuple[int, str, str] | None:
    """Match :func:`_find_variant` but ignore pages past ``max_page``."""

    subset = {page: text for page, text in pages.items() if page <= max_page}
    return _find_variant(subset, variants)


def _has_table_structure(
    text: str,
    *,
    min_cols: int,
    min_rows: int,
) -> bool:
    """True when ``text`` looks like a table of counts.

    Two signals, either of which qualifies:

    * At least ``min_rows`` lines contain ``min_cols + 1`` or more segments
      separated by runs of whitespace (tab, Arabic wide space, two or more
      regular spaces). This matches PDF text-extraction output that
      preserves column gaps as whitespace runs.
    * The page contains at least ``min_rows * (min_cols + 1)`` digit groups
      (Western 0-9 or Arabic-Indic ٠-٩). Arabic reports often strip
      inter-column whitespace during extraction, so a dense cluster of
      numbers near the anchor is the reliable fallback signal that the
      page is a table of counts rather than prose about counts.
    """

    row_hits = 0
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        segments = [seg for seg in re.split(r"[\t\u3000 ]{2,}", stripped) if seg]
        if len(segments) >= min_cols + 1:
            row_hits += 1
            if row_hits >= min_rows:
                return True
    # Fallback: count digit groups on the page. A line like "97 108 111 105"
    # loses its column gaps during PDF text extraction but still carries the
    # tabular signal through the digits themselves.
    digit_group_count = len(re.findall(r"[0-9\u0660-\u0669]+", text))
    required = min_rows * (min_cols + 1)
    return digit_group_count >= required


def _window(text: str, index: int, length: int) -> str:
    start = max(0, index - 60)
    end = min(len(text), index + length + 180)
    return text[start:end].strip()


def _excerpt(page_text: str, needle: str) -> str:
    if not page_text or not needle.strip():
        return ""
    folded = page_text.casefold()
    index = folded.find(needle.casefold())
    if index < 0:
        compact_page = _WHITESPACE.sub(" ", page_text)
        compact_needle = _WHITESPACE.sub(" ", needle).casefold()
        compact_index = compact_page.casefold().find(compact_needle[:40])
        if compact_index < 0:
            return ""
        return compact_page[compact_index : compact_index + 240].strip()
    return _window(page_text, index, min(len(needle), 80))


def _method(requirement: Requirement) -> str:
    if requirement.evaluation_method == "deterministic":
        return "deterministic"
    return "llm"


def _evidence(
    document_id: str,
    page_number: int,
    section: str,
    text: str,
    *,
    verified: bool,
    chunk_suffix: str = "c00",
    chunk_id: str = "",
) -> Evidence:
    return Evidence(
        text=text,
        source_document=document_id,
        page=page_number,
        section=section or f"Page {page_number}",
        chunk_id=chunk_id or f"{document_id}:p{page_number}:{chunk_suffix}",
        verified=verified,
    )


def _result(
    requirement: Requirement,
    *,
    status: str,
    reason: str,
    evidence: Evidence | None,
    confidence: float,
    evaluation_method: str,
) -> ReviewResult:
    human_review = "Required"
    if (
        evaluation_method == "deterministic"
        and status in {"Met", "Not Met"}
        and evidence is not None
        and evidence.verified
        and confidence >= 0.85
    ):
        human_review = "Not Required"
    return ReviewResult(
        requirement_id=requirement.requirement_id,
        status=status,  # type: ignore[arg-type]
        reason=reason,
        evidence=evidence,
        confidence=confidence,
        human_review=human_review,  # type: ignore[arg-type]
        evaluation_method=evaluation_method,  # type: ignore[arg-type]
    )


def judge_requirement(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    index: Bm25Index | None,
    *,
    dense: DenseIndex | None = None,
    reranker: CrossEncoder | None = None,
    cer_store: MeasurementStore | None = None,
    judge: JudgeCallable | None = None,
) -> ReviewResult:
    """Run the production per-requirement review and return the ``ReviewResult``.

    This is the single shared code path used by both the upload pipeline and
    the regression runner. For deterministic requirements it delegates to
    :func:`review_document`. For local-model requirements it runs retrieval,
    calls the injected ``judge`` (defaulting to :func:`judge_passage`), and
    verifies the quote through :func:`apply_model_judgment`. A model error is
    caught and the finding falls back to the retrieved passage with the error
    appended to the reason, which matches the upload pipeline's behavior.

    The ``judge`` parameter lets tests and the offline regression runner
    inject a deterministic stand-in for the local model so the function stays
    reproducible across runs.
    """

    # Lazy import to avoid the ``ollama`` → ``findings`` → ``ollama`` cycle.
    from jsc_review.review.ollama import LocalModelError, judge_passage

    judge_fn: JudgeCallable = judge if judge is not None else judge_passage

    if requirement.evaluation_method != "llm":
        finding = review_document(
            document_id,
            requirement,
            pages,
            ocr_pages,
            index,
            dense=dense,
            reranker=reranker,
            cer_store=cer_store,
        )
        return finding.result

    passage = locate_passage(document_id, requirement, pages, index, dense=dense, reranker=reranker)
    if passage is None:
        finding = review_document(
            document_id,
            requirement,
            pages,
            ocr_pages,
            index,
            dense=dense,
            reranker=reranker,
            cer_store=cer_store,
        )
        return finding.result

    try:
        judgment = judge_fn(requirement, passage.text, passage_is_ocr=passage.passage_is_ocr)
    except LocalModelError as exc:
        finding = review_document(
            document_id,
            requirement,
            pages,
            ocr_pages,
            index,
            dense=dense,
            reranker=reranker,
            cer_store=cer_store,
        )
        return finding.result.model_copy(
            update={"reason": f"{finding.result.reason} خطأ النموذج المحلي: {exc}"}
        )

    return apply_model_judgment(
        document_id,
        requirement,
        passage,
        status=judgment.status,
        reason=judgment.reason,
        quote=judgment.quote,
        confidence=judgment.confidence,
        pages=pages,
        ocr_pages=ocr_pages,
        cer_store=cer_store,
    )
