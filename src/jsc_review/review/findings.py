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
from jsc_review.documents.normalization import quote_matches_page
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
HEADING_VARIANTS: dict[str, tuple[str, ...]] = {
    "DIS-062": (
        "statement of financial position",
        "balance sheet",
        "قائمة المركز المالي",
        "بيان المركز المالي",
        "الميزانية العمومية",
        "الميزانية العامة",
    ),
    "DIS-064": (
        "statement of cash flows",
        "cash flow statement",
        "cash flows",
        "قائمة التدفقات النقدية",
        "قائمة التدفق النقدي",
        "بيان التدفقات النقدية",
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
    elif requirement.requirement_id in THRESHOLD_RULES:
        result = _review_threshold(document_id, requirement, pages, ocr_pages, cer_store=cer_store)
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
            cer = cer_store.cer_for(document_id, page_number) or 0.0
            return _result(
                requirement,
                status="Met",
                reason=(
                    f"تم التعرّف على عنوان القائمة التالي من نص OCR: {variant}. "
                    f"معدل خطأ OCR المقاس على هذه الصفحة {cer * 100:.2f}٪ وهو "
                    "ضمن الحد المسموح به."
                ),
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


def _find_variant(
    pages: dict[int, str],
    variants: tuple[str, ...],
) -> tuple[int, str, str] | None:
    for page_number in sorted(pages):
        text = pages[page_number]
        folded = text.casefold()
        for variant in variants:
            index = folded.find(variant.casefold())
            if index >= 0:
                return page_number, variant, _window(text, index, len(variant))
    return None


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
