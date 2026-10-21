"""Requirement findings from local report text.

Language-model requirements are retrieved here. A separate local-model step
judges those passages. Ownership and customer thresholds are checked from
embedded page text.
"""

import re
from dataclasses import dataclass

from jsc_review.core.models import Evidence, ReviewResult
from jsc_review.retrieval.bm25 import Bm25Index
from jsc_review.review.thresholds import match_customer_disclosure, match_owner_disclosure

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
) -> DocumentFinding:
    """Review one requirement against one report's extracted pages."""

    if requirement.requirement_id in HEADING_VARIANTS:
        result = _review_heading(document_id, requirement, pages, ocr_pages)
    elif requirement.requirement_id in THRESHOLD_RULES:
        result = _review_threshold(document_id, requirement, pages, ocr_pages)
    else:
        result = _review_retrieved(
            document_id,
            requirement,
            pages,
            index,
            judged=False,
            reason_found=(
                "The closest passage is shown. No local model is available to judge "
                "whether the requirement is met."
            ),
        )
    return DocumentFinding(document_id=document_id, result=result)


def _review_threshold(
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    ocr_pages: dict[int, str],
) -> ReviewResult:
    if requirement.requirement_id == "DIS-017":
        match = match_customer_disclosure(pages, ocr_pages)
    else:
        match = match_owner_disclosure(
            pages,
            ocr_pages,
            require_share_count=requirement.requirement_id == "DIS-011",
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
) -> ReviewResult:
    variants = HEADING_VARIANTS[requirement.requirement_id]
    found = _find_variant(pages, variants)
    if found is not None:
        page_number, variant, excerpt = found
        return _result(
            requirement,
            status="Met",
            reason=f"A recognized heading for this statement was found: {variant}.",
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
        return _result(
            requirement,
            status="Needs Review",
            reason=(
                f"The heading '{variant}' appeared only in OCR text. "
                "OCR text cannot support Met until its error rate is within 5%."
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
            "No recognized heading was found. Unfamiliar wording is Needs Review, "
            "not a confirmed miss."
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
) -> LocatedPassage | None:
    """Find the closest passage in one report.

    An embedded page is used when one ranks near the requirement. OCR text is
    used only when no embedded page is in that ranking.
    """

    if index is None:
        return None
    hits = index.search(
        f"{requirement.text_ar} {requirement.text_en}",
        k=100,
        document_id=document_id,
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
) -> ReviewResult:
    """Accept a local-model judgment only when its quote is on the cited page."""

    if passage is None:
        return _missing(requirement)
    page_text = pages.get(passage.page, "")
    quote_on_page = bool(quote) and quote in page_text and not passage.passage_is_ocr
    usable_status = status if status in {"Met", "Not Met", "Needs Review"} else "Needs Review"
    if usable_status in {"Met", "Not Met"} and not quote_on_page:
        usable_status = "Needs Review"
        reason = (
            "The model judgment was not accepted because its quote is not verified "
            "on the embedded report page. " + reason
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
    judged: bool,
    reason_found: str,
) -> ReviewResult:
    del judged
    passage = locate_passage(document_id, requirement, pages, index)
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
        reason="No passage in the embedded report text could be tied to this requirement.",
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
