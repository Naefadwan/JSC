"""Tests for requirement findings."""

from jsc_review.retrieval.bm25 import Bm25Index, IndexedChunk, build_index
from jsc_review.review.findings import (
    LocatedPassage,
    Requirement,
    apply_model_judgment,
    locate_passage,
    review_document,
)


def _requirement(requirement_id: str, method: str = "llm") -> Requirement:
    return Requirement(
        requirement_id=requirement_id,
        section="بند ج",
        text_ar="قائمة التدفق النقدي",
        text_en="Cash flow statement",
        evaluation_method=method,
    )


def _index() -> Bm25Index:
    return build_index(
        [
            IndexedChunk(
                chunk_id="report:p3:c01",
                document_id="report",
                page=3,
                section="Risks",
                text="وصف للمخاطر التي تتعرض لها الشركة في السوق المحلي.",
                extraction_method="pymupdf-text",
            )
        ]
    )


def test_recognized_heading_is_met_with_verified_evidence() -> None:
    finding = review_document(
        "report",
        _requirement("DIS-064", "deterministic"),
        pages={4: "The cash flow statement for the year is presented below."},
        ocr_pages={},
    )

    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is True
    assert finding.result.evidence.page == 4


def test_missing_heading_stays_needs_review() -> None:
    finding = review_document(
        "report",
        _requirement("DIS-062", "deterministic"),
        pages={1: "Description of the company's activities."},
        ocr_pages={},
    )

    assert finding.result.status == "Needs Review"
    assert finding.result.evidence is None


def test_heading_found_only_by_ocr_cannot_be_met() -> None:
    finding = review_document(
        "report",
        _requirement("DIS-064", "deterministic"),
        pages={2: ""},
        ocr_pages={2: "قائمة التدفقات النقدية للسنة المنتهية"},
    )

    assert finding.result.status == "Needs Review"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is False


def test_language_requirement_shows_a_passage_and_stays_needs_review() -> None:
    requirement = Requirement(
        requirement_id="DIS-026",
        section="بند 10",
        text_ar="وصف للمخاطر التي تتعرض لها الشركة",
        text_en="Description of the risks to which the company is exposed.",
        evaluation_method="llm",
    )

    finding = review_document(
        "report",
        requirement,
        pages={3: "وصف للمخاطر التي تتعرض لها الشركة في السوق المحلي."},
        ocr_pages={},
        index=_index(),
    )

    assert finding.result.status == "Needs Review"
    assert finding.result.human_review == "Required"
    assert finding.result.evaluation_method == "llm"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is True


def test_model_met_requires_a_quote_on_the_embedded_page() -> None:
    page = "The board adopted a risk management policy for the company."
    passage = LocatedPassage(
        page=4,
        section="Risks",
        text=page,
        extraction_method="pymupdf-text",
        chunk_id="report:p4:c01",
        passage_is_ocr=False,
    )
    requirement = Requirement(
        requirement_id="GOV-001",
        section="Risks",
        text_ar="سياسة إدارة المخاطر",
        text_en="Adoption of a risk policy.",
        evaluation_method="llm",
    )

    accepted = apply_model_judgment(
        "report",
        requirement,
        passage,
        status="Met",
        reason="The passage states that the policy was adopted.",
        quote="adopted a risk management policy",
        confidence=0.8,
        pages={4: page},
    )
    rejected = apply_model_judgment(
        "report",
        requirement,
        passage,
        status="Met",
        reason="The model invented a quote.",
        quote="this sentence is not on the page",
        confidence=0.9,
        pages={4: page},
    )

    assert accepted.status == "Met"
    assert accepted.human_review == "Required"
    assert accepted.evaluation_method == "llm"
    assert rejected.status == "Needs Review"


def test_locate_passage_prefers_an_embedded_page_over_ocr() -> None:
    requirement = Requirement(
        requirement_id="DIS-026",
        section="بند 10",
        text_ar="وصف للمخاطر التي تتعرض لها الشركة",
        text_en="Description of the risks to which the company is exposed.",
        evaluation_method="llm",
    )
    index = build_index(
        [
            IndexedChunk(
                chunk_id="report:p2:ocr",
                document_id="report",
                page=2,
                section="",
                text="وصف للمخاطر التي تتعرض لها الشركة " * 4,
                extraction_method="tesseract-ara+eng",
            ),
            IndexedChunk(
                chunk_id="report:p5:c01",
                document_id="report",
                page=5,
                section="Risks",
                text="وصف للمخاطر التي تتعرض لها الشركة في السوق المحلي.",
                extraction_method="pymupdf-text",
            ),
        ]
    )

    passage = locate_passage(
        "report",
        requirement,
        pages={5: "وصف للمخاطر التي تتعرض لها الشركة في السوق المحلي."},
        index=index,
    )

    assert passage is not None
    assert passage.passage_is_ocr is False
    assert passage.page == 5
