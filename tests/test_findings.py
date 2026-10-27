"""Tests for requirement findings."""

from jsc_review.documents.cer_measurement import MeasurementStore, PageMeasurement
from jsc_review.retrieval.bm25 import Bm25Index, IndexedChunk, build_index
from jsc_review.review.findings import (
    LocatedPassage,
    Requirement,
    apply_model_judgment,
    locate_passage,
    review_document,
)


def _cer_store(
    document_id: str, page: int, *, cer: float, engine: str = "indra"
) -> MeasurementStore:
    return MeasurementStore(
        threshold=0.05,
        measurements=(
            PageMeasurement(
                document_id=document_id,
                page=page,
                engine=engine,
                cer=cer,
                wer=None,
                reference_chars=100,
                hypothesis_chars=100,
                verified_evidence_eligible=cer <= 0.05,
                measured_at="2026-10-25T00:00:00+00:00",
            ),
        ),
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


def test_heading_from_ocr_is_promoted_when_cer_is_measured_low() -> None:
    store = _cer_store("report", 2, cer=0.03)

    finding = review_document(
        "report",
        _requirement("DIS-064", "deterministic"),
        pages={2: ""},
        ocr_pages={2: "قائمة التدفقات النقدية للسنة المنتهية"},
        cer_store=store,
    )

    assert finding.result.status == "Met"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is True
    assert finding.result.evidence.chunk_id.endswith(":ocr")


def test_heading_from_ocr_stays_review_when_cer_too_high() -> None:
    store = _cer_store("report", 2, cer=0.3)

    finding = review_document(
        "report",
        _requirement("DIS-064", "deterministic"),
        pages={2: ""},
        ocr_pages={2: "قائمة التدفقات النقدية للسنة المنتهية"},
        cer_store=store,
    )

    assert finding.result.status == "Needs Review"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is False


def test_language_requirement_shows_a_passage_and_stays_needs_review() -> None:
    # DIS-999 is not covered by any deterministic rule, so review_document
    # falls through to the retrieval-only path (no local model is loaded)
    # and emits a Needs-Review recommendation with the top passage attached.
    requirement = Requirement(
        requirement_id="DIS-999",
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


def test_model_met_on_ocr_passage_is_accepted_when_cer_is_measured_low() -> None:
    ocr_text = "قائمة التدفقات النقدية الموحدة للسنة المنتهية في ٣١ كانون الأول"
    passage = LocatedPassage(
        page=30,
        section="",
        text=ocr_text,
        extraction_method="tesseract-ara+eng",
        chunk_id="report:p30:ocr",
        passage_is_ocr=True,
    )
    requirement = Requirement(
        requirement_id="DIS-064",
        section="بند ج",
        text_ar="قائمة التدفقات النقدية",
        text_en="Statement of cash flows",
        evaluation_method="llm",
    )

    accepted = apply_model_judgment(
        "report",
        requirement,
        passage,
        status="Met",
        reason="The model quoted the heading.",
        quote="قائمة التدفقات النقدية",
        confidence=0.86,
        pages={30: ""},
        ocr_pages={30: ocr_text},
        cer_store=_cer_store("report", 30, cer=0.03),
    )

    assert accepted.status == "Met"
    assert accepted.evidence is not None
    assert accepted.evidence.verified is True


def test_model_met_on_ocr_passage_is_rejected_when_no_measurement_exists() -> None:
    ocr_text = "قائمة التدفقات النقدية الموحدة للسنة المنتهية في ٣١ كانون الأول"
    passage = LocatedPassage(
        page=30,
        section="",
        text=ocr_text,
        extraction_method="tesseract-ara+eng",
        chunk_id="report:p30:ocr",
        passage_is_ocr=True,
    )
    requirement = Requirement(
        requirement_id="DIS-064",
        section="بند ج",
        text_ar="قائمة التدفقات النقدية",
        text_en="Statement of cash flows",
        evaluation_method="llm",
    )

    rejected = apply_model_judgment(
        "report",
        requirement,
        passage,
        status="Met",
        reason="The model quoted the heading.",
        quote="قائمة التدفقات النقدية",
        confidence=0.86,
        pages={30: ""},
        ocr_pages={30: ocr_text},
    )

    assert rejected.status == "Needs Review"
    assert rejected.evidence is not None
    assert rejected.evidence.verified is False


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


def test_judge_requirement_passes_reranker_to_hybrid_search(
    monkeypatch,
) -> None:
    """When the caller opts the reranker in, judge_requirement threads it through."""

    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from jsc_review.review import findings as findings_module

    captured: list[object] = []

    def fake_hybrid_search(
        _requirement, _document_id, *, bm25, dense=None, reranker=None, config=None
    ):
        captured.append(reranker)
        return ()

    monkeypatch.setattr(findings_module, "hybrid_search", fake_hybrid_search)

    class _FakeReranker:
        def predict(self, _pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
            return np.asarray([], dtype=np.float32)

    requirement = _requirement("DIS-900", method="llm")
    reranker = _FakeReranker()

    # The call falls through to review_document (no passage was found), so the
    # reranker lands on locate_passage via judge_requirement and then review_document.
    findings_module.judge_requirement(
        "report",
        requirement,
        pages={1: "page text"},
        ocr_pages={},
        index=_index(),
        reranker=reranker,  # type: ignore[arg-type]
    )

    assert captured, "hybrid_search was never reached"
    assert any(entry is reranker for entry in captured)


def test_judge_requirement_defaults_to_no_reranker(monkeypatch) -> None:
    """Omitting the reranker leaves the retrieval pipeline rerank-free."""

    from jsc_review.review import findings as findings_module

    captured: list[object] = []

    def fake_hybrid_search(
        _requirement, _document_id, *, bm25, dense=None, reranker=None, config=None
    ):
        captured.append(reranker)
        return ()

    monkeypatch.setattr(findings_module, "hybrid_search", fake_hybrid_search)

    requirement = _requirement("DIS-901", method="llm")
    findings_module.judge_requirement(
        "report",
        requirement,
        pages={1: "page text"},
        ocr_pages={},
        index=_index(),
    )

    assert captured, "hybrid_search was never reached"
    assert all(entry is None for entry in captured)
