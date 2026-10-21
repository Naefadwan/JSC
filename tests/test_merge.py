"""Tests for merging OCR pages into heading chunks."""

from jsc_review.documents.chunk import TextChunk
from jsc_review.documents.merge import merge_ocr_chunks
from jsc_review.documents.ocr import ocr_page_record


def _chunk(page: int, text: str = "Embedded heading text for this section.") -> TextChunk:
    return TextChunk(
        document_id="report",
        page=page,
        section="Board",
        chunk_id=f"report:p{page}:c01",
        text=text,
        extraction_method="pymupdf-text",
    )


def test_useful_ocr_replaces_embedded_chunks_on_that_page() -> None:
    ocr = ocr_page_record(
        document_id="report",
        page_number=3,
        text="Recovered Arabic text from the scanned page content.",
    )

    merged = merge_ocr_chunks([_chunk(2), _chunk(3)], [ocr])

    assert [chunk.page for chunk in merged] == [2, 3]
    assert merged[0].extraction_method == "pymupdf-text"
    assert merged[1].chunk_id == "report:p3:ocr"
    assert merged[1].extraction_method == "tesseract-ara+eng"
    assert merged[1].section == ""
    assert ocr.supports_verified_evidence is False


def test_short_ocr_does_not_replace_embedded_text() -> None:
    ocr = ocr_page_record(document_id="report", page_number=3, text="short")

    merged = merge_ocr_chunks([_chunk(3)], [ocr])

    assert len(merged) == 1
    assert merged[0].extraction_method == "pymupdf-text"
