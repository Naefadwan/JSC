"""Add recovered OCR text to the heading-chunk set."""

from collections.abc import Sequence

from jsc_review.documents.chunk import TextChunk
from jsc_review.documents.inspection import TEXT_PAGE_MIN_CHARS
from jsc_review.documents.ocr import EXTRACTION_METHOD, OcrPage


def merge_ocr_chunks(
    chunks: Sequence[TextChunk],
    ocr_pages: Sequence[OcrPage],
    *,
    min_chars: int = TEXT_PAGE_MIN_CHARS,
) -> tuple[TextChunk, ...]:
    """Replace short embedded chunks on OCR pages with the recovered page text.

    OCR chunks stay ineligible for ``Met`` until an error rate is stored on the
    source record and meets the confirmed limit.
    """

    useful = [page for page in ocr_pages if len(page.text.strip()) >= min_chars]
    replaced = {(page.document_id, page.page) for page in useful}
    kept = [chunk for chunk in chunks if (chunk.document_id, chunk.page) not in replaced]
    added = [
        TextChunk(
            document_id=page.document_id,
            page=page.page,
            section="",
            chunk_id=f"{page.document_id}:p{page.page}:ocr",
            text=page.text.strip(),
            extraction_method=page.extraction_method or EXTRACTION_METHOD,
        )
        for page in useful
    ]
    return tuple([*kept, *added])
