"""Page-level text and table extraction with provenance."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from jsc_review.documents.inspection import TEXT_PAGE_MIN_CHARS
from jsc_review.documents.pdf_io import RawPage, read_pdf

TextMethod = Literal["pymupdf-text"]
TableMethod = Literal["pymupdf-table"]


@dataclass(frozen=True)
class ExtractedTable:
    """A table cited back to one page."""

    document_id: str
    page: int
    chunk_id: str
    extraction_method: TableMethod
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ExtractedPage:
    """Text and tables from one page. Short text still needs OCR."""

    document_id: str
    page: int
    section: str
    chunk_id: str
    text: str
    extraction_method: TextMethod
    needs_ocr: bool
    tables: tuple[ExtractedTable, ...]


@dataclass(frozen=True)
class ExtractedDocument:
    """A local PDF reduced to page-level records."""

    document_id: str
    path: Path
    title: str
    pages: tuple[ExtractedPage, ...]


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract text and tables. Pages below the text threshold are marked for OCR."""

    title, raw_pages = read_pdf(path)
    document_id = path.stem
    pages = tuple(
        _to_page(document_id, page_number, raw_page)
        for page_number, raw_page in enumerate(raw_pages, start=1)
    )
    return ExtractedDocument(document_id=document_id, path=path, title=title, pages=pages)


def save_extraction(extracted: ExtractedDocument, directory: Path) -> Path:
    """Write one JSON file. The caller chooses a directory that is not committed."""

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{extracted.document_id}.json"
    payload = {
        "document_id": extracted.document_id,
        "title": extracted.title,
        "source_path": extracted.path.as_posix(),
        "page_count": len(extracted.pages),
        "pages": [asdict(page) for page in extracted.pages],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _to_page(document_id: str, page_number: int, raw_page: RawPage) -> ExtractedPage:
    tables = tuple(
        ExtractedTable(
            document_id=document_id,
            page=page_number,
            chunk_id=f"{document_id}:p{page_number}:t{index:02d}",
            extraction_method="pymupdf-table",
            rows=table.rows,
        )
        for index, table in enumerate(raw_page.tables)
    )
    return ExtractedPage(
        document_id=document_id,
        page=page_number,
        section="",
        chunk_id=f"{document_id}:p{page_number}:c00",
        text=raw_page.text,
        extraction_method="pymupdf-text",
        needs_ocr=len(raw_page.text) < TEXT_PAGE_MIN_CHARS,
        tables=tables,
    )
