"""Single local PDF read used by inspection and extraction."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class RawTable:
    """One table as extracted, with blank cells removed from consideration later."""

    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RawPage:
    """Text and tables from one PDF page, before provenance is assigned."""

    text: str
    tables: tuple[RawTable, ...]


def read_pdf(path: Path) -> tuple[str, tuple[RawPage, ...]]:
    """Read page text, tables, and the document title. This does not run OCR."""

    document = pymupdf.open(path)  # type: ignore[no-untyped-call]
    try:
        page_count = int(document.page_count)
        pages = tuple(_read_page(document, index) for index in range(page_count))
        metadata = document.metadata or {}
        title = str(metadata.get("title") or "").strip()
    finally:
        document.close()  # type: ignore[no-untyped-call]
    return title, pages


def _read_page(document: pymupdf.Document, index: int) -> RawPage:
    page = document.load_page(index)  # type: ignore[no-untyped-call]
    text = str(page.get_text("text")).strip()
    return RawPage(text=text, tables=_read_tables(page))


def _read_tables(page: pymupdf.Page) -> tuple[RawTable, ...]:
    finder = page.find_tables()  # type: ignore[no-untyped-call]
    tables: list[RawTable] = []
    for table in finder.tables:
        rows = _normalize_rows(table.extract())
        if rows:
            tables.append(RawTable(rows=rows))
    return tuple(tables)


def _normalize_rows(extracted: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(extracted, list):
        return ()
    rows: list[tuple[str, ...]] = []
    for row in extracted:
        if not isinstance(row, list):
            continue
        cells = tuple("" if cell is None else str(cell).strip() for cell in row)
        if any(cells):
            rows.append(cells)
    return tuple(rows)
