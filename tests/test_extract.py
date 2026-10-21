"""Tests for page-level extraction and provenance."""

import json
from pathlib import Path

import pymupdf

from jsc_review.documents.extract import extract_pdf, save_extraction


def _save(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        for text in pages:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)  # type: ignore[no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]


def test_extract_assigns_page_provenance(tmp_path: Path) -> None:
    path = tmp_path / "annual-report.pdf"
    text = "The board adopted a risk management policy for the company."
    _save(path, [text, ""])

    extracted = extract_pdf(path)

    assert extracted.document_id == "annual-report"
    assert len(extracted.pages) == 2
    first, second = extracted.pages
    assert first.page == 1
    assert first.chunk_id == "annual-report:p1:c00"
    assert first.text == text
    assert first.extraction_method == "pymupdf-text"
    assert first.needs_ocr is False
    assert second.page == 2
    assert second.chunk_id == "annual-report:p2:c00"
    assert second.needs_ocr is True
    assert second.tables == ()


def test_extract_keeps_a_detected_table(tmp_path: Path) -> None:
    path = tmp_path / "holders.pdf"
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        page = document.new_page()
        page.draw_rect(pymupdf.Rect(72, 72, 300, 140))  # type: ignore[no-untyped-call]
        page.draw_line(pymupdf.Point(72, 106), pymupdf.Point(300, 106))  # type: ignore[no-untyped-call]
        page.draw_line(pymupdf.Point(186, 72), pymupdf.Point(186, 140))  # type: ignore[no-untyped-call]
        page.insert_text((80, 96), "Holder")
        page.insert_text((196, 96), "Shares")
        page.insert_text((80, 128), "Investor")
        page.insert_text((196, 128), "1000")
        document.save(path)  # type: ignore[no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]

    extracted = extract_pdf(path)
    tables = extracted.pages[0].tables

    assert len(tables) == 1
    assert tables[0].chunk_id == "holders:p1:t00"
    assert tables[0].extraction_method == "pymupdf-table"
    assert tables[0].page == 1
    flat = " ".join(cell for row in tables[0].rows for cell in row)
    assert "Holder" in flat
    assert "1000" in flat


def test_save_extraction_writes_local_json(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    _save(source, ["Names of the audit committee members are disclosed here."])

    target = save_extraction(extract_pdf(source), tmp_path / "out")
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert target.name == "report.json"
    assert payload["page_count"] == 1
    assert payload["pages"][0]["chunk_id"] == "report:p1:c00"
    assert payload["pages"][0]["extraction_method"] == "pymupdf-text"
