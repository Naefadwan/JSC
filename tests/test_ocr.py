"""Tests for the local Tesseract adapter."""

from pathlib import Path

import pymupdf
import pytest

from jsc_review.documents.ocr import (
    EXTRACTION_METHOD,
    OcrEngineError,
    find_tesseract,
    ocr_page_record,
    recognize_image,
    score_embedded_page,
)


def test_unmeasured_page_record_cannot_support_met() -> None:
    record = ocr_page_record(document_id="report", page_number=4, text="نص")

    assert record.chunk_id == "report:p4:c00"
    assert record.extraction_method == EXTRACTION_METHOD
    assert record.character_error_rate is None
    assert record.supports_verified_evidence is False


def test_tesseract_reads_a_rendered_english_page(tmp_path: Path) -> None:
    try:
        find_tesseract()
    except OcrEngineError:
        pytest.skip("Tesseract is not installed")

    image_path = tmp_path / "page.png"
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        page = document.new_page()
        page.insert_text((72, 144), "The board adopted a risk policy.")
        pixmap = page.get_pixmap(dpi=200)
        pixmap.save(image_path)  # type: ignore[no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]

    text = recognize_image(image_path, languages="eng")

    assert "board" in text.lower()
    assert "policy" in text.lower()


def test_rendered_english_page_is_scored_against_embedded_text(tmp_path: Path) -> None:
    try:
        find_tesseract()
    except OcrEngineError:
        pytest.skip("Tesseract is not installed")

    path = tmp_path / "page.pdf"
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        page = document.new_page()
        page.insert_text((72, 144), "The board adopted a risk policy.")
        document.save(path)  # type: ignore[no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]

    character_rate, word_rate = score_embedded_page(path, 1)

    assert character_rate is not None
    assert word_rate is not None
    assert character_rate <= 0.05
