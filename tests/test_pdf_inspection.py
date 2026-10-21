"""Tests for local PDF inspection."""

from pathlib import Path

import pymupdf

from jsc_review.documents.inspection import detect_language, inspect_pdf


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


def test_text_pdf_is_classified_as_text(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    _save(path, ["The board adopted a risk management policy for the company."])

    inspection = inspect_pdf(path)

    assert inspection.page_count == 1
    assert inspection.classification == "text"
    assert inspection.language == "en"
    assert inspection.document_id == "report"


def test_blank_pdf_is_classified_as_scan(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _save(path, [""])

    inspection = inspect_pdf(path)

    assert inspection.classification == "scan"
    assert inspection.language == ""
    assert inspection.non_text_page_count == 1


def test_mixed_pdf_reports_both_page_kinds(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdf"
    _save(
        path,
        [
            "Description of the company's main activities in the annual report.",
            "",
        ],
    )

    inspection = inspect_pdf(path)

    assert inspection.classification == "mixed"
    assert inspection.language == "en"
    assert inspection.text_page_count == 1
    assert inspection.non_text_page_count == 1


def test_language_detection_recognizes_arabic_and_english() -> None:
    assert detect_language("وصف لأنشطة الشركة") == "ar"
    assert detect_language("وصف activities") == "ar-en"
    assert detect_language("123") == ""
