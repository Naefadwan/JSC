"""Tests for heading chunks."""

from pathlib import Path

import pymupdf

from jsc_review.documents.chunk import chunk_pdf, chunks_from_lines


def test_larger_font_starts_a_section() -> None:
    chunks = chunks_from_lines(
        "report",
        2,
        [
            ("Board composition", 18.0),
            ("The board has seven members.", 11.0),
            ("Committee meetings", 18.0),
            ("The audit committee met four times.", 11.0),
        ],
    )

    assert [chunk.section for chunk in chunks] == ["Board composition", "Committee meetings"]
    assert chunks[0].chunk_id == "report:p2:c01"
    assert chunks[1].chunk_id == "report:p2:c02"
    assert "seven members" in chunks[0].text
    assert chunks[0].extraction_method == "pymupdf-text"


def test_uniform_text_stays_one_chunk() -> None:
    chunks = chunks_from_lines(
        "report",
        1,
        [("First sentence.", 11.0), ("Second sentence.", 11.0)],
    )

    assert len(chunks) == 1
    assert chunks[0].section == ""
    assert chunks[0].chunk_id == "report:p1:c01"


def test_pdf_heading_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "governance.pdf"
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        page = document.new_page()
        page.insert_text((72, 72), "Risk policy", fontsize=18)
        page.insert_text((72, 110), "The company adopted a risk management policy.", fontsize=11)
        document.save(path)  # type: ignore[no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]

    chunks = chunk_pdf(path).chunks

    assert len(chunks) == 1
    assert chunks[0].section == "Risk policy"
    assert "risk management policy" in chunks[0].text
