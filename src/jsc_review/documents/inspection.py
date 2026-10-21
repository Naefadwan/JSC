"""Local PDF inspection for page count, language, and text-or-scan class."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jsc_review.documents.pdf_io import read_pdf

PdfClass = Literal["text", "scan", "mixed"]
TEXT_PAGE_MIN_CHARS = 30
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class PdfInspection:
    """Measured properties of one local PDF."""

    document_id: str
    path: Path
    page_count: int
    title: str
    language: str
    text_page_count: int
    non_text_page_count: int
    classification: PdfClass


def inspect_pdf(path: Path) -> PdfInspection:
    """Classify a PDF from its extracted text. This does not run OCR."""

    title, raw_pages = read_pdf(path)
    page_texts = [page.text for page in raw_pages]

    text_pages = [text for text in page_texts if len(text) >= TEXT_PAGE_MIN_CHARS]
    text_page_count = len(text_pages)
    page_count = len(page_texts)
    non_text_page_count = page_count - text_page_count
    return PdfInspection(
        document_id=path.stem,
        path=path,
        page_count=page_count,
        title=title,
        language=detect_language("\n".join(text_pages)),
        text_page_count=text_page_count,
        non_text_page_count=non_text_page_count,
        classification=classify_pages(text_page_count, non_text_page_count),
    )


def classify_pages(text_page_count: int, non_text_page_count: int) -> PdfClass:
    """Classify a document from its text and non-text page counts."""

    if text_page_count == 0:
        return "scan"
    if non_text_page_count == 0:
        return "text"
    return "mixed"


def detect_language(text: str) -> str:
    """Return ar, en, ar-en, or an empty string when no letters were extracted."""

    has_arabic = _ARABIC.search(text) is not None
    has_latin = _LATIN.search(text) is not None
    if has_arabic and has_latin:
        return "ar-en"
    if has_arabic:
        return "ar"
    if has_latin:
        return "en"
    return ""
