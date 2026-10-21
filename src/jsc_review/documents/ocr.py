"""Local Tesseract OCR for pages whose embedded text is missing."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from jsc_review.documents.ocr_metrics import (
    character_error_rate,
    supports_verified_evidence,
    word_error_rate,
)

OCR_LANGUAGES = "ara+eng"
OCR_RENDER_DPI = 300
EXTRACTION_METHOD = "tesseract-ara+eng"


class OcrEngineError(RuntimeError):
    """Raised when local Tesseract cannot read a page."""


@dataclass(frozen=True)
class OcrPage:
    """OCR text for one page. Unmeasured error rates cannot support Met."""

    document_id: str
    page: int
    chunk_id: str
    text: str
    extraction_method: str
    character_error_rate: float | None
    word_error_rate: float | None
    supports_verified_evidence: bool


def find_tesseract() -> Path:
    """Locate the local tesseract executable without calling a network service."""

    configured = os.environ.get("JSC_TESSERACT_CMD", "").strip()
    if configured:
        path = Path(configured)
        if path.is_file():
            return path
        raise OcrEngineError(f"JSC_TESSERACT_CMD does not point to tesseract: {configured}")
    discovered = shutil.which("tesseract")
    if discovered:
        return Path(discovered)
    local = Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"
    if local.is_file():
        return local
    raise OcrEngineError("Tesseract was not found on PATH or in the local programs folder")


def recognize_image(image_path: Path, *, languages: str = OCR_LANGUAGES) -> str:
    """Run Tesseract on one image and return the recognized text."""

    executable = find_tesseract()
    completed = subprocess.run(
        [str(executable), str(image_path), "stdout", "-l", languages, "--psm", "3"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OcrEngineError(detail or f"Tesseract exited with status {completed.returncode}")
    return completed.stdout.decode("utf-8", errors="replace").strip()


def recognize_pdf_page(pdf_path: Path, page_number: int) -> str:
    """Render one 1-based PDF page and recognize it locally."""

    if page_number < 1:
        raise OcrEngineError("page numbers start at 1")
    document = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    try:
        if page_number > int(document.page_count):
            raise OcrEngineError(f"page {page_number} is outside {pdf_path.name}")
        page = document.load_page(page_number - 1)  # type: ignore[no-untyped-call]
        pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page.png"
            pixmap.save(image_path)
            return recognize_image(image_path)
    finally:
        document.close()  # type: ignore[no-untyped-call]


def score_embedded_page(
    pdf_path: Path,
    page_number: int,
) -> tuple[float | None, float | None]:
    """Compare OCR of a rendered page with its embedded text.

    This scores a digital re-render. It is not a ground-truth score for a scan.
    """

    reference = embedded_page_text(pdf_path, page_number)
    hypothesis = recognize_pdf_page(pdf_path, page_number)
    return (
        character_error_rate(reference, hypothesis),
        word_error_rate(reference, hypothesis),
    )


def embedded_page_text(pdf_path: Path, page_number: int) -> str:
    """Return the embedded text of one 1-based page."""

    if page_number < 1:
        raise OcrEngineError("page numbers start at 1")
    document = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    try:
        if page_number > int(document.page_count):
            raise OcrEngineError(f"page {page_number} is outside {pdf_path.name}")
        page = document.load_page(page_number - 1)  # type: ignore[no-untyped-call]
        return str(page.get_text("text")).strip()
    finally:
        document.close()  # type: ignore[no-untyped-call]


def ocr_page_record(
    *,
    document_id: str,
    page_number: int,
    text: str,
    character_error_rate: float | None = None,
    word_error_rate: float | None = None,
) -> OcrPage:
    """Build a page record. Missing error rates cannot support verified evidence."""

    return OcrPage(
        document_id=document_id,
        page=page_number,
        chunk_id=f"{document_id}:p{page_number}:c00",
        text=text,
        extraction_method=EXTRACTION_METHOD,
        character_error_rate=character_error_rate,
        word_error_rate=word_error_rate,
        supports_verified_evidence=supports_verified_evidence(character_error_rate),
    )
