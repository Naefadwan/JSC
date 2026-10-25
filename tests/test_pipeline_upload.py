"""Single-document pipeline helpers: sanitising, PDF validation, save flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from jsc_review.documents.pipeline import (
    UploadError,
    is_pdf_bytes,
    sanitize_document_id,
    save_uploaded_pdf,
)


class TestSanitizeDocumentId:
    """Filenames become safe document ids or raise UploadError."""

    def test_strips_pdf_suffix(self) -> None:
        assert sanitize_document_id("165_10164_900.pdf") == "165_10164_900"

    def test_preserves_dash_underscore_dot(self) -> None:
        assert sanitize_document_id("165-10164_900.v2.pdf") == "165-10164_900.v2"

    def test_replaces_unicode_and_spaces(self) -> None:
        assert sanitize_document_id("تقرير مالي 2026.pdf") == "2026"

    def test_rejects_empty_stem(self) -> None:
        with pytest.raises(UploadError):
            sanitize_document_id("")

    def test_rejects_names_with_no_safe_characters(self) -> None:
        with pytest.raises(UploadError):
            sanitize_document_id("??.pdf")


class TestIsPdfBytes:
    """A real PDF begins with ``%PDF-``."""

    def test_detects_pdf_header(self) -> None:
        assert is_pdf_bytes(b"%PDF-1.7\n...")

    def test_rejects_other_bytes(self) -> None:
        assert not is_pdf_bytes(b"PK\x03\x04 docx bytes")

    def test_rejects_empty_bytes(self) -> None:
        assert not is_pdf_bytes(b"")


class TestSaveUploadedPdf:
    """Uploads land in the reports directory and refuse duplicates."""

    def test_writes_the_pdf_using_the_sanitised_stem(self, tmp_path: Path) -> None:
        target = save_uploaded_pdf(
            data=b"%PDF-1.4\n%EOF\n",
            original_filename="rep.pdf",
            reports_dir=tmp_path,
        )
        assert target.name == "rep.pdf"
        assert target.read_bytes() == b"%PDF-1.4\n%EOF\n"

    def test_refuses_non_pdf_bytes(self, tmp_path: Path) -> None:
        with pytest.raises(UploadError):
            save_uploaded_pdf(
                data=b"not a pdf",
                original_filename="rep.pdf",
                reports_dir=tmp_path,
            )

    def test_refuses_duplicate_document_id(self, tmp_path: Path) -> None:
        save_uploaded_pdf(
            data=b"%PDF-1.4\n%EOF\n",
            original_filename="rep.pdf",
            reports_dir=tmp_path,
        )
        with pytest.raises(UploadError, match="already exists"):
            save_uploaded_pdf(
                data=b"%PDF-1.4\n%EOF\n",
                original_filename="rep.pdf",
                reports_dir=tmp_path,
            )

    def test_creates_the_reports_directory_when_missing(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "nested" / "reports"
        target = save_uploaded_pdf(
            data=b"%PDF-1.4\n%EOF\n",
            original_filename="rep.pdf",
            reports_dir=target_dir,
        )
        assert target.parent == target_dir
        assert target.exists()
