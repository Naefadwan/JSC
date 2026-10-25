"""HTTP tests for the upload router.

Background processing is replaced with a lightweight stub so these tests
run in milliseconds. The stub records the job it was asked to run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jsc_review.api.app import create_app
from jsc_review.api.routes import uploads as uploads_module
from jsc_review.core.settings import Settings, get_settings
from jsc_review.reviewer.jobs import get_job_store


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Return a TestClient with local-only settings pointed at tmp_path."""

    reports_dir = tmp_path / "reports"
    extracted_dir = reports_dir / "extracted"
    ocr_dir = reports_dir / "ocr"
    findings_path = reports_dir / "reviews" / "findings.json"
    decisions_path = tmp_path / "reviewer" / "decisions.json"
    register_path = tmp_path / "requirements_register.csv"
    register_path.write_text(
        "requirement_id,section,requirement_text_ar,requirement_text_en,"
        "evaluation_method,poc_selected\n",
        encoding="utf-8",
    )

    def override_settings() -> Settings:
        return Settings(
            findings_path=findings_path,
            reviewer_decisions_path=decisions_path,
            reports_dir=reports_dir,
            extracted_dir=extracted_dir,
            ocr_dir=ocr_dir,
            requirements_register_path=register_path,
            max_upload_mb=1,
        )

    recorded: list[dict[str, object]] = []

    def fake_launch(**kwargs: object) -> None:
        recorded.append(kwargs)

    monkeypatch.setattr(uploads_module, "_launch_worker", fake_launch)
    get_job_store().clear()
    app = create_app()
    app.dependency_overrides[get_settings] = override_settings
    with TestClient(app) as test_client:
        test_client.recorded = recorded  # type: ignore[attr-defined]
        yield test_client
    get_job_store().clear()


def _pdf_bytes() -> bytes:
    """Return the smallest byte sequence that looks like a PDF to our checks."""

    return b"%PDF-1.4\n%minimal\n%EOF\n"


def test_upload_queues_a_job_and_persists_the_file(
    client: TestClient, tmp_path: Path
) -> None:
    """A valid PDF returns 202 with a job and writes the file to disk."""

    response = client.post(
        "/reviewer/documents",
        files={"file": ("report_a.pdf", _pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == "report_a"
    assert body["status"] == "queued"
    assert (tmp_path / "reports" / "report_a.pdf").exists()
    assert len(client.recorded) == 1  # type: ignore[attr-defined]


def test_upload_rejects_non_pdf_bytes(client: TestClient) -> None:
    """A file that does not start with ``%PDF-`` is rejected with 422."""

    response = client.post(
        "/reviewer/documents",
        files={"file": ("report_a.pdf", b"PK\x03\x04 not pdf", "application/pdf")},
    )
    assert response.status_code == 422
    assert "not a PDF" in response.json()["detail"]


def test_upload_rejects_duplicate_document_id(client: TestClient) -> None:
    """A second upload with the same stem returns 409."""

    first = client.post(
        "/reviewer/documents",
        files={"file": ("report_a.pdf", _pdf_bytes(), "application/pdf")},
    )
    assert first.status_code == 202

    second = client.post(
        "/reviewer/documents",
        files={"file": ("report_a.pdf", _pdf_bytes(), "application/pdf")},
    )
    assert second.status_code == 409
    assert "already exists" in second.json()["detail"]


def test_upload_rejects_empty_file(client: TestClient) -> None:
    """An empty upload returns 422."""

    response = client.post(
        "/reviewer/documents",
        files={"file": ("rep.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 422


def test_upload_rejects_file_larger_than_the_configured_limit(
    client: TestClient,
) -> None:
    """A file larger than ``max_upload_mb`` returns 413."""

    big = b"%PDF-1.4\n" + (b"x" * (1024 * 1024 + 10))
    response = client.post(
        "/reviewer/documents",
        files={"file": ("huge.pdf", big, "application/pdf")},
    )
    assert response.status_code == 413


def test_list_jobs_returns_every_known_job(client: TestClient) -> None:
    """The GET endpoint echoes everything the store has."""

    client.post(
        "/reviewer/documents",
        files={"file": ("one.pdf", _pdf_bytes(), "application/pdf")},
    )
    client.post(
        "/reviewer/documents",
        files={"file": ("two.pdf", _pdf_bytes(), "application/pdf")},
    )
    response = client.get("/reviewer/documents")
    assert response.status_code == 200
    jobs = response.json()
    document_ids = {job["document_id"] for job in jobs}
    assert document_ids == {"one", "two"}


def test_get_job_returns_404_for_unknown_id(client: TestClient) -> None:
    """An unknown job id surfaces as 404."""

    response = client.get("/reviewer/documents/missing")
    assert response.status_code == 404
