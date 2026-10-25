"""Upload a local PDF and process it in a background thread.

All processing happens on this machine. The route accepts a multipart PDF,
stores it beside the other local reports, creates an in-memory job, and
launches a background worker that extracts, OCRs, indexes, and judges.
The reviewer screen polls the job endpoint for progress.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from jsc_review.core.settings import Settings, get_settings
from jsc_review.documents.pipeline import (
    UploadError,
    process_uploaded_pdf,
    sanitize_document_id,
    save_uploaded_pdf,
)
from jsc_review.review.registry import load_selected_requirements
from jsc_review.reviewer.jobs import JobStore, UploadJob, get_job_store

router = APIRouter(prefix="/reviewer/documents", tags=["reviewer"])


@router.post(
    "",
    response_model=UploadJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    file: Annotated[UploadFile, File(description="A local PDF report")],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadJob:
    """Accept one PDF upload and queue it for background processing."""

    store = get_job_store()
    original_filename = file.filename or ""
    try:
        document_id = sanitize_document_id(original_filename)
    except UploadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if not data:
        raise HTTPException(status_code=422, detail="The uploaded file is empty")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"The uploaded file is larger than {settings.max_upload_mb} MB",
        )
    try:
        pdf_path = save_uploaded_pdf(
            data=data,
            original_filename=original_filename,
            reports_dir=settings.reports_dir,
        )
    except UploadError as exc:
        status_code = 409 if "already exists" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    job = store.create(document_id=document_id, original_filename=original_filename)
    _launch_worker(job_id=job.job_id, pdf_path=pdf_path, store=store, settings=settings)
    return job


@router.get("", response_model=list[UploadJob])
def list_jobs() -> list[UploadJob]:
    """Return every known upload job in creation order."""

    jobs = get_job_store().list()
    return sorted(jobs, key=lambda job: job.created_at)


@router.get("/{job_id}", response_model=UploadJob)
def get_job(job_id: str) -> UploadJob:
    """Return the latest state of one upload job."""

    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id {job_id}")
    return job


def _launch_worker(
    *,
    job_id: str,
    pdf_path: Path,
    store: JobStore,
    settings: Settings,
) -> None:
    """Start a daemon thread so the HTTP response does not block on judging."""

    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "job_id": job_id,
            "pdf_path": pdf_path,
            "store": store,
            "settings": settings,
        },
        daemon=True,
        name=f"upload-{job_id[:8]}",
    )
    thread.start()


def _run_job(
    *,
    job_id: str,
    pdf_path: Path,
    store: JobStore,
    settings: Settings,
) -> None:
    """Execute the upload pipeline and record progress against the job."""

    store.update(
        job_id,
        status="running",
        stage="extracting",
        message="بدء المعالجة",
        started_at=datetime.now(UTC),
    )
    try:
        requirements = load_selected_requirements(settings.requirements_register_path)
    except FileNotFoundError as exc:
        store.update(
            job_id,
            status="error",
            stage="error",
            message=f"سجل المتطلبات غير موجود: {exc}",
            finished_at=datetime.now(UTC),
        )
        return

    counter = {"n": 0}

    def report(stage: str, message: str) -> None:
        updates: dict[str, object] = {"message": message}
        if stage in {"extracting", "ocr", "indexing", "judging", "done", "error"}:
            updates["stage"] = stage
        if stage == "judging":
            counter["n"] += 1
            updates["finding_count"] = counter["n"]
        store.update(job_id, **updates)  # type: ignore[arg-type]

    try:
        summary = process_uploaded_pdf(
            pdf_path,
            requirements=requirements,
            extracted_dir=settings.extracted_dir,
            ocr_dir=settings.ocr_dir,
            findings_path=settings.findings_path,
            progress=report,
        )
    except Exception as exc:  # noqa: BLE001  (log via message; thread cannot raise)
        store.update(
            job_id,
            status="error",
            stage="error",
            message=f"خطأ في المعالجة: {exc}",
            finished_at=datetime.now(UTC),
        )
        return
    store.update(
        job_id,
        status="done",
        stage="done",
        message=f"اكتملت معالجة {summary.finding_count} متطلب",
        finding_count=summary.finding_count,
        finished_at=datetime.now(UTC),
    )
