"""Upload a local PDF and process it in a background thread.

All processing happens on this machine. The route accepts a multipart PDF,
stores it beside the other local reports, creates an in-memory job, and
launches a background worker that extracts, OCRs, indexes, and judges.

Workers are serialized on a module-level lock so parallel uploads queue
up instead of competing for OCR, embedding, and local-model resources on
the same machine. The HTTP response always returns instantly; the worker
blocks on the lock between the "queued" and "running" phases so the UI
can show accurate queue state.

The reviewer screen polls the job endpoint for progress. It can also
cancel a queued or running job and dismiss a terminal one.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from jsc_review.core.settings import Settings, get_settings
from jsc_review.documents.pipeline import (
    AutolinkHook,
    UploadCanceled,
    UploadError,
    process_uploaded_pdf,
    sanitize_document_id,
    save_uploaded_pdf,
)
from jsc_review.documents.removal import DeletionSummary, remove_document
from jsc_review.retrieval.reranker import CrossEncoder
from jsc_review.review.brief_service import refresh_brief_safely
from jsc_review.review.registry import load_selected_requirements
from jsc_review.reviewer.jobs import JobStore, UploadJob, get_job_store

_LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/reviewer/documents", tags=["reviewer"])

# One worker runs at a time so OCR, embedding, and local-model inference do
# not compete for the same local resources.
_processing_lock = threading.Lock()


def _resolve_reranker(settings: Settings) -> CrossEncoder | None:
    """Return the shared cross-encoder when the setting is on, else ``None``.

    Loads the model lazily on the first opt-in request so the sentence
    transformers import stays off the hot path for the default (off) setup.
    A load failure is logged and the pipeline continues without reranking;
    that keeps the upload working on machines where the model is missing.
    """

    if not settings.use_reranker:
        return None
    try:
        from jsc_review.retrieval.reranker import shared_cross_encoder

        return shared_cross_encoder()
    except Exception:  # noqa: BLE001  (load must never kill an upload)
        _LOGGER.exception("Reranker is enabled but could not be loaded; continuing without it")
        return None


def _build_autolink_hook(settings: Settings) -> AutolinkHook | None:
    """Return a hook that auto-links the uploaded doc when confidence is high.

    The hook consults the local company registry (fast, in-memory) and
    writes the link via the shared mapping store when the policy accepts.
    Failures are logged and swallowed; the review always proceeds.
    """

    from jsc_review.companies.autolink import decide_autolink, rank_candidates
    from jsc_review.companies.mapping import get_link, set_link
    from jsc_review.companies.registry import load_registry

    registry = load_registry(settings.companies_registry_path)
    if not registry.companies:
        return None

    def hook(document_id: str, pages: tuple[tuple[int, str], ...]) -> None:
        # Respect any manual pre-existing link; never silently overwrite.
        existing = get_link(settings.document_companies_path, document_id)
        if existing is not None:
            return
        candidates = rank_candidates(
            pages, registry, scan_pages=settings.auto_link_scan_pages
        )
        decision = decide_autolink(
            candidates,
            threshold=settings.auto_link_confidence_threshold,
            min_lead_margin=settings.auto_link_min_lead_margin,
        )
        if decision.accepted is None:
            _LOGGER.info(
                "Autolink skipped %s: %s (top=%s)",
                document_id,
                decision.reason,
                decision.candidates[0].company.key if decision.candidates else "n/a",
            )
            return
        company = decision.accepted.company
        set_link(
            settings.document_companies_path,
            document_id=document_id,
            company_key=company.key,
            linked_by="auto",
        )
        _LOGGER.info(
            "Autolinked %s to %s (score=%.2f)",
            document_id,
            company.key,
            decision.accepted.score,
        )

    return hook


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


@router.post("/{job_id}/cancel", response_model=UploadJob)
def cancel_job(job_id: str) -> UploadJob:
    """Request cancellation of a queued or running job.

    The worker observes the flag between stages and inside the OCR and
    judging loops. Returns the updated job so the client can confirm the
    request landed. Terminal jobs return 409 so the UI can switch to the
    dismiss affordance instead.
    """

    store = get_job_store()
    existing = store.get(job_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id {job_id}")
    if existing.is_terminal:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is already {existing.status}",
        )
    updated = store.request_cancel(job_id)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id {job_id}")
    return updated


@router.delete("/by-id/{document_id}", response_model=DeletionSummary)
def delete_document_fully(
    document_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> DeletionSummary:
    """Delete every trace of a document from every local store.

    Removes the PDF, extracted JSON, OCR JSON, chunk mirror, findings
    entries, reviewer decisions, CER measurements, and global BM25 and
    dense index chunks. Any matching finished job in the registry is
    dismissed afterward so the UI's jobs panel does not surface a doc
    that no longer exists.

    Returns 409 when an active job is still processing the document; the
    caller must cancel the job first so no in-flight write races the
    delete.
    """

    if "/" in document_id or "\\" in document_id or ".." in document_id:
        raise HTTPException(status_code=422, detail="Invalid document id")

    store = get_job_store()
    active_for_doc = [
        job
        for job in store.list()
        if job.document_id == document_id and job.status in {"queued", "running"}
    ]
    if active_for_doc:
        raise HTTPException(
            status_code=409,
            detail=f"An upload for {document_id} is still active; cancel it first",
        )

    summary = remove_document(
        document_id,
        reports_dir=settings.reports_dir,
        extracted_dir=settings.extracted_dir,
        ocr_dir=settings.ocr_dir,
        chunks_with_ocr_dir=settings.chunks_with_ocr_dir,
        findings_path=settings.findings_path,
        decisions_path=settings.reviewer_decisions_path,
        cer_measurements_path=settings.cer_measurements_path,
        bm25_index_path=settings.bm25_index_path,
        dense_index_path=settings.dense_index_path,
        document_companies_path=settings.document_companies_path,
    )

    # Dismiss any finished jobs for this doc so the UI panel stops listing it.
    for job in store.list():
        if job.document_id == document_id and job.is_terminal:
            store.remove(job.job_id)

    if settings.auto_refresh_brief:
        refresh_brief_safely(
            findings_path=settings.findings_path,
            decisions_path=settings.reviewer_decisions_path,
            brief_path=settings.brief_path,
            gate_measurements_path=settings.retrieval_gate_measurements_path,
            context=f"delete {document_id}",
        )

    return summary


@router.delete("/{job_id}", response_model=UploadJob)
def dismiss_job(job_id: str) -> UploadJob:
    """Drop a terminal job from the registry.

    Only finished, errored, or canceled jobs can be dismissed. An active
    job returns 409 so the UI prompts the reviewer to cancel first.
    """

    store = get_job_store()
    existing = store.get(job_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id {job_id}")
    if not existing.is_terminal:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is still {existing.status}; cancel it first",
        )
    removed = store.remove(job_id)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id {job_id}")
    return removed


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

    # Block here until any earlier worker finishes. The job stays "queued"
    # so the UI can show the right place in line.
    with _processing_lock:
        current = store.get(job_id)
        if current is not None and current.cancel_requested:
            store.update(
                job_id,
                status="canceled",
                stage="canceled",
                message="تم إلغاء المعالجة قبل البدء",
                finished_at=datetime.now(UTC),
                reset_stage_progress=True,
            )
            return

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
                reset_stage_progress=True,
            )
            return

        counter = {"n": 0}

        def report(stage: str, message: str) -> None:
            updates: dict[str, object] = {"message": message}
            if stage in {
                "extracting",
                "ocr",
                "indexing",
                "indexing_global",
                "judging",
                "done",
                "error",
            }:
                updates["stage"] = stage
            # Only count judging messages that name a specific requirement, not
            # the initial "حكم N متطلب" opener.
            if stage == "judging" and "/" in message:
                counter["n"] += 1
                updates["finding_count"] = counter["n"]
            store.update(job_id, **updates)  # type: ignore[arg-type]

        def report_stage(stage: str, done: int | None, total: int | None) -> None:
            if done in (None, 0):
                # Start of a stage: reset counters and record the start time so
                # the client can compute an ETA from elapsed/done.
                store.update(
                    job_id,
                    reset_stage_progress=True,
                    stage_total=total,
                    stage_done=done if done is not None else 0,
                    stage_started_at=datetime.now(UTC),
                )
                return
            store.update(job_id, stage_done=done, stage_total=total)

        def check_cancel() -> bool:
            current = store.get(job_id)
            return bool(current and current.cancel_requested)

        reranker = _resolve_reranker(settings)
        autolink_hook = _build_autolink_hook(settings) if settings.auto_link_company else None
        try:
            summary = process_uploaded_pdf(
                pdf_path,
                requirements=requirements,
                extracted_dir=settings.extracted_dir,
                ocr_dir=settings.ocr_dir,
                findings_path=settings.findings_path,
                progress=report,
                stage_progress=report_stage,
                is_canceled=check_cancel,
                bm25_index_path=settings.bm25_index_path,
                dense_index_path=settings.dense_index_path,
                update_dense=settings.update_dense_on_upload,
                use_dense_at_review=settings.use_dense_at_review,
                reranker=reranker,
                cer_measurements_path=settings.cer_measurements_path,
                chunks_with_ocr_dir=settings.chunks_with_ocr_dir,
                autolink_hook=autolink_hook,
            )
        except UploadCanceled:
            store.update(
                job_id,
                status="canceled",
                stage="canceled",
                message="تم إلغاء المعالجة",
                finished_at=datetime.now(UTC),
                reset_stage_progress=True,
            )
            return
        except Exception as exc:  # noqa: BLE001  (log via message; thread cannot raise)
            store.update(
                job_id,
                status="error",
                stage="error",
                message=f"خطأ في المعالجة: {exc}",
                finished_at=datetime.now(UTC),
                reset_stage_progress=True,
            )
            return
        if settings.auto_refresh_brief:
            refresh_brief_safely(
                findings_path=settings.findings_path,
                decisions_path=settings.reviewer_decisions_path,
                brief_path=settings.brief_path,
                gate_measurements_path=settings.retrieval_gate_measurements_path,
                context=f"upload {job_id}",
            )
        store.update(
            job_id,
            status="done",
            stage="done",
            message=f"اكتملت معالجة {summary.finding_count} متطلب",
            finding_count=summary.finding_count,
            finished_at=datetime.now(UTC),
            reset_stage_progress=True,
        )
