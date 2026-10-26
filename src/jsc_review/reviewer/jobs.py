"""In-memory job registry for uploaded documents.

Each upload becomes one job. The job carries its current stage, message,
and per-stage counters so the reviewer screen can poll for progress and
compute an ETA without touching disk.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "done", "error", "canceled"]
JobStage = Literal[
    "queued",
    "extracting",
    "ocr",
    "indexing",
    "indexing_global",
    "judging",
    "done",
    "canceled",
    "error",
]

_TERMINAL_STATUSES: frozenset[JobStatus] = frozenset({"done", "error", "canceled"})


class UploadJob(BaseModel):
    """One upload-and-process job.

    ``stage_total`` and ``stage_done`` describe progress within the current
    stage so the UI can draw a progress bar. ``stage_started_at`` lets the
    client compute an ETA from the processing rate so far.
    """

    model_config = ConfigDict(frozen=False)

    job_id: str
    document_id: str
    original_filename: str
    status: JobStatus = "queued"
    stage: JobStage = "queued"
    message: str = ""
    finding_count: int = 0
    stage_total: int | None = None
    stage_done: int | None = None
    stage_started_at: datetime | None = None
    cancel_requested: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        """Return True when this job will never change state again."""

        return self.status in _TERMINAL_STATUSES


class JobStore:
    """Thread-safe in-memory registry. Jobs do not survive a server restart."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, UploadJob] = {}

    def create(self, *, document_id: str, original_filename: str) -> UploadJob:
        job = UploadJob(
            job_id=uuid.uuid4().hex,
            document_id=document_id,
            original_filename=original_filename,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> UploadJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy() if job else None

    def list(self) -> list[UploadJob]:
        with self._lock:
            return [job.model_copy() for job in self._jobs.values()]

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: JobStage | None = None,
        message: str | None = None,
        finding_count: int | None = None,
        stage_total: int | None = None,
        stage_done: int | None = None,
        stage_started_at: datetime | None = None,
        reset_stage_progress: bool = False,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> UploadJob | None:
        """Mutate one job. Only the fields passed in are changed.

        ``reset_stage_progress`` clears ``stage_total``, ``stage_done``, and
        ``stage_started_at`` back to None. Use it when a new stage begins so
        the client stops showing the old stage's bar.
        """

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if reset_stage_progress:
                job.stage_total = None
                job.stage_done = None
                job.stage_started_at = None
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if message is not None:
                job.message = message
            if finding_count is not None:
                job.finding_count = finding_count
            if stage_total is not None:
                job.stage_total = stage_total
            if stage_done is not None:
                job.stage_done = stage_done
            if stage_started_at is not None:
                job.stage_started_at = stage_started_at
            if started_at is not None:
                job.started_at = started_at
            if finished_at is not None:
                job.finished_at = finished_at
            return job.model_copy()

    def request_cancel(self, job_id: str) -> UploadJob | None:
        """Flag the job for cancellation. Returns the updated copy.

        The pipeline worker checks ``cancel_requested`` between stages and
        inside the OCR and judging loops. A terminal job returns the
        existing copy unchanged so callers can distinguish "unknown" (None)
        from "already finished" (``is_terminal`` is True).
        """

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status not in _TERMINAL_STATUSES:
                job.cancel_requested = True
            return job.model_copy()

    def remove(self, job_id: str) -> UploadJob | None:
        """Drop one job from the registry. Returns the removed copy or None.

        Removal is only valid when the job is terminal; the caller should
        enforce that policy at the route layer.
        """

        with self._lock:
            job = self._jobs.pop(job_id, None)
            return job.model_copy() if job else None

    def clear(self) -> None:
        """Remove every job. Intended for tests."""

        with self._lock:
            self._jobs.clear()


_default_store = JobStore()


def get_job_store() -> JobStore:
    """Return the process-wide job store."""

    return _default_store
