"""In-memory job registry for uploaded documents.

Each upload becomes one job. The job carries its current stage and message
so the reviewer screen can poll for progress without touching disk.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "done", "error"]
JobStage = Literal[
    "queued", "extracting", "ocr", "indexing", "judging", "done", "error"
]


class UploadJob(BaseModel):
    """One upload-and-process job."""

    model_config = ConfigDict(frozen=False)

    job_id: str
    document_id: str
    original_filename: str
    status: JobStatus = "queued"
    stage: JobStage = "queued"
    message: str = ""
    finding_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None


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
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> UploadJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if message is not None:
                job.message = message
            if finding_count is not None:
                job.finding_count = finding_count
            if started_at is not None:
                job.started_at = started_at
            if finished_at is not None:
                job.finished_at = finished_at
            return job.model_copy()

    def clear(self) -> None:
        """Remove every job. Intended for tests."""

        with self._lock:
            self._jobs.clear()


_default_store = JobStore()


def get_job_store() -> JobStore:
    """Return the process-wide job store."""

    return _default_store
