"""Typed models for the golden-label store.

A single golden entry pairs one finding (document_id + requirement_id) with
up to two independent reviewer labels. Phase 4 metrics compare the system
output against the agreed label.

The confirmed project statuses are ``Met``, ``Not Met``, and ``Needs Review``.
``Partially Met`` is rejected throughout the codebase and is rejected here
too.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["Met", "Not Met", "Needs Review"]


class ReviewerLabel(BaseModel):
    """One reviewer's label on one candidate."""

    model_config = ConfigDict(extra="forbid")

    reviewer_name: str = Field(min_length=1, max_length=120)
    label: Status
    reason: str = Field(default="", max_length=2000)
    labeled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("reviewer_name")
    @classmethod
    def _trim_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("reviewer_name may not be blank")
        return trimmed


class GoldenCandidate(BaseModel):
    """One (document, requirement) pair plus the proposed labels.

    ``ground_truth_page`` and ``ground_truth_text`` are the reviewer-pinned
    correct evidence for this candidate. They are what Hit@5 and evidence
    accuracy are scored against. ``system_*`` fields are the pipeline's
    current best guess, shown only to prime the reviewer; they never count
    as ground truth.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    system_status: Status | None = None
    system_reason: str = ""
    evidence_text: str = ""
    evidence_page: int | None = None
    ground_truth_page: int | None = Field(default=None, ge=1)
    ground_truth_text: str = Field(default="", max_length=5000)
    ground_truth_pinned_by: str = Field(default="", max_length=120)
    ground_truth_pinned_at: datetime | None = None
    added_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    labels: list[ReviewerLabel] = Field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.document_id, self.requirement_id)

    def label_for(self, reviewer_name: str) -> ReviewerLabel | None:
        """Return this reviewer's label or ``None`` when they have not labeled."""

        trimmed = reviewer_name.strip()
        return next(
            (label for label in self.labels if label.reviewer_name == trimmed),
            None,
        )

    def agreement_status(self) -> Literal["unlabeled", "single", "agree", "disagree"]:
        """Classify where this candidate sits in the double-review workflow."""

        if not self.labels:
            return "unlabeled"
        if len(self.labels) == 1:
            return "single"
        distinct = {label.label for label in self.labels}
        return "agree" if len(distinct) == 1 else "disagree"


class GoldenSet(BaseModel):
    """The on-disk envelope that holds every candidate plus simple metadata."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    candidates: list[GoldenCandidate] = Field(default_factory=list)

    def find(self, document_id: str, requirement_id: str) -> GoldenCandidate | None:
        key = (document_id, requirement_id)
        return next((cand for cand in self.candidates if cand.key == key), None)
