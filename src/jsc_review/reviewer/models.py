"""Pydantic models for reviewer decisions.

The reviewer's job is to confirm, change, or defer the system's recommendation
on one finding. The same three statuses apply here as everywhere else:
``Met``, ``Not Met``, ``Needs Review``. ``Partially Met`` is rejected by the
schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jsc_review.core.models import ReviewStatus

ReviewerAction = Literal["accepted", "changed", "deferred"]


class _DecisionCore(BaseModel):
    """Fields shared by the stored decision and the incoming submission."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    action: ReviewerAction
    final_status: ReviewStatus
    reviewer_name: str = Field(min_length=1, max_length=120)
    reviewer_notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def enforce_action_matches_status(self) -> Self:
        """Reviewer policy: a change needs notes; a defer stays Needs Review."""

        if self.action == "changed" and not self.reviewer_notes:
            raise ValueError(
                "A change requires reviewer notes that record why the status was changed."
            )
        if self.action == "deferred" and self.final_status != "Needs Review":
            raise ValueError("A deferred decision keeps the finding in Needs Review.")
        return self


class ReviewerDecision(_DecisionCore):
    """One reviewer decision tied to one system recommendation."""

    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DecisionSubmission(_DecisionCore):
    """Payload accepted by POST /reviewer/decisions.

    ``decided_at`` is set on the server to keep clock trust local to the API.
    """
