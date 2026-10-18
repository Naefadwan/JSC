"""Shared domain models for review results."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReviewStatus = Literal["Met", "Partially Met", "Not Met", "Needs Review"]
HumanReview = Literal["Required", "Not Required"]


class Evidence(BaseModel):
    """A report passage with provenance retained through the pipeline."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source_document: str = Field(min_length=1)
    page: int = Field(ge=1)
    section: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    verified: bool


class ReviewResult(BaseModel):
    """Schema-aligned recommendation for one audit requirement."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(min_length=1)
    status: ReviewStatus
    reason: str = Field(min_length=1)
    evidence: Evidence | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    human_review: HumanReview

    @model_validator(mode="after")
    def require_verified_evidence_for_met(self) -> Self:
        """Prevent an unsupported result from being represented as Met."""

        if self.status == "Met" and (self.evidence is None or not self.evidence.verified):
            raise ValueError("Met requires verified evidence from the cited report page")
        return self
