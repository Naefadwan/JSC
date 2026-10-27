"""Routes that back the Golden-labels drawer.

A reviewer (Role A or Role B, both from the Commission side) uses these
routes to:

* sample N new candidates from the current findings file,
* label each candidate as Met / Not Met / Needs Review with a short reason,
* and watch the agreement counters so Phase 4 can report inter-rater
  disagreement rather than hiding it.

All state lives in a local JSON file (``Settings.golden_labels_path``).
Nothing is sent to any remote service.
"""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from jsc_review.core.settings import get_settings
from jsc_review.golden.models import GoldenCandidate, ReviewerLabel, Status
from jsc_review.golden.sampler import pick_candidates_from_file
from jsc_review.golden.store import (
    add_candidates,
    delete_candidate,
    load_golden_set,
    pin_ground_truth,
    upsert_label,
)

router = APIRouter(prefix="/reviewer/golden", tags=["reviewer", "golden"])


class GoldenSummary(BaseModel):
    """Progress counters the UI renders at the top of the drawer."""

    model_config = ConfigDict(extra="forbid")

    total: int
    unlabeled: int
    single_reviewer: int
    agree: int
    disagree: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    reviewers: list[str] = Field(default_factory=list)
    agreement_rate: float | None = None


class GoldenListResponse(BaseModel):
    """Full response: progress summary plus every candidate."""

    model_config = ConfigDict(extra="forbid")

    summary: GoldenSummary
    candidates: list[GoldenCandidate]


class SampleRequest(BaseModel):
    """Payload for ``POST /sample``: how many candidates and from what seed."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=1, le=200, default=30)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class SampleResponse(BaseModel):
    """What the sampler added, and the updated summary."""

    model_config = ConfigDict(extra="forbid")

    added: int
    skipped_duplicates: int
    summary: GoldenSummary


class LabelRequest(BaseModel):
    """Payload for ``POST /label``: one reviewer labels one candidate."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    reviewer_name: str = Field(min_length=1, max_length=120)
    label: Status
    reason: str = Field(default="", max_length=2000)


def _summarize(candidates: list[GoldenCandidate]) -> GoldenSummary:
    total = len(candidates)
    status_counts: Counter[str] = Counter()
    reviewers: set[str] = set()
    unlabeled = single = agree = disagree = 0
    for cand in candidates:
        for label in cand.labels:
            reviewers.add(label.reviewer_name)
            status_counts[label.label] += 1
        agreement = cand.agreement_status()
        if agreement == "unlabeled":
            unlabeled += 1
        elif agreement == "single":
            single += 1
        elif agreement == "agree":
            agree += 1
        else:
            disagree += 1
    double_labeled = agree + disagree
    agreement_rate = (agree / double_labeled) if double_labeled else None
    return GoldenSummary(
        total=total,
        unlabeled=unlabeled,
        single_reviewer=single,
        agree=agree,
        disagree=disagree,
        status_counts=dict(status_counts),
        reviewers=sorted(reviewers),
        agreement_rate=agreement_rate,
    )


@router.get("", response_model=GoldenListResponse)
def list_golden() -> GoldenListResponse:
    """Return every candidate plus the running agreement counters."""

    settings = get_settings()
    golden_set = load_golden_set(settings.golden_labels_path)
    return GoldenListResponse(
        summary=_summarize(golden_set.candidates),
        candidates=golden_set.candidates,
    )


@router.post("/sample", response_model=SampleResponse, status_code=201)
def sample_candidates(request: SampleRequest) -> SampleResponse:
    """Draw stratified candidates from findings.json and append to the store."""

    settings = get_settings()
    existing = load_golden_set(settings.golden_labels_path)
    exclude_keys = {cand.key for cand in existing.candidates}
    picked = pick_candidates_from_file(
        settings.findings_path,
        count=request.count,
        exclude_keys=exclude_keys,
        seed=request.seed,
    )
    before = len(existing.candidates)
    updated = add_candidates(settings.golden_labels_path, picked)
    added = len(updated.candidates) - before
    skipped = len(picked) - added
    return SampleResponse(
        added=added,
        skipped_duplicates=skipped,
        summary=_summarize(updated.candidates),
    )


@router.post("/label", response_model=GoldenCandidate)
def label_candidate(request: LabelRequest) -> GoldenCandidate:
    """Record one reviewer's label on one candidate."""

    settings = get_settings()
    label = ReviewerLabel(
        reviewer_name=request.reviewer_name,
        label=request.label,
        reason=request.reason,
    )
    try:
        return upsert_label(
            settings.golden_labels_path,
            document_id=request.document_id,
            requirement_id=request.requirement_id,
            label=label,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class EvidenceRequest(BaseModel):
    """Payload for ``POST /evidence``: pin or clear the ground-truth evidence."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    text: str = Field(default="", max_length=5000)
    pinned_by: str = Field(default="", max_length=120)


@router.post("/evidence", response_model=GoldenCandidate)
def pin_evidence(request: EvidenceRequest) -> GoldenCandidate:
    """Pin the reviewer-chosen correct page + passage for one candidate."""

    settings = get_settings()
    try:
        return pin_ground_truth(
            settings.golden_labels_path,
            document_id=request.document_id,
            requirement_id=request.requirement_id,
            page=request.page,
            text=request.text,
            pinned_by=request.pinned_by,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{document_id}/{requirement_id}", status_code=204)
def remove_golden_candidate(document_id: str, requirement_id: str) -> None:
    """Remove one candidate from the golden set (both reviewer labels drop)."""

    settings = get_settings()
    removed = delete_candidate(
        settings.golden_labels_path,
        document_id=document_id,
        requirement_id=requirement_id,
    )
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"No golden candidate for {document_id} / {requirement_id}.",
        )
