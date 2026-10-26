"""Reviewer screen transport.

Serves the local reviewer HTML page, lists findings with the latest reviewer
decision, and records a new decision when the reviewer accepts or changes a
recommendation. All data stays on this machine.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from jsc_review.core.settings import get_settings
from jsc_review.review.brief_service import refresh_brief_safely
from jsc_review.review.register_edit import (
    RegisterRow,
    SelectionUpdate,
    load_register,
    update_register_selections,
)
from jsc_review.reviewer.findings import FindingBundle, build_bundle
from jsc_review.reviewer.models import DecisionSubmission, ReviewerDecision
from jsc_review.reviewer.store import load_decisions, save_decision

router = APIRouter(prefix="/reviewer", tags=["reviewer"])

_REGISTER_PATH = Path("Data/checklists/requirements_register.csv")


class RegisterResponse(BaseModel):
    """Everything the UI needs to render the requirement toggle panel."""

    model_config = ConfigDict(extra="forbid")

    rows: list[RegisterRow]
    selected_count: int
    total_count: int


class SelectionUpdateResponse(BaseModel):
    """The reply to a selection PATCH."""

    model_config = ConfigDict(extra="forbid")

    rows: list[RegisterRow]
    selected_count: int
    total_count: int
    unknown_requirement_ids: list[str] = []


@router.get("/findings", response_model=FindingBundle)
def list_findings() -> FindingBundle:
    """Return every finding plus the latest reviewer decision on it."""

    settings = get_settings()
    if not settings.findings_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Findings file not found at {settings.findings_path}. Run the pipeline first."
            ),
        )
    decisions = load_decisions(settings.reviewer_decisions_path)
    return build_bundle(
        settings.findings_path,
        decisions,
        requirements_register=_REGISTER_PATH,
    )


@router.post("/decisions", response_model=ReviewerDecision)
def submit_decision(submission: DecisionSubmission) -> ReviewerDecision:
    """Record one reviewer decision for one finding."""

    settings = get_settings()
    decision = ReviewerDecision(
        document_id=submission.document_id,
        requirement_id=submission.requirement_id,
        action=submission.action,
        final_status=submission.final_status,
        reviewer_name=submission.reviewer_name,
        reviewer_notes=submission.reviewer_notes,
        decided_at=datetime.now(UTC),
    )
    save_decision(settings.reviewer_decisions_path, decision)
    if settings.auto_refresh_brief:
        refresh_brief_safely(
            findings_path=settings.findings_path,
            decisions_path=settings.reviewer_decisions_path,
            brief_path=settings.brief_path,
            gate_measurements_path=settings.retrieval_gate_measurements_path,
            context=f"decision {decision.document_id}/{decision.requirement_id}",
        )
    return decision


@router.get("/requirements", response_model=RegisterResponse)
def list_requirements() -> RegisterResponse:
    """Return the full requirements register with the current selection flags."""

    settings = get_settings()
    rows = load_register(settings.requirements_register_path)
    selected = sum(1 for row in rows if row.poc_selected)
    return RegisterResponse(rows=rows, selected_count=selected, total_count=len(rows))


@router.patch("/requirements", response_model=SelectionUpdateResponse)
def update_requirement_selections(update: SelectionUpdate) -> SelectionUpdateResponse:
    """Apply reviewer toggle changes to ``poc_selected`` on the register CSV.

    Only the ``poc_selected`` flag is editable from the UI. The other
    columns describe the requirement itself and must stay aligned with the
    source checklist. Unknown ids are returned in
    ``unknown_requirement_ids`` so the UI can surface a warning rather than
    silently dropping stale toggles.

    The next upload picks up the new selection automatically because
    :func:`jsc_review.review.registry.load_selected_requirements` reads the
    register on each run.
    """

    settings = get_settings()
    rows, unknown = update_register_selections(
        settings.requirements_register_path, update.selections
    )
    selected = sum(1 for row in rows if row.poc_selected)
    return SelectionUpdateResponse(
        rows=rows,
        selected_count=selected,
        total_count=len(rows),
        unknown_requirement_ids=unknown,
    )


@router.get("/", response_class=HTMLResponse)
def reviewer_page() -> HTMLResponse:
    """Serve the single-page Arabic reviewer interface."""

    html = _PAGE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


_PAGE_PATH = Path(__file__).resolve().parent.parent / "static" / "reviewer.html"
