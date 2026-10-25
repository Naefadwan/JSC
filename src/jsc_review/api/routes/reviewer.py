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

from jsc_review.core.settings import get_settings
from jsc_review.reviewer.findings import FindingBundle, build_bundle
from jsc_review.reviewer.models import DecisionSubmission, ReviewerDecision
from jsc_review.reviewer.store import load_decisions, save_decision

router = APIRouter(prefix="/reviewer", tags=["reviewer"])

_REGISTER_PATH = Path("Data/checklists/requirements_register.csv")


@router.get("/findings", response_model=FindingBundle)
def list_findings() -> FindingBundle:
    """Return every finding plus the latest reviewer decision on it."""

    settings = get_settings()
    if not settings.findings_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Findings file not found at {settings.findings_path}. "
                "Run the pipeline first."
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
    return decision


@router.get("/", response_class=HTMLResponse)
def reviewer_page() -> HTMLResponse:
    """Serve the single-page Arabic reviewer interface."""

    html = _PAGE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


_PAGE_PATH = Path(__file__).resolve().parent.parent / "static" / "reviewer.html"
