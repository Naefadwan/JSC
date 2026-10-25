"""Tests for reviewer decision models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jsc_review.reviewer.models import DecisionSubmission, ReviewerDecision


def test_accepted_decision_passes() -> None:
    decision = ReviewerDecision(
        document_id="doc1",
        requirement_id="GOV-001",
        action="accepted",
        final_status="Met",
        reviewer_name="A. Reviewer",
    )
    assert decision.action == "accepted"
    assert decision.final_status == "Met"


def test_change_requires_notes() -> None:
    with pytest.raises(ValidationError):
        ReviewerDecision(
            document_id="doc1",
            requirement_id="GOV-001",
            action="changed",
            final_status="Not Met",
            reviewer_name="A. Reviewer",
            reviewer_notes="",
        )


def test_deferred_must_be_needs_review() -> None:
    with pytest.raises(ValidationError):
        ReviewerDecision(
            document_id="doc1",
            requirement_id="GOV-001",
            action="deferred",
            final_status="Met",
            reviewer_name="A. Reviewer",
        )


def test_partially_met_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewerDecision(
            document_id="doc1",
            requirement_id="GOV-001",
            action="accepted",
            final_status="Partially Met",  # type: ignore[arg-type]
            reviewer_name="A. Reviewer",
        )


def test_decision_submission_rejects_extras() -> None:
    with pytest.raises(ValidationError):
        DecisionSubmission(
            document_id="doc1",
            requirement_id="GOV-001",
            action="accepted",
            final_status="Met",
            reviewer_name="A. Reviewer",
            unknown_field="x",  # type: ignore[call-arg]
        )
