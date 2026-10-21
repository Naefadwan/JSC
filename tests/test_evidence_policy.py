"""Typed-model and evidence-policy tests."""

import pytest
from pydantic import ValidationError

from jsc_review.core.models import Evidence, ReviewResult
from jsc_review.review.evidence import (
    EvidencePolicyError,
    human_review_not_required_min_confidence,
    validate_evidence_policy,
)


def evidence(*, verified: bool) -> Evidence:
    return Evidence(
        text="The board has seven members.",
        source_document="report-001",
        page=12,
        section="Board composition",
        chunk_id="report-001:p12:c03",
        verified=verified,
    )


def test_model_accepts_met_with_verified_evidence() -> None:
    result = ReviewResult(
        requirement_id="GOV-014",
        status="Met",
        reason="The required board composition is disclosed.",
        evidence=evidence(verified=True),
        confidence=0.95,
        human_review="Required",
        evaluation_method="llm",
    )

    assert result.evidence is not None
    assert result.evidence.verified is True


def test_model_accepts_not_met_with_verified_evidence() -> None:
    result = ReviewResult(
        requirement_id="GOV-014",
        status="Not Met",
        reason="The cited page contradicts the requirement.",
        evidence=evidence(verified=True),
        confidence=0.95,
        human_review="Required",
        evaluation_method="llm",
    )

    assert result.status == "Not Met"


@pytest.mark.parametrize("status", ["Met", "Not Met"])
def test_model_rejects_conclusive_status_with_unverified_evidence(status: str) -> None:
    with pytest.raises(ValidationError, match=f"{status} requires verified evidence"):
        ReviewResult(
            requirement_id="GOV-014",
            status=status,  # type: ignore[arg-type]
            reason="The required board composition is disclosed.",
            evidence=evidence(verified=False),
            confidence=0.95,
            human_review="Required",
            evaluation_method="deterministic",
        )


@pytest.mark.parametrize("status", ["Met", "Not Met"])
def test_model_rejects_conclusive_status_without_evidence(status: str) -> None:
    with pytest.raises(ValidationError, match=f"{status} requires verified evidence"):
        ReviewResult(
            requirement_id="GOV-014",
            status=status,  # type: ignore[arg-type]
            reason="The required board composition is disclosed.",
            confidence=0.95,
            human_review="Required",
            evaluation_method="deterministic",
        )


def test_model_rejects_partially_met() -> None:
    with pytest.raises(ValidationError):
        ReviewResult(
            requirement_id="GOV-014",
            status="Partially Met",  # type: ignore[arg-type]
            reason="The old status is no longer used.",
            evidence=evidence(verified=True),
            confidence=0.95,
            human_review="Required",
            evaluation_method="llm",
        )


def test_needs_review_may_omit_evidence() -> None:
    result = ReviewResult(
        requirement_id="GOV-014",
        status="Needs Review",
        reason="No verifiable passage was found.",
        confidence=0.2,
        human_review="Required",
        evaluation_method="llm",
    )

    assert result.evidence is None


def test_model_accepts_not_required_for_a_qualifying_deterministic_result() -> None:
    result = ReviewResult(
        requirement_id="DIS-062",
        status="Met",
        reason="The statement heading is present.",
        evidence=evidence(verified=True),
        confidence=human_review_not_required_min_confidence(),
        human_review="Not Required",
        evaluation_method="deterministic",
    )

    assert result.human_review == "Not Required"


def test_model_rejects_not_required_for_a_local_model_result() -> None:
    with pytest.raises(ValidationError, match="Not Required is allowed only"):
        ReviewResult(
            requirement_id="GOV-014",
            status="Met",
            reason="The required board composition is disclosed.",
            evidence=evidence(verified=True),
            confidence=0.95,
            human_review="Not Required",
            evaluation_method="llm",
        )


def test_policy_rejects_constructed_unsupported_met() -> None:
    unsupported = ReviewResult.model_construct(
        requirement_id="GOV-014",
        status="Met",
        reason="Unsupported result.",
        evidence=evidence(verified=False),
        confidence=0.5,
        human_review="Required",
        evaluation_method="deterministic",
    )

    with pytest.raises(EvidencePolicyError, match="Met requires verified evidence"):
        validate_evidence_policy(unsupported)
