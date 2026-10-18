"""Typed-model and evidence-policy tests."""

import pytest
from pydantic import ValidationError

from jsc_review.core.models import Evidence, ReviewResult
from jsc_review.review.evidence import EvidencePolicyError, validate_evidence_policy


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
        human_review="Not Required",
    )

    assert result.evidence is not None
    assert result.evidence.verified is True


def test_model_rejects_met_with_unverified_evidence() -> None:
    with pytest.raises(ValidationError, match="Met requires verified evidence"):
        ReviewResult(
            requirement_id="GOV-014",
            status="Met",
            reason="The required board composition is disclosed.",
            evidence=evidence(verified=False),
            confidence=0.95,
            human_review="Not Required",
        )


def test_model_rejects_met_without_evidence() -> None:
    with pytest.raises(ValidationError, match="Met requires verified evidence"):
        ReviewResult(
            requirement_id="GOV-014",
            status="Met",
            reason="The required board composition is disclosed.",
            confidence=0.95,
            human_review="Not Required",
        )


def test_needs_review_may_omit_evidence() -> None:
    result = ReviewResult(
        requirement_id="GOV-014",
        status="Needs Review",
        reason="No verifiable passage was found.",
        confidence=0.2,
        human_review="Required",
    )

    assert result.evidence is None


def test_policy_rejects_constructed_unsupported_met() -> None:
    unsupported = ReviewResult.model_construct(
        requirement_id="GOV-014",
        status="Met",
        reason="Unsupported result.",
        evidence=evidence(verified=False),
        confidence=0.5,
        human_review="Required",
    )

    with pytest.raises(EvidencePolicyError, match="Met requires verified evidence"):
        validate_evidence_policy(unsupported)
