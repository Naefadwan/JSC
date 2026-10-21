"""Versioned JSON Schema tests."""

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import ValidationError, validate

from jsc_review.review.evidence import human_review_not_required_min_confidence

SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "review_result.schema.json"


def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        return json.load(schema_file)  # type: ignore[no-any-return]


def result(
    *,
    status: str,
    verified: bool | None,
    human_review: str = "Required",
    evaluation_method: str = "llm",
    confidence: float = 0.9,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "requirement_id": "GOV-014",
        "status": status,
        "reason": "Schema test result.",
        "confidence": confidence,
        "human_review": human_review,
        "evaluation_method": evaluation_method,
    }
    if verified is not None:
        value["evidence"] = {
            "text": "The board has seven members.",
            "source_document": "report-001",
            "page": 12,
            "section": "Board composition",
            "chunk_id": "report-001:p12:c03",
            "verified": verified,
        }
    return value


def test_schema_accepts_verified_met() -> None:
    validate(result(status="Met", verified=True), load_schema())


@pytest.mark.parametrize("status", ["Met", "Not Met"])
@pytest.mark.parametrize("verified", [False, None])
def test_schema_rejects_conclusive_status_without_verified_evidence(
    status: str,
    verified: bool | None,
) -> None:
    with pytest.raises(ValidationError):
        validate(result(status=status, verified=verified), load_schema())


def test_schema_rejects_partially_met() -> None:
    with pytest.raises(ValidationError):
        validate(result(status="Partially Met", verified=True), load_schema())


def test_schema_accepts_needs_review_without_evidence() -> None:
    validate(result(status="Needs Review", verified=None), load_schema())


def test_schema_accepts_not_required_for_a_qualifying_deterministic_result() -> None:
    validate(
        result(
            status="Met",
            verified=True,
            human_review="Not Required",
            evaluation_method="deterministic",
            confidence=human_review_not_required_min_confidence(),
        ),
        load_schema(),
    )


def test_schema_threshold_matches_versioned_policy() -> None:
    schema = load_schema()
    not_required_rule = schema["allOf"][1]["then"]["properties"]["confidence"]["minimum"]

    assert not_required_rule == human_review_not_required_min_confidence()


@pytest.mark.parametrize(
    ("human_review", "evaluation_method", "confidence", "verified"),
    [
        ("Not Required", "llm", 0.95, True),
        ("Not Required", "deterministic", 0.84, True),
        ("Not Required", "deterministic", 0.95, False),
        ("Not Required", "deterministic", 0.95, None),
    ],
)
def test_schema_rejects_not_required_outside_the_confirmed_rule(
    human_review: str,
    evaluation_method: str,
    confidence: float,
    verified: bool | None,
) -> None:
    with pytest.raises(ValidationError):
        validate(
            result(
                status="Met",
                verified=verified,
                human_review=human_review,
                evaluation_method=evaluation_method,
                confidence=confidence,
            ),
            load_schema(),
        )
