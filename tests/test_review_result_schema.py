"""Versioned JSON Schema tests."""

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import ValidationError, validate

SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "review_result.schema.json"


def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        return json.load(schema_file)  # type: ignore[no-any-return]


def result(*, status: str, verified: bool | None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "requirement_id": "GOV-014",
        "status": status,
        "reason": "Schema test result.",
        "confidence": 0.9,
        "human_review": "Not Required",
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


@pytest.mark.parametrize("verified", [False, None])
def test_schema_rejects_met_without_verified_evidence(verified: bool | None) -> None:
    with pytest.raises(ValidationError):
        validate(result(status="Met", verified=verified), load_schema())


def test_schema_accepts_needs_review_without_evidence() -> None:
    validate(result(status="Needs Review", verified=None), load_schema())
