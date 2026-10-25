"""Read findings and merge the latest reviewer decision for the UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from jsc_review.core.models import ReviewResult
from jsc_review.reviewer.models import ReviewerDecision


class FindingView(BaseModel):
    """One finding plus the latest reviewer decision, ready for display."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    result: ReviewResult
    decision: ReviewerDecision | None


class FindingBundle(BaseModel):
    """Everything the reviewer screen needs on first load."""

    model_config = ConfigDict(extra="forbid")

    findings: list[FindingView]
    requirement_text_ar: dict[str, str]
    requirement_text_en: dict[str, str]


def load_findings(path: Path) -> list[tuple[str, ReviewResult]]:
    """Read findings.json into typed (document_id, result) pairs."""

    payload: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    findings: list[tuple[str, ReviewResult]] = []
    for row in payload:
        document_id = str(row["document_id"])
        result = ReviewResult.model_validate(row["result"])
        findings.append((document_id, result))
    return findings


def build_bundle(
    findings_path: Path,
    decisions: list[ReviewerDecision],
    *,
    requirements_register: Path | None = None,
) -> FindingBundle:
    """Combine findings, reviewer decisions, and requirement labels."""

    by_key: dict[tuple[str, str], ReviewerDecision] = {
        (d.document_id, d.requirement_id): d for d in decisions
    }
    findings = load_findings(findings_path)
    views = [
        FindingView(
            document_id=document_id,
            result=result,
            decision=by_key.get((document_id, result.requirement_id)),
        )
        for document_id, result in findings
    ]
    text_ar, text_en = _load_requirement_text(requirements_register)
    return FindingBundle(
        findings=views,
        requirement_text_ar=text_ar,
        requirement_text_en=text_en,
    )


def _load_requirement_text(path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}, {}
    import csv

    text_ar: dict[str, str] = {}
    text_en: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            requirement_id = row.get("requirement_id")
            if not requirement_id:
                continue
            text_ar[requirement_id] = row.get("requirement_text_ar", "")
            text_en[requirement_id] = row.get("requirement_text_en", "")
    return text_ar, text_en
