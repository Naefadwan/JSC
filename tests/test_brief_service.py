"""Tests for the brief-refresh service shim.

The service wraps the pure brief module with disk I/O and a never-raise
contract. Both guarantees are tested here: the shim must gracefully no-op
when the brief or findings file is missing, and a corrupt or unreadable
findings file must not raise through the ``_safely`` wrapper that the
upload worker and decision endpoint call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jsc_review.review.brief_service import (
    refresh_brief_from_disk,
    refresh_brief_safely,
)

_SENTINEL_BRIEF = (
    "# Owner brief\n\n"
    "Prose stays.\n\n"
    "<!-- auto:begin:counts -->\n"
    "- old counts\n"
    "<!-- auto:end:counts -->\n"
)


def _finding(document: str, requirement: str, status: str) -> dict[str, object]:
    return {
        "document_id": document,
        "result": {
            "requirement_id": requirement,
            "status": status,
            "reason": "A reason.",
            "evidence": {
                "text": "Quote.",
                "source_document": document,
                "page": 10,
                "section": "s",
                "chunk_id": f"{document}:p10:c01",
                "verified": True,
            },
            "confidence": 0.9,
            "human_review": "Required",
            "evaluation_method": "llm",
        },
    }


def test_refresh_returns_none_when_findings_missing(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.md"
    brief_path.write_text(_SENTINEL_BRIEF, encoding="utf-8")

    summary = refresh_brief_from_disk(
        findings_path=tmp_path / "missing.json",
        decisions_path=tmp_path / "missing-decisions.json",
        brief_path=brief_path,
    )

    assert summary is None
    # Brief untouched.
    assert brief_path.read_text(encoding="utf-8") == _SENTINEL_BRIEF


def test_refresh_returns_none_when_brief_missing(tmp_path: Path) -> None:
    findings_path = tmp_path / "findings.json"
    findings_path.write_text(json.dumps([_finding("doc-a", "GOV-001", "Met")]), encoding="utf-8")

    summary = refresh_brief_from_disk(
        findings_path=findings_path,
        decisions_path=tmp_path / "decisions.json",
        brief_path=tmp_path / "brief.md",
    )

    assert summary is None


def test_refresh_rewrites_brief_from_findings(tmp_path: Path) -> None:
    findings_path = tmp_path / "findings.json"
    findings_path.write_text(
        json.dumps(
            [
                _finding("doc-a", "GOV-001", "Met"),
                _finding("doc-a", "DIS-010", "Needs Review"),
            ]
        ),
        encoding="utf-8",
    )
    brief_path = tmp_path / "brief.md"
    brief_path.write_text(_SENTINEL_BRIEF, encoding="utf-8")

    summary = refresh_brief_from_disk(
        findings_path=findings_path,
        decisions_path=tmp_path / "decisions.json",
        brief_path=brief_path,
    )

    assert summary is not None
    assert summary.findings_count == 2
    assert summary.decisions_count == 0
    assert summary.wrote is True
    body = brief_path.read_text(encoding="utf-8")
    assert "Prose stays." in body
    assert "`Met`: 1" in body
    assert "`Needs Review`: 1" in body
    # Appended blocks that were missing from the sentinel-only template.
    assert "<!-- auto:begin:met_table -->" in body


def _sentinel_brief_with_gate_block() -> str:
    return _SENTINEL_BRIEF + (
        "\n<!-- auto:begin:retrieval_gate -->\n"
        "- Retrieval gate: not measured yet.\n"
        "<!-- auto:end:retrieval_gate -->\n"
    )


def _gate_payload() -> dict[str, object]:
    return {
        "gate": {"k": 5, "threshold": 0.8, "passed": True},
        "summaries": {
            "fused": {"total": 56, "rate_at_5": 0.82},
            "fused+dedup+augment": {"total": 56, "rate_at_5": 0.875},
        },
    }


def test_refresh_reads_gate_file_when_present(tmp_path: Path) -> None:
    findings_path = tmp_path / "findings.json"
    findings_path.write_text(json.dumps([_finding("doc-a", "GOV-001", "Met")]), encoding="utf-8")
    brief_path = tmp_path / "brief.md"
    brief_path.write_text(_sentinel_brief_with_gate_block(), encoding="utf-8")
    gate_path = tmp_path / "measurements.json"
    gate_path.write_text(json.dumps(_gate_payload()), encoding="utf-8")

    summary = refresh_brief_from_disk(
        findings_path=findings_path,
        decisions_path=tmp_path / "decisions.json",
        brief_path=brief_path,
        gate_measurements_path=gate_path,
    )

    assert summary is not None
    assert summary.wrote is True
    body = brief_path.read_text(encoding="utf-8")
    assert "PASS" in body
    assert "fused+dedup+augment" in body
    assert "not measured yet" not in body


def test_refresh_tolerates_missing_gate_file(tmp_path: Path) -> None:
    """A missing gate file falls back to the unmeasured block, not an error."""

    findings_path = tmp_path / "findings.json"
    findings_path.write_text(json.dumps([_finding("doc-a", "GOV-001", "Met")]), encoding="utf-8")
    brief_path = tmp_path / "brief.md"
    brief_path.write_text(_sentinel_brief_with_gate_block(), encoding="utf-8")

    summary = refresh_brief_from_disk(
        findings_path=findings_path,
        decisions_path=tmp_path / "decisions.json",
        brief_path=brief_path,
        gate_measurements_path=tmp_path / "missing.json",
    )

    assert summary is not None
    body = brief_path.read_text(encoding="utf-8")
    assert "not measured yet" in body


def test_refresh_safely_logs_and_continues_on_corrupt_gate_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt gate file must not fail the brief refresh."""

    findings_path = tmp_path / "findings.json"
    findings_path.write_text(json.dumps([_finding("doc-a", "GOV-001", "Met")]), encoding="utf-8")
    brief_path = tmp_path / "brief.md"
    brief_path.write_text(_sentinel_brief_with_gate_block(), encoding="utf-8")
    gate_path = tmp_path / "measurements.json"
    gate_path.write_text("{not valid json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        summary = refresh_brief_safely(
            findings_path=findings_path,
            decisions_path=tmp_path / "decisions.json",
            brief_path=brief_path,
            gate_measurements_path=gate_path,
            context="unit-test",
        )

    assert summary is not None
    assert summary.wrote is True
    body = brief_path.read_text(encoding="utf-8")
    assert "not measured yet" in body
    assert any("retrieval gate file unreadable" in r.message for r in caplog.records)


def test_refresh_safely_swallows_corrupt_findings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt findings file must not propagate through the safe wrapper."""

    findings_path = tmp_path / "findings.json"
    findings_path.write_text("{not valid json", encoding="utf-8")
    brief_path = tmp_path / "brief.md"
    brief_path.write_text(_SENTINEL_BRIEF, encoding="utf-8")

    with caplog.at_level("ERROR"):
        summary = refresh_brief_safely(
            findings_path=findings_path,
            decisions_path=tmp_path / "decisions.json",
            brief_path=brief_path,
            context="unit-test",
        )

    assert summary is None
    assert any("brief refresh failed" in record.message for record in caplog.records)
    # Brief must be untouched when the service bails out.
    assert brief_path.read_text(encoding="utf-8") == _SENTINEL_BRIEF
