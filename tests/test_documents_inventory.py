"""Tests for the Documents drawer inventory assembler."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

from jsc_review.companies.mapping import set_link
from jsc_review.companies.registry import Company
from jsc_review.documents.inventory import build_inventory
from jsc_review.reviewer.models import ReviewerDecision


def _write_findings(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _write_decisions(path: Path, decisions: list[ReviewerDecision]) -> None:
    payload = [decision.model_dump(mode="json") for decision in decisions]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _finding(
    doc_id: str,
    requirement_id: str,
    status: str = "Needs Review",
) -> dict[str, object]:
    evidence: dict[str, object] | None = None
    if status in {"Met", "Not Met"}:
        evidence = {
            "text": "verified passage",
            "source_document": doc_id,
            "page": 1,
            "section": "fixture",
            "chunk_id": f"{doc_id}-{requirement_id}",
            "verified": True,
        }
    return {
        "document_id": doc_id,
        "result": {
            "requirement_id": requirement_id,
            "status": status,
            "reason": "fixture",
            "evidence": evidence,
            "confidence": 0.5,
            "human_review": "Required",
            "evaluation_method": "llm",
        },
    }


def test_build_inventory_unions_doc_ids_from_every_source(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    extracted = tmp_path / "extracted"
    reports.mkdir()
    extracted.mkdir()
    findings_path = tmp_path / "findings.json"
    decisions_path = tmp_path / "decisions.json"
    mapping_path = tmp_path / "doc_companies.json"

    # Doc A: in findings only
    _write_findings(findings_path, [_finding("doc_a_15-04-2026_09-00-00", "R1")])
    # Doc B: pdf + extracted, never judged
    (reports / "doc_b_01-02-2026_10-00-00.pdf").write_bytes(b"%PDF")
    (extracted / "doc_b_01-02-2026_10-00-00.json").write_text(
        json.dumps({"pages": []}), encoding="utf-8"
    )
    # Doc C: linked-only (manual link to a company we never uploaded)
    set_link(mapping_path, document_id="doc_c_31-03-2026_12-00-00", company_key="JOIB")

    entries = build_inventory(
        findings_path=findings_path,
        extracted_dir=extracted,
        reports_dir=reports,
        decisions_path=decisions_path,
        document_companies_path=mapping_path,
        companies=(
            Company(name_ar="Bank", ticker="JOIB", numeric_code="111001"),
        ),
    )
    ids = {entry.document_id for entry in entries}
    assert ids == {
        "doc_a_15-04-2026_09-00-00",
        "doc_b_01-02-2026_10-00-00",
        "doc_c_31-03-2026_12-00-00",
    }


def test_build_inventory_tally_counts_and_decisions(tmp_path: Path) -> None:
    findings_path = tmp_path / "findings.json"
    decisions_path = tmp_path / "decisions.json"
    _write_findings(
        findings_path,
        [
            _finding("doc_a_15-04-2026_09-00-00", "R1", "Met"),
            _finding("doc_a_15-04-2026_09-00-00", "R2", "Not Met"),
            _finding("doc_a_15-04-2026_09-00-00", "R3", "Needs Review"),
        ],
    )
    _write_decisions(
        decisions_path,
        [
            ReviewerDecision(
                document_id="doc_a_15-04-2026_09-00-00",
                requirement_id="R1",
                action="accepted",
                final_status="Met",
                reviewer_name="Nayef",
            )
        ],
    )

    entries = build_inventory(
        findings_path=findings_path,
        extracted_dir=None,
        reports_dir=None,
        decisions_path=decisions_path,
        document_companies_path=None,
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry.counts.total == 3
    assert entry.counts.met == 1
    assert entry.counts.not_met == 1
    assert entry.counts.needs_review == 1
    assert entry.counts.reviewed == 1
    assert entry.counts.unreviewed == 2
    assert entry.judged is True


def test_build_inventory_parses_submission_date_from_doc_id(tmp_path: Path) -> None:
    findings_path = tmp_path / "findings.json"
    _write_findings(
        findings_path,
        [_finding("doc_a_15-04-2026_09-30-45", "R1")],
    )
    entries = build_inventory(
        findings_path=findings_path,
        extracted_dir=None,
        reports_dir=None,
        decisions_path=None,
        document_companies_path=None,
    )
    stamp = entries[0].submitted_at
    assert stamp is not None
    assert stamp.year == 2026
    assert stamp.month == 4
    assert stamp.day == 15
    assert stamp.tzinfo is UTC


def test_build_inventory_resolves_company_from_link(tmp_path: Path) -> None:
    findings_path = tmp_path / "findings.json"
    mapping_path = tmp_path / "doc_companies.json"
    _write_findings(
        findings_path,
        [_finding("doc_a_15-04-2026_09-00-00", "R1")],
    )
    set_link(mapping_path, document_id="doc_a_15-04-2026_09-00-00", company_key="JOIB")
    entries = build_inventory(
        findings_path=findings_path,
        extracted_dir=None,
        reports_dir=None,
        decisions_path=None,
        document_companies_path=mapping_path,
        companies=(
            Company(name_ar="Jordan Islamic Bank", ticker="JOIB", numeric_code="111001"),
        ),
    )
    assert entries[0].link is not None
    assert entries[0].link.company_key == "JOIB"
    assert entries[0].company is not None
    assert entries[0].company.ticker == "JOIB"


def test_build_inventory_handles_all_none_paths() -> None:
    entries = build_inventory(
        findings_path=None,
        extracted_dir=None,
        reports_dir=None,
        decisions_path=None,
        document_companies_path=None,
    )
    assert entries == []


def test_build_inventory_sorts_newest_first_undated_last(tmp_path: Path) -> None:
    findings_path = tmp_path / "findings.json"
    _write_findings(
        findings_path,
        [
            _finding("old_15-01-2026_09-00-00", "R1"),
            _finding("new_15-05-2026_09-00-00", "R1"),
            _finding("undated_doc", "R1"),
            _finding("mid_15-03-2026_09-00-00", "R1"),
        ],
    )
    entries = build_inventory(
        findings_path=findings_path,
        extracted_dir=None,
        reports_dir=None,
        decisions_path=None,
        document_companies_path=None,
    )
    ordered = [entry.document_id for entry in entries]
    assert ordered[0] == "new_15-05-2026_09-00-00"
    assert ordered[1] == "mid_15-03-2026_09-00-00"
    assert ordered[2] == "old_15-01-2026_09-00-00"
    assert ordered[-1] == "undated_doc"


def test_build_inventory_rejects_unsafe_doc_ids(tmp_path: Path) -> None:
    """A traversal-style doc id in findings must not leak into the inventory."""

    findings_path = tmp_path / "findings.json"
    _write_findings(
        findings_path,
        [
            _finding("../etc/passwd", "R1"),
            _finding("legit_15-04-2026_09-00-00", "R1"),
        ],
    )
    entries = build_inventory(
        findings_path=findings_path,
        extracted_dir=None,
        reports_dir=None,
        decisions_path=None,
        document_companies_path=None,
    )
    assert [entry.document_id for entry in entries] == ["legit_15-04-2026_09-00-00"]
