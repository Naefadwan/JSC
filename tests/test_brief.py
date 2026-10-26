"""Tests for the owner-brief refresh pipeline.

The brief refresher composes three concerns: it folds the reviewer's decision
into the system's recommendation to produce an effective status, it renders
each sentinel-marked block as markdown, and it splices those blocks into a
brief file while leaving all editorial prose alone. Each layer is tested in
isolation so a regression in one does not mask the others.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from jsc_review.core.models import Evidence, ReviewResult
from jsc_review.review.brief import (
    BLOCK_NAMES,
    compute_snapshot,
    parse_gate_measurements,
    render_sections,
    update_brief,
)
from jsc_review.reviewer.models import ReviewerDecision


def _evidence(*, document: str = "doc-a", page: int = 10, verified: bool = True) -> Evidence:
    return Evidence(
        text="Board composition statement.",
        source_document=document,
        page=page,
        section="Board composition",
        chunk_id=f"{document}:p{page}:c01",
        verified=verified,
    )


def _result(
    *,
    requirement: str,
    status: str,
    method: str = "llm",
    human_review: str = "Required",
    confidence: float = 0.6,
    page: int = 10,
    document: str = "doc-a",
) -> ReviewResult:
    return ReviewResult(
        requirement_id=requirement,
        status=status,  # type: ignore[arg-type]
        reason="Example reason.",
        evidence=_evidence(document=document, page=page, verified=True),
        confidence=confidence,
        human_review=human_review,  # type: ignore[arg-type]
        evaluation_method=method,  # type: ignore[arg-type]
    )


def _decision(
    *,
    document: str,
    requirement: str,
    action: str,
    final_status: str,
    notes: str = "",
) -> ReviewerDecision:
    return ReviewerDecision(
        document_id=document,
        requirement_id=requirement,
        action=action,  # type: ignore[arg-type]
        final_status=final_status,  # type: ignore[arg-type]
        reviewer_name="Nayef",
        reviewer_notes=notes or ("changed" if action == "changed" else ""),
    )


def test_snapshot_prefers_decision_status_over_system_status() -> None:
    """The reviewer's final status wins when a decision exists."""

    findings = [
        ("doc-a", _result(requirement="GOV-001", status="Needs Review")),
    ]
    decisions = [
        _decision(
            document="doc-a",
            requirement="GOV-001",
            action="changed",
            final_status="Met",
            notes="confirmed from heading",
        )
    ]

    snapshot = compute_snapshot(findings, decisions)

    assert snapshot.status_counts.get("Met") == 1
    assert snapshot.status_counts.get("Needs Review", 0) == 0
    assert snapshot.findings[0].source == "reviewer:changed"
    assert snapshot.findings[0].reviewer_name == "Nayef"


def test_snapshot_falls_back_to_system_status_when_no_decision() -> None:
    findings = [
        ("doc-a", _result(requirement="DIS-010", status="Needs Review")),
        ("doc-b", _result(requirement="DIS-010", status="Needs Review", document="doc-b")),
    ]

    snapshot = compute_snapshot(findings, [])

    assert snapshot.status_counts.get("Needs Review") == 2
    assert all(f.source == "system" for f in snapshot.findings)
    assert snapshot.total_decisions == 0


def test_snapshot_counts_awaiting_review_only_for_system_met_with_required_review() -> None:
    """A `Met` already accepted by the reviewer no longer waits for a decision."""

    findings = [
        (
            "doc-a",
            _result(
                requirement="DIS-062",
                status="Met",
                method="deterministic",
                human_review="Not Required",
                confidence=0.95,
            ),
        ),
        ("doc-a", _result(requirement="GOV-001", status="Met")),
        ("doc-b", _result(requirement="GOV-001", status="Met", document="doc-b")),
    ]
    decisions = [
        _decision(document="doc-a", requirement="GOV-001", action="accepted", final_status="Met")
    ]

    snapshot = compute_snapshot(findings, decisions)

    awaiting = snapshot.met_awaiting_review
    assert len(awaiting) == 1
    assert awaiting[0].document_id == "doc-b"
    assert awaiting[0].requirement_id == "GOV-001"


def test_snapshot_tracks_unique_documents_and_requirements_in_insertion_order() -> None:
    findings = [
        ("doc-b", _result(requirement="DIS-010", status="Needs Review", document="doc-b")),
        ("doc-a", _result(requirement="GOV-001", status="Needs Review")),
        ("doc-b", _result(requirement="GOV-001", status="Needs Review", document="doc-b")),
    ]

    snapshot = compute_snapshot(findings, [])

    assert snapshot.document_ids == ("doc-b", "doc-a")
    assert snapshot.requirement_ids == ("DIS-010", "GOV-001")
    assert snapshot.total_findings == 3


def test_render_sections_produces_every_block_name() -> None:
    findings = [
        ("doc-a", _result(requirement="GOV-001", status="Met", method="llm")),
    ]
    snapshot = compute_snapshot(findings, [])

    sections = render_sections(snapshot, now=datetime(2026, 10, 25, 12, 0, tzinfo=UTC))

    assert set(sections.keys()) == set(BLOCK_NAMES)
    assert "Reports in scope: 1" in sections["scope"]
    assert "`Met`: 1" in sections["counts"]
    assert "| `doc-a` | `GOV-001` | llm | 10 | system |" in sections["met_table"]
    assert "awaiting a reviewer decision: 1" in sections["owner_items"]
    assert "2026-10-25 12:00 UTC" in sections["footer"]


def test_render_met_table_is_empty_message_when_no_met() -> None:
    findings = [("doc-a", _result(requirement="GOV-001", status="Needs Review"))]
    snapshot = compute_snapshot(findings, [])

    sections = render_sections(snapshot)

    assert "No `Met` recommendations" in sections["met_table"]


def test_update_brief_replaces_only_sentinel_blocks_and_keeps_prose() -> None:
    original = (
        "# Owner brief\n\n"
        "Editorial prose that must survive.\n\n"
        "<!-- auto:begin:scope -->\n"
        "- OLD scope line.\n"
        "<!-- auto:end:scope -->\n\n"
        "More editorial prose about the system.\n"
    )
    sections = {"scope": "- NEW scope line."}

    updated, result = update_brief(original, sections)

    assert "Editorial prose that must survive." in updated
    assert "More editorial prose about the system." in updated
    assert "- OLD scope line." not in updated
    assert "- NEW scope line." in updated
    assert result.replaced == ("scope",)
    assert result.appended == ()


def test_update_brief_is_idempotent_when_sections_unchanged() -> None:
    original = (
        "# Owner brief\n\n<!-- auto:begin:scope -->\n- Same content.\n<!-- auto:end:scope -->\n"
    )
    sections = {"scope": "- Same content."}

    once, result_once = update_brief(original, sections)
    twice, result_twice = update_brief(once, sections)

    assert once == twice
    assert result_once.replaced == ()
    assert result_once.unchanged == ("scope",)
    assert result_twice.replaced == ()


def test_update_brief_appends_missing_blocks_at_end() -> None:
    original = "# Owner brief\n\nJust prose, no sentinels yet.\n"
    sections = {"scope": "- Fresh scope line."}

    updated, result = update_brief(original, sections)

    assert "Just prose, no sentinels yet." in updated
    assert "<!-- auto:begin:scope -->" in updated
    assert "- Fresh scope line." in updated
    assert "<!-- auto:end:scope -->" in updated
    assert result.appended == ("scope",)
    assert result.replaced == ()


def test_full_pipeline_round_trip_against_a_minimal_brief() -> None:
    """compute -> render -> update produces a brief that re-renders unchanged."""

    findings = [
        ("doc-a", _result(requirement="GOV-001", status="Met", method="llm")),
        (
            "doc-a",
            _result(
                requirement="DIS-062",
                status="Met",
                method="deterministic",
                human_review="Not Required",
                confidence=0.95,
            ),
        ),
    ]
    decisions = [
        _decision(document="doc-a", requirement="GOV-001", action="accepted", final_status="Met")
    ]

    snapshot = compute_snapshot(findings, decisions)
    sections = render_sections(snapshot, now=datetime(2026, 10, 25, 12, 0, tzinfo=UTC))

    brief = "# Owner brief\n\nProse stays.\n"
    once, first = update_brief(brief, sections)
    twice, second = update_brief(once, sections)

    assert once == twice
    assert set(first.appended) == set(BLOCK_NAMES)
    assert second.replaced == ()
    assert "Prose stays." in once


def _gate_payload(
    *, passed: bool = True, best_rate: float = 0.875, k: int = 5
) -> dict[str, object]:
    return {
        "gate": {"k": k, "threshold": 0.8, "passed": passed},
        "summaries": {
            "fused": {
                "total": 56,
                "hits": {"5": 46},
                "rate_at_5": 0.8214285714285714,
            },
            "fused+dedup+augment": {
                "total": 56,
                "hits": {"5": int(best_rate * 56)},
                "rate_at_5": best_rate,
            },
        },
    }


def test_parse_gate_measurements_returns_unmeasured_when_payload_is_none() -> None:
    status = parse_gate_measurements(None)

    assert status.measured is False
    assert status.k == 0


def test_parse_gate_measurements_picks_best_pipeline_by_rate_at_k() -> None:
    status = parse_gate_measurements(_gate_payload(best_rate=0.875))

    assert status.measured is True
    assert status.k == 5
    assert status.passed is True
    assert status.best_method == "fused+dedup+augment"
    assert status.best_rate == pytest.approx(0.875)
    assert status.best_total == 56
    assert set(status.method_rates.keys()) == {"fused", "fused+dedup+augment"}


def test_render_retrieval_gate_prompts_the_user_when_not_measured() -> None:
    findings = [("doc-a", _result(requirement="GOV-001", status="Needs Review"))]
    snapshot = compute_snapshot(findings, [])

    sections = render_sections(snapshot)

    assert "retrieval_gate" in sections
    assert "not measured yet" in sections["retrieval_gate"]
    assert "_measure_retrieval_gate.py" in sections["retrieval_gate"]


def test_render_retrieval_gate_shows_pass_block_when_gate_passed() -> None:
    findings = [("doc-a", _result(requirement="GOV-001", status="Needs Review"))]
    snapshot = compute_snapshot(findings, [], gate_measurements=_gate_payload(passed=True))

    sections = render_sections(snapshot)
    block = sections["retrieval_gate"]

    assert "PASS" in block
    assert "87.5%" in block
    assert "fused+dedup+augment" in block
    assert "Threshold is 80%" in block


def test_render_retrieval_gate_shows_fail_block_when_gate_failed() -> None:
    findings = [("doc-a", _result(requirement="GOV-001", status="Needs Review"))]
    payload = _gate_payload(passed=False, best_rate=0.70)
    snapshot = compute_snapshot(findings, [], gate_measurements=payload)

    sections = render_sections(snapshot)

    assert "FAIL" in sections["retrieval_gate"]
    assert "70.0%" in sections["retrieval_gate"]


def _rerank_payload(*, off_rate: float, on_rate: float) -> dict[str, object]:
    """Build a payload where both rerank-off and rerank-on branches are measured."""

    return {
        "gate": {"k": 5, "threshold": 0.8, "passed": max(off_rate, on_rate) >= 0.8},
        "summaries": {
            "fused+dedup+augment": {
                "total": 56,
                "hits": {"5": int(off_rate * 56)},
                "rate_at_5": off_rate,
            },
            "fused+dedup+rerank+augment": {
                "total": 56,
                "hits": {"5": int(on_rate * 56)},
                "rate_at_5": on_rate,
            },
        },
    }


def test_retrieval_gate_recommends_keeping_reranker_off_when_rerank_off_wins() -> None:
    """Rerank-off > rerank-on → brief recommends ``use_reranker=False``."""

    findings = [("doc-a", _result(requirement="GOV-001", status="Needs Review"))]
    payload = _rerank_payload(off_rate=0.875, on_rate=0.821)
    snapshot = compute_snapshot(findings, [], gate_measurements=payload)

    block = render_sections(snapshot)["retrieval_gate"]

    assert "keep `Settings.use_reranker=False`" in block
    assert "87.5%" in block
    assert "82.1%" in block


def test_retrieval_gate_recommends_turning_reranker_on_when_rerank_on_wins() -> None:
    """Rerank-on > rerank-off → brief surfaces the gain and the setting."""

    findings = [("doc-a", _result(requirement="GOV-001", status="Needs Review"))]
    payload = _rerank_payload(off_rate=0.75, on_rate=0.90)
    snapshot = compute_snapshot(findings, [], gate_measurements=payload)

    block = render_sections(snapshot)["retrieval_gate"]

    assert "Consider setting `Settings.use_reranker=True`" in block
    assert "90.0%" in block


def test_retrieval_gate_stays_quiet_when_only_rerank_off_branch_was_measured() -> None:
    """No rerank measurements → no recommendation, just the baseline block."""

    findings = [("doc-a", _result(requirement="GOV-001", status="Needs Review"))]
    snapshot = compute_snapshot(findings, [], gate_measurements=_gate_payload())

    block = render_sections(snapshot)["retrieval_gate"]

    assert "Settings.use_reranker" not in block


def test_block_names_include_retrieval_gate() -> None:
    assert "retrieval_gate" in BLOCK_NAMES


def test_deferred_decision_marks_source_as_reviewer_deferred_and_keeps_needs_review() -> None:
    findings = [("doc-a", _result(requirement="GOV-011", status="Needs Review"))]
    decisions = [
        _decision(
            document="doc-a",
            requirement="GOV-011",
            action="deferred",
            final_status="Needs Review",
            notes="need more context",
        )
    ]

    snapshot = compute_snapshot(findings, decisions)

    assert snapshot.findings[0].source == "reviewer:deferred"
    assert snapshot.findings[0].status == "Needs Review"
    assert snapshot.decision_counts.get("deferred") == 1
