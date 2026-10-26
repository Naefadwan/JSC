"""Tests for the golden dataset builder.

The builder has three responsibilities that must each hold independently:
turn each reviewer decision into one record with the right truth label,
split documents deterministically between dev and val with no leakage, and
summarize accuracy correctly, including a per-method breakdown and a
confusion matrix.
"""

from __future__ import annotations

import pytest

from jsc_review.core.models import Evidence, ReviewResult
from jsc_review.review.golden import (
    DEFAULT_SALT,
    build_golden_dataset,
    render_report,
    summarize_golden,
)
from jsc_review.reviewer.models import ReviewerDecision


def _evidence(document: str, page: int = 10) -> Evidence:
    return Evidence(
        text="Verified passage.",
        source_document=document,
        page=page,
        section="s",
        chunk_id=f"{document}:p{page}:c01",
        verified=True,
    )


def _finding(
    *,
    document: str,
    requirement: str,
    status: str,
    method: str = "llm",
    confidence: float = 0.6,
    human_review: str = "Required",
    with_evidence: bool = True,
) -> tuple[str, ReviewResult]:
    result = ReviewResult(
        requirement_id=requirement,
        status=status,  # type: ignore[arg-type]
        reason="A reason.",
        evidence=_evidence(document) if with_evidence else None,
        confidence=confidence,
        human_review=human_review,  # type: ignore[arg-type]
        evaluation_method=method,  # type: ignore[arg-type]
    )
    return (document, result)


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
        reviewer_name="A. Reviewer",
        reviewer_notes=notes or ("changed note" if action == "changed" else ""),
    )


def test_accepted_decision_records_system_prediction_and_ground_truth() -> None:
    findings = [_finding(document="doc-a", requirement="GOV-001", status="Met")]
    decisions = [
        _decision(document="doc-a", requirement="GOV-001", action="accepted", final_status="Met")
    ]

    records = build_golden_dataset(findings, decisions)

    assert len(records) == 1
    record = records[0]
    assert record.system_status == "Met"
    assert record.final_status == "Met"
    assert record.correct is True
    assert record.split in {"dev", "val"}


def test_changed_decision_captures_system_wrong_truth_right() -> None:
    findings = [_finding(document="doc-a", requirement="GOV-001", status="Needs Review")]
    decisions = [
        _decision(
            document="doc-a",
            requirement="GOV-001",
            action="changed",
            final_status="Met",
            notes="confirmed from heading",
        )
    ]

    records = build_golden_dataset(findings, decisions)

    assert records[0].system_status == "Needs Review"
    assert records[0].final_status == "Met"
    assert records[0].correct is False


def test_deferred_decision_is_excluded_with_reason() -> None:
    findings = [_finding(document="doc-a", requirement="GOV-011", status="Needs Review")]
    decisions = [
        _decision(
            document="doc-a",
            requirement="GOV-011",
            action="deferred",
            final_status="Needs Review",
            notes="need more context",
        )
    ]

    records = build_golden_dataset(findings, decisions)

    assert records[0].split == "excluded"
    assert records[0].excluded_reason == "deferred"


def test_decision_without_matching_finding_is_excluded_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    decisions = [
        _decision(
            document="doc-missing",
            requirement="GOV-001",
            action="accepted",
            final_status="Met",
        )
    ]

    with caplog.at_level("WARNING"):
        records = build_golden_dataset([], decisions)

    assert records[0].split == "excluded"
    assert records[0].excluded_reason == "no matching finding"
    assert any("no matching finding" in r.message for r in caplog.records)


def test_split_is_deterministic_across_runs() -> None:
    findings = [
        _finding(document="doc-a", requirement="GOV-001", status="Met"),
        _finding(document="doc-b", requirement="GOV-001", status="Met"),
    ]
    decisions = [
        _decision(document="doc-a", requirement="GOV-001", action="accepted", final_status="Met"),
        _decision(document="doc-b", requirement="GOV-001", action="accepted", final_status="Met"),
    ]

    once = build_golden_dataset(findings, decisions)
    twice = build_golden_dataset(findings, decisions)

    assert [r.split for r in once] == [r.split for r in twice]


def test_salt_changes_the_split_assignment() -> None:
    findings = [
        _finding(document=f"doc-{i}", requirement="GOV-001", status="Met") for i in range(50)
    ]
    decisions = [
        _decision(document=f"doc-{i}", requirement="GOV-001", action="accepted", final_status="Met")
        for i in range(50)
    ]

    with_default = build_golden_dataset(findings, decisions, salt=DEFAULT_SALT)
    with_alt = build_golden_dataset(findings, decisions, salt="alt-salt")

    splits_default = [r.split for r in with_default]
    splits_alt = [r.split for r in with_alt]

    assert splits_default != splits_alt


def test_holdout_ratio_zero_puts_everything_in_dev() -> None:
    findings = [
        _finding(document=f"doc-{i}", requirement="GOV-001", status="Met") for i in range(30)
    ]
    decisions = [
        _decision(document=f"doc-{i}", requirement="GOV-001", action="accepted", final_status="Met")
        for i in range(30)
    ]

    records = build_golden_dataset(findings, decisions, holdout_ratio=0.0)

    assert all(r.split == "dev" for r in records)


def test_holdout_ratio_one_puts_everything_in_val() -> None:
    findings = [
        _finding(document=f"doc-{i}", requirement="GOV-001", status="Met") for i in range(30)
    ]
    decisions = [
        _decision(document=f"doc-{i}", requirement="GOV-001", action="accepted", final_status="Met")
        for i in range(30)
    ]

    records = build_golden_dataset(findings, decisions, holdout_ratio=1.0)

    assert all(r.split == "val" for r in records)


def test_document_never_leaks_between_dev_and_val() -> None:
    """A document that lands in dev must never also appear in val."""

    findings = [
        _finding(document="doc-a", requirement="GOV-001", status="Met"),
        _finding(document="doc-a", requirement="DIS-010", status="Met"),
        _finding(document="doc-a", requirement="DIS-062", status="Met"),
    ]
    decisions = [
        _decision(document="doc-a", requirement="GOV-001", action="accepted", final_status="Met"),
        _decision(document="doc-a", requirement="DIS-010", action="accepted", final_status="Met"),
        _decision(document="doc-a", requirement="DIS-062", action="accepted", final_status="Met"),
    ]

    records = build_golden_dataset(findings, decisions)

    splits = {r.split for r in records}
    assert splits.issubset({"dev"}) or splits.issubset({"val"})


def test_summary_computes_accuracy_and_per_method_breakdown() -> None:
    findings = [
        _finding(document="doc-a", requirement="GOV-001", status="Met", method="llm"),
        _finding(document="doc-a", requirement="DIS-010", status="Met", method="deterministic"),
        _finding(document="doc-b", requirement="GOV-001", status="Needs Review", method="llm"),
    ]
    decisions = [
        _decision(document="doc-a", requirement="GOV-001", action="accepted", final_status="Met"),
        _decision(document="doc-a", requirement="DIS-010", action="accepted", final_status="Met"),
        _decision(
            document="doc-b",
            requirement="GOV-001",
            action="changed",
            final_status="Met",
            notes="confirmed",
        ),
    ]

    records = build_golden_dataset(findings, decisions)
    summary = summarize_golden(records)

    assert summary.labeled == 3
    assert summary.correct == 2
    assert summary.accuracy == pytest.approx(2 / 3)
    # Deterministic: 1/1 correct. LLM: 1/2 correct.
    assert summary.accuracy_by_method["deterministic"] == pytest.approx(1.0)
    assert summary.accuracy_by_method["llm"] == pytest.approx(0.5)
    assert summary.labeled_by_method == {"llm": 2, "deterministic": 1}


def test_summary_confusion_matrix_has_all_nine_cells() -> None:
    findings = [_finding(document="doc-a", requirement="GOV-001", status="Met")]
    decisions = [
        _decision(document="doc-a", requirement="GOV-001", action="accepted", final_status="Met")
    ]

    records = build_golden_dataset(findings, decisions)
    summary = summarize_golden(records)

    assert len(summary.confusion) == 9
    met_met = next(
        c for c in summary.confusion if c.system_status == "Met" and c.final_status == "Met"
    )
    assert met_met.count == 1


def test_render_report_mentions_core_numbers() -> None:
    findings = [_finding(document="doc-a", requirement="GOV-001", status="Met")]
    decisions = [
        _decision(document="doc-a", requirement="GOV-001", action="accepted", final_status="Met")
    ]

    summary = summarize_golden(build_golden_dataset(findings, decisions))
    report = render_report(summary, salt=DEFAULT_SALT, holdout_ratio=0.3)

    assert "Golden dataset report" in report
    assert "salt=`jsc-golden-v1`" in report
    assert "Accuracy on labeled rows: 100.0%" in report
    assert "Confusion (system vs reviewer final)" in report
