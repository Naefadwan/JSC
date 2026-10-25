"""Tests for the local reviewer decisions store."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jsc_review.reviewer.models import ReviewerDecision
from jsc_review.reviewer.store import load_decisions, save_decision


def _decision(
    document_id: str,
    requirement_id: str,
    status: str = "Met",
    notes: str = "",
) -> ReviewerDecision:
    return ReviewerDecision(
        document_id=document_id,
        requirement_id=requirement_id,
        action="accepted",
        final_status=status,  # type: ignore[arg-type]
        reviewer_name="Tester",
        reviewer_notes=notes,
        decided_at=datetime(2026, 10, 22, 10, tzinfo=UTC),
    )


def test_load_from_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_decisions(tmp_path / "missing.json") == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    save_decision(path, _decision("doc1", "GOV-001"))
    decisions = load_decisions(path)
    assert len(decisions) == 1
    assert decisions[0].requirement_id == "GOV-001"


def test_second_save_replaces_the_first_for_the_same_key(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    save_decision(path, _decision("doc1", "GOV-001", status="Met"))
    save_decision(
        path,
        ReviewerDecision(
            document_id="doc1",
            requirement_id="GOV-001",
            action="changed",
            final_status="Not Met",
            reviewer_name="Tester",
            reviewer_notes="updated after second look",
        ),
    )
    decisions = load_decisions(path)
    assert len(decisions) == 1
    assert decisions[0].action == "changed"
    assert decisions[0].final_status == "Not Met"


def test_saving_different_keys_keeps_both(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    save_decision(path, _decision("doc1", "GOV-001"))
    save_decision(path, _decision("doc1", "GOV-002"))
    save_decision(path, _decision("doc2", "GOV-001"))
    decisions = load_decisions(path)
    assert len(decisions) == 3


def test_empty_file_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    path.write_text("", encoding="utf-8")
    assert load_decisions(path) == []
