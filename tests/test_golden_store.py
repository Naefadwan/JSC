"""Tests for the golden-label store and sampler."""

from __future__ import annotations

from pathlib import Path

import pytest

from jsc_review.core.models import Evidence, ReviewResult
from jsc_review.golden.models import GoldenCandidate, ReviewerLabel
from jsc_review.golden.sampler import pick_candidates
from jsc_review.golden.store import (
    add_candidates,
    delete_candidate,
    load_golden_set,
    save_golden_set,
    upsert_label,
)

# -------------------------------------------------- models + agreement ---


def test_agreement_status_cycles_through_four_states() -> None:
    cand = GoldenCandidate(document_id="d", requirement_id="r")
    assert cand.agreement_status() == "unlabeled"
    cand.labels.append(ReviewerLabel(reviewer_name="A", label="Met"))
    assert cand.agreement_status() == "single"
    cand.labels.append(ReviewerLabel(reviewer_name="B", label="Met"))
    assert cand.agreement_status() == "agree"
    cand.labels[-1] = ReviewerLabel(reviewer_name="B", label="Not Met")
    assert cand.agreement_status() == "disagree"


# -------------------------------------------------- store atomic ops ----


def test_add_candidates_is_idempotent_on_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    first = add_candidates(
        path,
        [
            GoldenCandidate(document_id="d1", requirement_id="r1"),
            GoldenCandidate(document_id="d1", requirement_id="r2"),
        ],
    )
    assert len(first.candidates) == 2
    # Re-adding the same keys with different metadata must be a no-op.
    second = add_candidates(
        path,
        [GoldenCandidate(document_id="d1", requirement_id="r1", system_reason="x")],
    )
    assert len(second.candidates) == 2
    assert second.candidates[0].system_reason == ""


def test_upsert_label_replaces_prior_label_from_same_reviewer(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    add_candidates(path, [GoldenCandidate(document_id="d1", requirement_id="r1")])
    upsert_label(
        path,
        document_id="d1",
        requirement_id="r1",
        label=ReviewerLabel(reviewer_name="Nayef", label="Met", reason="first"),
    )
    upsert_label(
        path,
        document_id="d1",
        requirement_id="r1",
        label=ReviewerLabel(reviewer_name="Nayef", label="Not Met", reason="updated"),
    )
    loaded = load_golden_set(path)
    assert len(loaded.candidates[0].labels) == 1
    assert loaded.candidates[0].labels[0].label == "Not Met"
    assert loaded.candidates[0].labels[0].reason == "updated"


def test_upsert_label_keeps_other_reviewers_intact(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    add_candidates(path, [GoldenCandidate(document_id="d1", requirement_id="r1")])
    upsert_label(
        path,
        document_id="d1",
        requirement_id="r1",
        label=ReviewerLabel(reviewer_name="A", label="Met"),
    )
    upsert_label(
        path,
        document_id="d1",
        requirement_id="r1",
        label=ReviewerLabel(reviewer_name="B", label="Not Met"),
    )
    loaded = load_golden_set(path)
    names = {label.reviewer_name for label in loaded.candidates[0].labels}
    assert names == {"A", "B"}


def test_upsert_label_rejects_unknown_candidate(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    with pytest.raises(KeyError):
        upsert_label(
            path,
            document_id="d1",
            requirement_id="r1",
            label=ReviewerLabel(reviewer_name="A", label="Met"),
        )


def test_delete_candidate_removes_entry(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    add_candidates(
        path,
        [
            GoldenCandidate(document_id="d1", requirement_id="r1"),
            GoldenCandidate(document_id="d1", requirement_id="r2"),
        ],
    )
    assert delete_candidate(path, document_id="d1", requirement_id="r1") is True
    remaining = load_golden_set(path)
    keys = {cand.key for cand in remaining.candidates}
    assert keys == {("d1", "r2")}


def test_delete_candidate_misses_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    assert delete_candidate(path, document_id="x", requirement_id="y") is False


def test_save_and_load_roundtrip_preserves_everything(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    cand = GoldenCandidate(
        document_id="d1",
        requirement_id="r1",
        system_status="Met",
        system_reason="probe",
        evidence_text="verified quote",
        evidence_page=42,
        labels=[ReviewerLabel(reviewer_name="A", label="Met", reason="ok")],
    )
    add_candidates(path, [cand])
    loaded = load_golden_set(path)
    assert loaded.candidates[0].evidence_page == 42
    assert loaded.candidates[0].labels[0].reason == "ok"


# -------------------------------------------------- sampler ------------


def _finding(doc_id: str, req_id: str, status: str) -> tuple[str, ReviewResult]:
    evidence: Evidence | None = None
    if status in {"Met", "Not Met"}:
        evidence = Evidence(
            text="verified passage",
            source_document=doc_id,
            page=1,
            section="fixture",
            chunk_id=f"{doc_id}-{req_id}",
            verified=True,
        )
    return doc_id, ReviewResult(
        requirement_id=req_id,
        status=status,  # type: ignore[arg-type]
        reason="fixture",
        evidence=evidence,
        confidence=0.5,
        human_review="Required",
        evaluation_method="llm",
    )


def test_pick_candidates_stratifies_across_statuses() -> None:
    findings = [
        _finding("doc1", "R1", "Met"),
        _finding("doc1", "R2", "Met"),
        _finding("doc1", "R3", "Not Met"),
        _finding("doc1", "R4", "Not Met"),
        _finding("doc1", "R5", "Needs Review"),
        _finding("doc1", "R6", "Needs Review"),
    ]
    picked = pick_candidates(findings, count=3, seed=42)
    statuses = {cand.system_status for cand in picked}
    assert statuses == {"Met", "Not Met", "Needs Review"}


def test_pick_candidates_respects_exclusion() -> None:
    findings = [
        _finding("doc1", "R1", "Met"),
        _finding("doc1", "R2", "Not Met"),
    ]
    picked = pick_candidates(
        findings, count=5, exclude_keys={("doc1", "R1")}, seed=0
    )
    assert all(cand.requirement_id != "R1" for cand in picked)
    assert len(picked) == 1


def test_pick_candidates_is_deterministic_with_seed() -> None:
    findings = [
        _finding("doc1", "R1", "Met"),
        _finding("doc1", "R2", "Met"),
        _finding("doc1", "R3", "Needs Review"),
        _finding("doc1", "R4", "Needs Review"),
    ]
    first = pick_candidates(findings, count=4, seed=123)
    second = pick_candidates(findings, count=4, seed=123)
    assert [cand.key for cand in first] == [cand.key for cand in second]


def test_pick_candidates_handles_zero_count() -> None:
    assert pick_candidates([], count=0) == []
    assert pick_candidates(
        [_finding("doc1", "R1", "Met")], count=0, seed=0
    ) == []


def test_save_golden_set_atomic_tmp_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    cand = GoldenCandidate(document_id="d1", requirement_id="r1")
    from jsc_review.golden.models import GoldenSet

    save_golden_set(path, GoldenSet(candidates=[cand]))
    assert path.exists()
    # The .tmp sidecar must not linger after a successful write.
    assert not (tmp_path / "golden.json.tmp").exists()
