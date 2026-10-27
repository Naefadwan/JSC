"""Unit tests for the Phase 4 scorecard assembler and the pin-evidence store."""

from __future__ import annotations

from pathlib import Path

from jsc_review.core.models import Evidence, ReviewResult
from jsc_review.golden.models import GoldenCandidate, GoldenSet, ReviewerLabel
from jsc_review.golden.store import load_golden_set, pin_ground_truth
from jsc_review.metrics.scorecard import PHASE4_TARGETS, score_findings_against_golden

# ------------------------------------------------- helpers / fixtures ---


def _result(req_id: str, status: str, page: int | None = 1) -> ReviewResult:
    evidence: Evidence | None = None
    if status in {"Met", "Not Met"} and page is not None:
        evidence = Evidence(
            text="verified passage",
            source_document="doc1",
            page=page,
            section="fixture",
            chunk_id=f"doc1-{req_id}",
            verified=True,
        )
    return ReviewResult(
        requirement_id=req_id,
        status=status,  # type: ignore[arg-type]
        reason="fixture",
        evidence=evidence,
        confidence=0.5,
        human_review="Required",
        evaluation_method="llm",
    )


def _agreed_candidate(
    document_id: str,
    requirement_id: str,
    agreed_label: str,
    *,
    ground_truth_page: int | None = None,
) -> GoldenCandidate:
    cand = GoldenCandidate(
        document_id=document_id,
        requirement_id=requirement_id,
        ground_truth_page=ground_truth_page,
    )
    cand.labels.append(ReviewerLabel(reviewer_name="A", label=agreed_label))  # type: ignore[arg-type]
    cand.labels.append(ReviewerLabel(reviewer_name="B", label=agreed_label))  # type: ignore[arg-type]
    return cand


# -------------------------------------------------------- scorecard ----


def test_scorecard_counts_agreement_states() -> None:
    unlabeled = GoldenCandidate(document_id="d", requirement_id="r1")
    single = GoldenCandidate(document_id="d", requirement_id="r2")
    single.labels.append(ReviewerLabel(reviewer_name="A", label="Met"))
    disagree = GoldenCandidate(document_id="d", requirement_id="r3")
    disagree.labels.append(ReviewerLabel(reviewer_name="A", label="Met"))
    disagree.labels.append(ReviewerLabel(reviewer_name="B", label="Not Met"))
    agree = _agreed_candidate("d", "r4", "Met")
    golden = GoldenSet(candidates=[unlabeled, single, disagree, agree])
    scorecard = score_findings_against_golden([], golden)
    assert scorecard.total_golden == 4
    assert scorecard.unlabeled_count == 1
    assert scorecard.disagreement_count == 1
    # single-reviewer candidates are neither disagreement nor scored
    assert scorecard.scored_golden == 0  # no finding for r4
    assert scorecard.with_ground_truth_evidence == 0


def test_scorecard_status_accuracy_and_recall_unmet() -> None:
    golden = GoldenSet(
        candidates=[
            _agreed_candidate("d1", "R1", "Met"),
            _agreed_candidate("d1", "R2", "Not Met"),
            _agreed_candidate("d1", "R3", "Not Met"),
            _agreed_candidate("d1", "R4", "Needs Review"),
        ]
    )
    findings = [
        ("d1", _result("R1", "Met")),
        ("d1", _result("R2", "Not Met")),
        ("d1", _result("R3", "Met")),  # miss
        ("d1", _result("R4", "Needs Review")),
    ]
    sc = score_findings_against_golden(findings, golden)
    status = next(m for m in sc.metrics if m.name == "status_accuracy")
    assert status.value is not None
    assert status.value == 0.75  # 3/4
    assert status.sample_size == 4
    recall = next(m for m in sc.metrics if m.name == "recall_unmet")
    assert recall.value == 0.5  # 1/2
    assert recall.sample_size == 2
    fn = next(m for m in sc.metrics if m.name == "false_negative_rate_unmet")
    assert fn.value == 0.5


def test_scorecard_evidence_accuracy_only_counts_pinned_pages() -> None:
    golden = GoldenSet(
        candidates=[
            _agreed_candidate("d1", "R1", "Met", ground_truth_page=5),
            _agreed_candidate("d1", "R2", "Met", ground_truth_page=10),
            _agreed_candidate("d1", "R3", "Met"),  # unpinned, excluded
        ]
    )
    findings = [
        ("d1", _result("R1", "Met", page=5)),  # hit
        ("d1", _result("R2", "Met", page=99)),  # miss
        ("d1", _result("R3", "Met", page=1)),
    ]
    sc = score_findings_against_golden(findings, golden)
    ev = next(m for m in sc.metrics if m.name == "evidence_accuracy")
    assert ev.sample_size == 2  # unpinned excluded
    assert ev.value == 0.5


def test_scorecard_targets_match_work_plan() -> None:
    assert PHASE4_TARGETS["status_accuracy"].target_value == 0.80
    assert PHASE4_TARGETS["recall_unmet"].target_value == 0.90
    assert PHASE4_TARGETS["false_negative_rate_unmet"].target_value == 0.10
    assert PHASE4_TARGETS["evidence_accuracy"].target_value == 0.95
    assert PHASE4_TARGETS["hit_at_5"].target_value == 0.90
    assert PHASE4_TARGETS["referral_rate"].target_value == 0.30


def test_scorecard_meets_target_flags() -> None:
    # 4 of 4 Met correctly, high accuracy -> should pass
    golden = GoldenSet(
        candidates=[_agreed_candidate("d1", f"R{i}", "Met") for i in range(1, 5)]
    )
    findings = [("d1", _result(f"R{i}", "Met")) for i in range(1, 5)]
    sc = score_findings_against_golden(findings, golden)
    status = next(m for m in sc.metrics if m.name == "status_accuracy")
    assert status.meets_target is True
    # FN rate 0% should pass (lower is better)
    fn = next(m for m in sc.metrics if m.name == "false_negative_rate_unmet")
    assert fn.meets_target is None  # no Not-Met samples


def test_scorecard_referral_rate_uses_full_findings() -> None:
    findings = [
        ("d1", ReviewResult(
            requirement_id="R1", status="Needs Review", reason="x",
            evidence=None, confidence=0.5, human_review="Required",
            evaluation_method="llm",
        )),
        ("d1", ReviewResult(
            requirement_id="R2", status="Met", reason="x",
            evidence=Evidence(
                text="v", source_document="d1", page=1, section="s",
                chunk_id="c", verified=True,
            ), confidence=0.9, human_review="Not Required",
            evaluation_method="deterministic",
        )),
    ]
    sc = score_findings_against_golden(findings, GoldenSet())
    rr = next(m for m in sc.metrics if m.name == "referral_rate")
    assert rr.value == 0.5
    assert rr.sample_size == 2
    assert rr.meets_target is False  # 0.5 > 0.30 target


def test_scorecard_hit_at_5_passthrough() -> None:
    sc = score_findings_against_golden(
        [], GoldenSet(), hit_at_5_value=0.92, hit_at_5_sample=25
    )
    hit = next(m for m in sc.metrics if m.name == "hit_at_5")
    assert hit.value == 0.92
    assert hit.sample_size == 25
    assert hit.meets_target is True


# -------------------------------------------------- pin_ground_truth ---


def test_pin_ground_truth_sets_and_clears(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    from jsc_review.golden.store import add_candidates

    add_candidates(path, [GoldenCandidate(document_id="d1", requirement_id="r1")])
    pin_ground_truth(
        path,
        document_id="d1",
        requirement_id="r1",
        page=42,
        text="quoted passage",
        pinned_by="Nayef",
    )
    loaded = load_golden_set(path)
    assert loaded.candidates[0].ground_truth_page == 42
    assert loaded.candidates[0].ground_truth_text == "quoted passage"
    assert loaded.candidates[0].ground_truth_pinned_by == "Nayef"
    assert loaded.candidates[0].ground_truth_pinned_at is not None
    # Clear path (page=None drops the pin)
    pin_ground_truth(
        path,
        document_id="d1",
        requirement_id="r1",
        page=None,
        text="",
        pinned_by="Nayef",
    )
    cleared = load_golden_set(path)
    assert cleared.candidates[0].ground_truth_page is None
    assert cleared.candidates[0].ground_truth_pinned_by == ""
    assert cleared.candidates[0].ground_truth_pinned_at is None
