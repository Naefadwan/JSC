"""Phase 4 metrics: score the pipeline's findings against the golden set.

Design points:

* The *agreed* label on a golden candidate is the ground truth for scoring.
  If the two reviewers disagree, that candidate is excluded from the status
  metrics (it is listed separately as a disagreement that still needs
  resolution).
* Hit@5 and evidence accuracy require ``ground_truth_page`` on the
  candidate. Candidates without a pinned page are counted as unscored.
* Targets live in :data:`PHASE4_TARGETS` so the UI drawer and this module
  share one source of truth.

The returned :class:`Scorecard` carries both the raw counts and a per-metric
summary with its target and above/below flag so the UI is a straight render.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jsc_review.core.models import ReviewResult
from jsc_review.golden.models import GoldenCandidate, GoldenSet, Status

MetricDirection = Literal["higher_is_better", "lower_is_better"]


class MetricTarget(BaseModel):
    """One Phase 4 target in the work plan."""

    model_config = ConfigDict(extra="forbid")

    name: str
    target_value: float
    direction: MetricDirection
    description: str = ""


# Confirmed in docs/work_plan.md under "Phase 4 — Evaluation and improvement".
PHASE4_TARGETS: dict[str, MetricTarget] = {
    "status_accuracy": MetricTarget(
        name="status_accuracy",
        target_value=0.80,
        direction="higher_is_better",
        description="Correct status for a labeled finding (Met / Not Met / Needs Review).",
    ),
    "recall_unmet": MetricTarget(
        name="recall_unmet",
        target_value=0.90,
        direction="higher_is_better",
        description=(
            "Of candidates reviewers agreed were Not Met, how many the system also called Not Met."
        ),
    ),
    "false_negative_rate_unmet": MetricTarget(
        name="false_negative_rate_unmet",
        target_value=0.10,
        direction="lower_is_better",
        description="Reviewer-agreed Not Met cases the system missed (not called Not Met).",
    ),
    "evidence_accuracy": MetricTarget(
        name="evidence_accuracy",
        target_value=0.95,
        direction="higher_is_better",
        description="System-cited evidence page matches the reviewer-pinned ground-truth page.",
    ),
    "hit_at_5": MetricTarget(
        name="hit_at_5",
        target_value=0.90,
        direction="higher_is_better",
        description="Ground-truth page appears within the top-5 hybrid retrieval results.",
    ),
    "referral_rate": MetricTarget(
        name="referral_rate",
        target_value=0.30,
        direction="lower_is_better",
        description="Findings sent to human review as a fraction of total findings.",
    ),
}


class MetricReport(BaseModel):
    """One metric row rendered by the scorecard drawer."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    target_value: float
    direction: MetricDirection
    value: float | None = None  # None => not enough labeled data to compute.
    sample_size: int = 0
    meets_target: bool | None = None
    notes: str = ""


class Scorecard(BaseModel):
    """Everything the UI needs to render the scorecard drawer."""

    model_config = ConfigDict(extra="forbid")

    total_golden: int = 0
    scored_golden: int = 0  # Candidates whose two reviewers agreed.
    disagreement_count: int = 0
    unlabeled_count: int = 0
    with_ground_truth_evidence: int = 0
    metrics: list[MetricReport] = Field(default_factory=list)


# ---------------------------------------------------------------- helpers


def _agreed_label(candidate: GoldenCandidate) -> Status | None:
    """Return the single agreed label, or None when reviewers disagree."""

    if candidate.agreement_status() == "agree":
        return candidate.labels[0].label
    return None


def _findings_by_pair(
    findings: Iterable[tuple[str, ReviewResult]],
) -> dict[tuple[str, str], ReviewResult]:
    """Build a (document_id, requirement_id) -> most recent result map."""

    out: dict[tuple[str, str], ReviewResult] = {}
    for document_id, result in findings:
        out[(document_id, result.requirement_id)] = result
    return out


def _metric(name: str, *, value: float | None, sample_size: int, notes: str = "") -> MetricReport:
    target = PHASE4_TARGETS[name]
    meets: bool | None
    if value is None or sample_size == 0:
        meets = None
    elif target.direction == "higher_is_better":
        meets = value >= target.target_value
    else:
        meets = value <= target.target_value
    return MetricReport(
        name=name,
        description=target.description,
        target_value=target.target_value,
        direction=target.direction,
        value=value,
        sample_size=sample_size,
        meets_target=meets,
        notes=notes,
    )


# ---------------------------------------------------------------- scorer


def score_findings_against_golden(
    findings: Sequence[tuple[str, ReviewResult]],
    golden_set: GoldenSet,
    *,
    hit_at_5_value: float | None = None,
    hit_at_5_sample: int = 0,
    hit_at_5_notes: str = "",
) -> Scorecard:
    """Compare the pipeline's findings against the golden set.

    ``hit_at_5_value`` is computed by a separate component (retrieval) and
    passed in so this scorer stays independent of the retrieval stack.
    """

    scorecard = Scorecard(total_golden=len(golden_set.candidates))
    index = _findings_by_pair(findings)

    # Counters for status metrics (operate only on agreed candidates).
    status_total = 0
    status_hits = 0
    unmet_total = 0
    unmet_hits = 0
    evidence_total = 0
    evidence_hits = 0

    for candidate in golden_set.candidates:
        agreement = candidate.agreement_status()
        if agreement == "unlabeled":
            scorecard.unlabeled_count += 1
            continue
        if agreement == "disagree":
            scorecard.disagreement_count += 1
            continue
        if agreement == "single":
            # Not yet usable for scoring; waits for a second reviewer.
            continue

        agreed = _agreed_label(candidate)
        if agreed is None:
            continue

        system = index.get((candidate.document_id, candidate.requirement_id))
        if system is None:
            # The golden set holds a candidate the pipeline hasn't judged.
            continue

        scorecard.scored_golden += 1
        status_total += 1
        if system.status == agreed:
            status_hits += 1

        if agreed == "Not Met":
            unmet_total += 1
            if system.status == "Not Met":
                unmet_hits += 1

        if candidate.ground_truth_page is not None:
            scorecard.with_ground_truth_evidence += 1
            evidence_total += 1
            system_page = (
                system.evidence.page if system.evidence is not None else None
            )
            if system_page is not None and system_page == candidate.ground_truth_page:
                evidence_hits += 1

    # Referral rate uses the full findings file, not the golden subset,
    # because it is a global KPI on how much the system defers to humans.
    referral_total = len(findings)
    referral_hits = sum(
        1 for _, result in findings if result.human_review == "Required"
    )

    scorecard.metrics = [
        _metric(
            "status_accuracy",
            value=(status_hits / status_total) if status_total else None,
            sample_size=status_total,
        ),
        _metric(
            "recall_unmet",
            value=(unmet_hits / unmet_total) if unmet_total else None,
            sample_size=unmet_total,
        ),
        _metric(
            "false_negative_rate_unmet",
            value=(
                (unmet_total - unmet_hits) / unmet_total if unmet_total else None
            ),
            sample_size=unmet_total,
        ),
        _metric(
            "evidence_accuracy",
            value=(
                (evidence_hits / evidence_total) if evidence_total else None
            ),
            sample_size=evidence_total,
            notes=(
                "Only counts candidates with a pinned ground_truth_page."
            ),
        ),
        _metric(
            "hit_at_5",
            value=hit_at_5_value,
            sample_size=hit_at_5_sample,
            notes=hit_at_5_notes or (
                "Run the scorecard with include_hit_at_5=true to compute."
            ),
        ),
        _metric(
            "referral_rate",
            value=(referral_hits / referral_total) if referral_total else None,
            sample_size=referral_total,
            notes="Fraction of all findings that still need human review.",
        ),
    ]
    return scorecard
