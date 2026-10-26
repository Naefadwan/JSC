"""Build the golden dataset from reviewer-confirmed decisions.

A golden record is one (document, requirement) pair where the reviewer has
spoken. The reviewer's ``final_status`` is the ground truth; the system's
``result.status`` is the prediction. The dataset supports two uses:

1. Measurement: report accuracy, per-status confusion, per-method breakdown
   so a code change can be judged against reviewer-confirmed truth.
2. Regression: when a change ships, re-score the system against the dataset
   and surface pairs whose prediction flipped.

Design choices
--------------
* ``deferred`` decisions are excluded. The reviewer explicitly said the
  evidence was inconclusive, so they do not label truth.
* Splitting is at the **document** level, not the pair level. The same
  document never appears in both dev and val, so evidence pages cannot leak
  between splits. New uploads land in whichever split their document hashes
  to, consistently across every run.
* The split is a deterministic hash of ``(salt, document_id)`` so the dev
  and val sets are reproducible across machines without storing split state.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from jsc_review.core.models import ReviewResult, ReviewStatus
from jsc_review.reviewer.models import ReviewerDecision

_LOGGER = logging.getLogger(__name__)

Split = Literal["dev", "val", "excluded"]

DEFAULT_SALT = "jsc-golden-v1"


@dataclass(frozen=True)
class GoldenRecord:
    """One reviewer-confirmed (document, requirement) pair for measurement."""

    document_id: str
    requirement_id: str
    system_status: ReviewStatus
    final_status: ReviewStatus
    evaluation_method: str
    reviewer_action: str
    reviewer_name: str
    confidence: float
    evidence_page: int | None
    evidence_verified: bool
    split: Split
    excluded_reason: str | None  # e.g. "deferred" or "no matching finding"

    @property
    def correct(self) -> bool:
        """Whether the system's recommendation matched the reviewer's truth."""

        return self.system_status == self.final_status


@dataclass(frozen=True)
class ConfusionCell:
    """One (system_status, final_status) count in the confusion matrix."""

    system_status: ReviewStatus
    final_status: ReviewStatus
    count: int


@dataclass(frozen=True)
class GoldenSummary:
    """Headline numbers for a golden dataset."""

    total_decisions: int
    labeled: int  # dev + val
    excluded: int
    dev_count: int
    val_count: int
    correct: int
    accuracy: float  # correct / labeled; 0.0 when labeled == 0
    accuracy_by_method: dict[str, float]
    labeled_by_method: dict[str, int]
    confusion: list[ConfusionCell]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _assign_split(document_id: str, *, salt: str, holdout_ratio: float) -> Literal["dev", "val"]:
    """Document-level deterministic split from a hash of ``(salt, doc_id)``.

    ``holdout_ratio`` is clamped to ``[0, 1]``. A document whose hashed value
    (mod 1000) falls below ``holdout_ratio * 1000`` lands in ``val``. Using
    1000 buckets gives sub-percent-level control over the split boundary.
    """

    clamped = max(0.0, min(1.0, holdout_ratio))
    digest = hashlib.sha256(f"{salt}|{document_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 1000
    threshold = int(round(clamped * 1000))
    return "val" if bucket < threshold else "dev"


def build_golden_dataset(
    findings: Iterable[tuple[str, ReviewResult]],
    decisions: Iterable[ReviewerDecision],
    *,
    salt: str = DEFAULT_SALT,
    holdout_ratio: float = 0.3,
) -> list[GoldenRecord]:
    """Fold decisions and findings into golden records.

    Each reviewer decision becomes one record. Deferred decisions are
    retained with ``split="excluded"`` so the summary can report how much
    the reviewer set aside as inconclusive.
    """

    findings_index: dict[tuple[str, str], ReviewResult] = {
        (document_id, result.requirement_id): result for document_id, result in findings
    }

    records: list[GoldenRecord] = []
    for decision in decisions:
        key = (decision.document_id, decision.requirement_id)
        finding = findings_index.get(key)
        if finding is None:
            _LOGGER.warning(
                "golden: decision has no matching finding; skipping %s/%s",
                decision.document_id,
                decision.requirement_id,
            )
            records.append(
                GoldenRecord(
                    document_id=decision.document_id,
                    requirement_id=decision.requirement_id,
                    system_status=decision.final_status,
                    final_status=decision.final_status,
                    evaluation_method="unknown",
                    reviewer_action=decision.action,
                    reviewer_name=decision.reviewer_name,
                    confidence=0.0,
                    evidence_page=None,
                    evidence_verified=False,
                    split="excluded",
                    excluded_reason="no matching finding",
                )
            )
            continue

        if decision.action == "deferred":
            split: Split = "excluded"
            excluded_reason: str | None = "deferred"
        else:
            split = _assign_split(decision.document_id, salt=salt, holdout_ratio=holdout_ratio)
            excluded_reason = None

        records.append(
            GoldenRecord(
                document_id=decision.document_id,
                requirement_id=decision.requirement_id,
                system_status=finding.status,
                final_status=decision.final_status,
                evaluation_method=finding.evaluation_method,
                reviewer_action=decision.action,
                reviewer_name=decision.reviewer_name,
                confidence=finding.confidence,
                evidence_page=finding.evidence.page if finding.evidence is not None else None,
                evidence_verified=(
                    finding.evidence.verified if finding.evidence is not None else False
                ),
                split=split,
                excluded_reason=excluded_reason,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


_STATUSES: tuple[ReviewStatus, ...] = ("Met", "Not Met", "Needs Review")


def summarize_golden(records: Iterable[GoldenRecord]) -> GoldenSummary:
    """Reduce records to the numbers that go in the measurement report."""

    records_list = list(records)
    total_decisions = len(records_list)

    labeled = [r for r in records_list if r.split != "excluded"]
    excluded = total_decisions - len(labeled)

    dev_count = sum(1 for r in labeled if r.split == "dev")
    val_count = sum(1 for r in labeled if r.split == "val")

    correct = sum(1 for r in labeled if r.correct)
    accuracy = correct / len(labeled) if labeled else 0.0

    labeled_by_method: Counter[str] = Counter(r.evaluation_method for r in labeled)
    correct_by_method: Counter[str] = Counter(r.evaluation_method for r in labeled if r.correct)
    accuracy_by_method = {
        method: (correct_by_method[method] / count) if count else 0.0
        for method, count in labeled_by_method.items()
    }

    cells = Counter((r.system_status, r.final_status) for r in labeled)
    confusion = [
        ConfusionCell(system_status=sys_s, final_status=fin_s, count=cells.get((sys_s, fin_s), 0))
        for sys_s in _STATUSES
        for fin_s in _STATUSES
    ]

    return GoldenSummary(
        total_decisions=total_decisions,
        labeled=len(labeled),
        excluded=excluded,
        dev_count=dev_count,
        val_count=val_count,
        correct=correct,
        accuracy=accuracy,
        accuracy_by_method=dict(accuracy_by_method),
        labeled_by_method=dict(labeled_by_method),
        confusion=confusion,
    )


# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------


def record_to_json(record: GoldenRecord) -> dict[str, object]:
    """Convert one record to a JSON-friendly dict."""

    return {
        "document_id": record.document_id,
        "requirement_id": record.requirement_id,
        "system_status": record.system_status,
        "final_status": record.final_status,
        "evaluation_method": record.evaluation_method,
        "reviewer_action": record.reviewer_action,
        "reviewer_name": record.reviewer_name,
        "confidence": record.confidence,
        "evidence_page": record.evidence_page,
        "evidence_verified": record.evidence_verified,
        "split": record.split,
        "excluded_reason": record.excluded_reason,
        "correct": record.correct,
    }


def render_report(summary: GoldenSummary, *, salt: str, holdout_ratio: float) -> str:
    """Produce the human-readable markdown report for the golden dataset."""

    header = "# Golden dataset report\n\n"
    overview = (
        f"- Reviewer decisions on file: {summary.total_decisions}.\n"
        f"- Labeled rows (dev + val): {summary.labeled}.\n"
        f"- Excluded: {summary.excluded} (deferred or orphaned decisions).\n"
        f"- Dev: {summary.dev_count}  Val: {summary.val_count}.\n"
        f"- Split: document-level, salt=`{salt}`, holdout={holdout_ratio:.2f}.\n"
        f"- Accuracy on labeled rows: {summary.accuracy:.1%}"
        f" ({summary.correct}/{summary.labeled}).\n"
    )

    if summary.labeled_by_method:
        method_lines = ["", "## Accuracy by evaluation method", ""]
        for method, count in sorted(summary.labeled_by_method.items()):
            acc = summary.accuracy_by_method.get(method, 0.0)
            method_lines.append(f"- `{method}`: {acc:.1%} on {count} rows.")
        method_block = "\n".join(method_lines) + "\n"
    else:
        method_block = ""

    confusion_lines = [
        "",
        "## Confusion (system vs reviewer final)",
        "",
        "| system \\ final | Met | Not Met | Needs Review |",
        "| --- | --- | --- | --- |",
    ]
    for sys_status in _STATUSES:
        row_cells = [f"**{sys_status}**"]
        for fin_status in _STATUSES:
            cell = next(
                (
                    c.count
                    for c in summary.confusion
                    if c.system_status == sys_status and c.final_status == fin_status
                ),
                0,
            )
            row_cells.append(str(cell))
        confusion_lines.append("| " + " | ".join(row_cells) + " |")
    confusion_block = "\n".join(confusion_lines) + "\n"

    return header + overview + method_block + confusion_block
