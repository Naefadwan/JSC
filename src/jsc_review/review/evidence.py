"""Evidence-policy checks independent of API transport."""

import json
from functools import lru_cache
from pathlib import Path

from jsc_review.core.models import ReviewResult

_CONCLUSIVE_STATUSES = frozenset({"Met", "Not Met"})


class EvidencePolicyError(ValueError):
    """Raised when a result violates the confirmed review policy."""


@lru_cache
def human_review_not_required_min_confidence() -> float:
    """Return the versioned confidence threshold for skipping human review."""

    policy_path = Path(__file__).with_name("policy.json")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    threshold = policy["human_review_not_required_min_confidence"]
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise EvidencePolicyError("human-review confidence threshold must be a number")
    return float(threshold)


def validate_evidence_policy(result: ReviewResult) -> None:
    """Apply the confirmed status, evidence, and human-review rules."""

    if result.status in _CONCLUSIVE_STATUSES and (
        result.evidence is None or not result.evidence.verified
    ):
        raise EvidencePolicyError(
            f"{result.status} requires verified evidence from the cited report page"
        )
    if result.human_review == "Not Required" and not _not_required_allowed(result):
        threshold = human_review_not_required_min_confidence()
        raise EvidencePolicyError(
            "Not Required is allowed only for a deterministic result with verified "
            f"evidence and confidence of at least {threshold}"
        )


def _not_required_allowed(result: ReviewResult) -> bool:
    return (
        result.status in _CONCLUSIVE_STATUSES
        and result.evaluation_method == "deterministic"
        and result.evidence is not None
        and result.evidence.verified
        and result.confidence >= human_review_not_required_min_confidence()
    )
