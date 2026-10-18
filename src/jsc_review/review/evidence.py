"""Evidence-policy checks independent of API transport."""

from jsc_review.core.models import ReviewResult


class EvidencePolicyError(ValueError):
    """Raised when a result violates the verified-evidence policy."""


def validate_evidence_policy(result: ReviewResult) -> None:
    """Require complete, verified evidence before returning ``Met``."""

    if result.status != "Met":
        return
    if result.evidence is None or not result.evidence.verified:
        raise EvidencePolicyError("Met requires verified evidence from the cited report page")
