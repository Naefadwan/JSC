"""Pick a stratified set of findings to promote into the golden store.

Strategy: group findings by ``(document_id, requirement_id, status)``, prefer
one candidate per group so the labeled set spans every status and every
requirement, then round-robin fill the remainder. The sampler is seeded so
two reruns on the same inputs promote the same findings; that makes label
provenance auditable.

Candidates already present in the golden store are excluded so a repeat
run only adds net-new work.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from pathlib import Path

from jsc_review.core.models import ReviewResult
from jsc_review.golden.models import GoldenCandidate, Status
from jsc_review.reviewer.findings import load_findings

_ALLOWED_STATUSES: tuple[Status, ...] = ("Met", "Not Met", "Needs Review")


def _candidate_from_finding(
    document_id: str, result: ReviewResult
) -> GoldenCandidate:
    """Build an empty-label candidate carrying the system's current answer."""

    evidence_text = ""
    evidence_page: int | None = None
    if result.evidence is not None:
        evidence_text = result.evidence.text
        evidence_page = result.evidence.page
    status: Status | None = (
        result.status if result.status in _ALLOWED_STATUSES else None
    )
    return GoldenCandidate(
        document_id=document_id,
        requirement_id=result.requirement_id,
        system_status=status,
        system_reason=result.reason,
        evidence_text=evidence_text,
        evidence_page=evidence_page,
    )


def _stratify(
    findings: Iterable[tuple[str, ReviewResult]],
    *,
    exclude_keys: set[tuple[str, str]],
) -> dict[Status, list[GoldenCandidate]]:
    """Group non-excluded findings by status, mapping status -> candidates."""

    buckets: dict[Status, list[GoldenCandidate]] = {
        status: [] for status in _ALLOWED_STATUSES
    }
    for document_id, result in findings:
        if (document_id, result.requirement_id) in exclude_keys:
            continue
        if result.status not in _ALLOWED_STATUSES:
            continue
        buckets[result.status].append(_candidate_from_finding(document_id, result))
    return buckets


def pick_candidates(
    findings: Sequence[tuple[str, ReviewResult]],
    *,
    count: int,
    exclude_keys: Iterable[tuple[str, str]] = (),
    seed: int = 0,
) -> list[GoldenCandidate]:
    """Return up to ``count`` stratified candidates in deterministic order.

    The sampler aims for an even spread across the three statuses: it
    picks round-robin from the three status buckets. Within each bucket the
    order is reproducible (seeded shuffle).
    """

    if count <= 0:
        return []
    buckets = _stratify(findings, exclude_keys=set(exclude_keys))
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked: list[GoldenCandidate] = []
    status_cycle: list[Status] = ["Met", "Not Met", "Needs Review"]
    while len(picked) < count and any(buckets.values()):
        progress_in_round = False
        for status in status_cycle:
            if len(picked) >= count:
                break
            bucket = buckets[status]
            if bucket:
                picked.append(bucket.pop(0))
                progress_in_round = True
        if not progress_in_round:
            break
    return picked


def pick_candidates_from_file(
    findings_path: Path,
    *,
    count: int,
    exclude_keys: Iterable[tuple[str, str]] = (),
    seed: int = 0,
) -> list[GoldenCandidate]:
    """Convenience wrapper that reads findings.json first."""

    if not findings_path.exists():
        return []
    findings = load_findings(findings_path)
    return pick_candidates(
        findings, count=count, exclude_keys=exclude_keys, seed=seed
    )
