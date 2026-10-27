"""Hit@5 scorer: for each golden candidate with a pinned ground-truth page,
run the hybrid retrieval pipeline and check whether the ground-truth page
appears in the top-5 results.

The scorer is deliberately dependency-light: it accepts the loaded BM25 and
optional dense indexes plus the requirement register. Everything runs on
this machine; nothing is sent to any remote service.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from jsc_review.golden.models import GoldenSet
from jsc_review.retrieval.bm25 import Bm25Index
from jsc_review.retrieval.dense import DenseIndex
from jsc_review.retrieval.search import HybridConfig, hybrid_search
from jsc_review.review.findings import Requirement
from jsc_review.review.registry import load_selected_requirements


@dataclass(frozen=True)
class HitAt5Result:
    """Running counts after scoring every candidate with a pinned page."""

    hits: int
    scored: int
    skipped_no_page: int
    skipped_unknown_requirement: int

    @property
    def ratio(self) -> float | None:
        return (self.hits / self.scored) if self.scored else None


def compute_hit_at_5(
    golden_set: GoldenSet,
    *,
    bm25: Bm25Index,
    requirements: Mapping[str, Requirement],
    dense: DenseIndex | None = None,
    top_k: int = 5,
) -> HitAt5Result:
    """For each candidate with a pinned page, check if it is in the top-k.

    The pipeline uses the exact same ``hybrid_search`` the review loop uses,
    so the measurement reflects production behaviour (BM25 + optional dense
    + cue augmentation + per-page dedup).
    """

    config = HybridConfig(final_top_k=top_k, dedupe_by_page=True)
    hits = scored = skipped_no_page = skipped_unknown_requirement = 0
    for candidate in golden_set.candidates:
        if candidate.ground_truth_page is None:
            skipped_no_page += 1
            continue
        requirement = requirements.get(candidate.requirement_id)
        if requirement is None:
            skipped_unknown_requirement += 1
            continue
        results = hybrid_search(
            requirement,
            candidate.document_id,
            bm25=bm25,
            dense=dense,
            config=config,
        )
        scored += 1
        if any(hit.page == candidate.ground_truth_page for hit in results):
            hits += 1
    return HitAt5Result(
        hits=hits,
        scored=scored,
        skipped_no_page=skipped_no_page,
        skipped_unknown_requirement=skipped_unknown_requirement,
    )


def requirements_by_id(path: Path) -> dict[str, Requirement]:
    """Load the register and return a {requirement_id: Requirement} map."""

    if not path.exists():
        return {}
    return {req.requirement_id: req for req in load_selected_requirements(path)}


def hit_at_5_notes(result: HitAt5Result, total_golden: int) -> str:
    """Human-readable notes for the scorecard drawer."""

    bits: list[str] = []
    bits.append(f"scored {result.scored}/{total_golden} candidates")
    if result.skipped_no_page:
        bits.append(f"skipped {result.skipped_no_page} with no pinned page")
    if result.skipped_unknown_requirement:
        bits.append(
            f"skipped {result.skipped_unknown_requirement} with unknown requirement"
        )
    return ", ".join(bits)


__all__ = [
    "HitAt5Result",
    "compute_hit_at_5",
    "hit_at_5_notes",
    "requirements_by_id",
]
