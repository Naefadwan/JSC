"""Phase 4 scorecard endpoint.

Reads the golden set and the current findings file, computes every metric
in :mod:`jsc_review.metrics.scorecard`, and (when ``include_hit_at_5`` is
true) runs the hybrid retriever against every candidate with a pinned
ground-truth page.

Everything is local; no outside call is made.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from jsc_review.core.models import ReviewResult
from jsc_review.core.settings import Settings, get_settings
from jsc_review.golden.models import GoldenSet
from jsc_review.golden.store import load_golden_set
from jsc_review.metrics.hit_at_5 import (
    compute_hit_at_5,
    hit_at_5_notes,
    requirements_by_id,
)
from jsc_review.metrics.scorecard import Scorecard, score_findings_against_golden
from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval import dense as dense_module
from jsc_review.retrieval.bm25 import Bm25Index
from jsc_review.retrieval.dense import DenseIndex
from jsc_review.review.findings import Requirement
from jsc_review.reviewer.findings import load_findings

router = APIRouter(prefix="/reviewer", tags=["reviewer", "scorecard"])


def _load_findings_safe(path: Path) -> list[tuple[str, ReviewResult]]:
    """Return findings or an empty list when the file is missing or empty."""

    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return load_findings(path)


def _try_hit_at_5(
    golden_set: GoldenSet, settings: Settings
) -> tuple[float | None, int, str]:
    """Compute Hit@5 against the current BM25 (+ optional dense) indexes.

    Returns ``(value, sample_size, notes)``. Any missing dependency degrades
    into a note instead of a crash so the drawer always renders.
    """

    if not any(cand.ground_truth_page for cand in golden_set.candidates):
        return (
            None,
            0,
            "No candidates have a pinned ground-truth page yet. Pin a page in the Golden drawer.",
        )
    if not settings.bm25_index_path.exists():
        return (None, 0, "BM25 index file is missing.")
    requirements = requirements_by_id(settings.requirements_register_path)
    if not requirements:
        return (
            None,
            0,
            "Requirements register has no selected requirements to query against.",
        )
    try:
        bm25 = bm25_module.load_index(settings.bm25_index_path)
    except (OSError, ValueError, KeyError) as exc:
        return (None, 0, f"Could not load BM25 index: {exc}")
    dense = None
    if settings.use_dense_at_review and settings.dense_index_path.exists():
        try:
            dense = dense_module.load_index(settings.dense_index_path)
            from jsc_review.retrieval.embedding import (
                SentenceTransformerEncoder,
                embedding_settings,
            )

            dense = dense.with_encoder(
                SentenceTransformerEncoder(embedding_settings())
            )
        except Exception as exc:  # noqa: BLE001  # best-effort fallback
            dense = None
            return _run_hit_at_5(
                golden_set,
                bm25,
                requirements,
                dense,
                extra_note=f"Dense retrieval skipped: {exc}",
            )
    return _run_hit_at_5(golden_set, bm25, requirements, dense)


def _run_hit_at_5(
    golden_set: GoldenSet,
    bm25: Bm25Index,
    requirements: dict[str, Requirement],
    dense: DenseIndex | None,
    extra_note: str = "",
) -> tuple[float | None, int, str]:
    result = compute_hit_at_5(
        golden_set,
        bm25=bm25,
        requirements=requirements,
        dense=dense,
        top_k=5,
    )
    notes = hit_at_5_notes(result, total_golden=len(golden_set.candidates))
    if extra_note:
        notes = f"{notes}. {extra_note}"
    return (result.ratio, result.scored, notes)


@router.get("/scorecard", response_model=Scorecard)
def scorecard(
    include_hit_at_5: bool = Query(
        default=False,
        description=(
            "When true, run the hybrid retriever over the golden set to compute "
            "Hit@5. Costs one search per pinned candidate (seconds, not minutes)."
        ),
    ),
) -> Scorecard:
    """Return the Phase 4 scorecard built from local findings + golden set."""

    settings = get_settings()
    try:
        golden_set = load_golden_set(settings.golden_labels_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not read golden set: {exc}"
        ) from exc
    findings = _load_findings_safe(settings.findings_path)
    hit_value: float | None = None
    hit_sample = 0
    hit_notes = ""
    if include_hit_at_5:
        hit_value, hit_sample, hit_notes = _try_hit_at_5(golden_set, settings)
    return score_findings_against_golden(
        findings,
        golden_set,
        hit_at_5_value=hit_value,
        hit_at_5_sample=hit_sample,
        hit_at_5_notes=hit_notes,
    )
