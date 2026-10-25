"""Rank-fusion of ranked lists from different retrieval signals.

Reciprocal rank fusion is parameter-light and robust to score-scale differences
between BM25 and dense cosine similarity.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from jsc_review.retrieval.bm25 import SearchHit


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[SearchHit]],
    *,
    k: int = 60,
    top_k: int = 100,
) -> tuple[SearchHit, ...]:
    """Combine ranked lists with RRF. The returned score is the fused weight."""

    if top_k < 1:
        return ()
    weights: dict[str, float] = {}
    first_seen: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            weights[hit.chunk_id] = weights.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    ordered = sorted(
        first_seen.values(),
        key=lambda hit: (-weights[hit.chunk_id], hit.chunk_id),
    )
    return tuple(replace(hit, score=weights[hit.chunk_id]) for hit in ordered[:top_k])
