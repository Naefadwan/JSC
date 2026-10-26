"""Rank-fusion and page-level deduplication of ranked passage lists.

Reciprocal rank fusion is parameter-light and robust to score-scale differences
between BM25 and dense cosine similarity. Page deduplication keeps the top-k
diverse so a verbose page cannot occupy every slot through near-duplicate
chunks.
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


def deduplicate_by_page(hits: Iterable[SearchHit], *, top_k: int = 10) -> tuple[SearchHit, ...]:
    """Keep only the first hit for each ``(document_id, page)`` pair.

    When a single page carries many chunks (long notes, multi-column tables)
    the top-k is otherwise flooded with near-duplicates. For a page-level
    retrieval target like ``Hit@5`` on pages, this makes every slot count a
    distinct page.
    """

    if top_k < 1:
        return ()
    seen: set[tuple[str, int]] = set()
    kept: list[SearchHit] = []
    for hit in hits:
        key = (hit.document_id, hit.page)
        if key in seen:
            continue
        seen.add(key)
        kept.append(hit)
        if len(kept) >= top_k:
            break
    return tuple(kept)
