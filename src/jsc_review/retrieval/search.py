"""Composed retrieval pipeline for a (requirement, document) query.

The pipeline is intentionally parameterized so a measurement harness can
evaluate each stage in isolation (BM25 only, +dense, +rerank, +augment)
while production uses the full composition. Nothing in this file reads the
document's company metadata, so a new upload of any company is treated the
same way as the last one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jsc_review.retrieval.bm25 import Bm25Index, SearchHit
from jsc_review.retrieval.dense import DenseIndex
from jsc_review.retrieval.hybrid import deduplicate_by_page, reciprocal_rank_fusion
from jsc_review.retrieval.query import build_query
from jsc_review.retrieval.reranker import CrossEncoder, rerank

if TYPE_CHECKING:
    from jsc_review.review.findings import Requirement


@dataclass(frozen=True)
class HybridConfig:
    """Knobs for the composed hybrid retrieval pipeline.

    The defaults are set for inference on an uploaded document: ask each
    engine for a wide candidate pool, let fusion pick the best 50, let the
    reranker (when present) re-score the top 20, and finally dedupe to one
    hit per page before returning.
    """

    bm25_top_k: int = 100
    dense_top_k: int = 100
    fused_top_k: int = 50
    rerank_candidates: int = 20
    final_top_k: int = 10
    use_cue_augmentation: bool = True
    dedupe_by_page: bool = True


def _query_text(requirement: Requirement, config: HybridConfig) -> str:
    if config.use_cue_augmentation:
        return build_query(requirement)
    parts: list[str] = []
    if requirement.text_ar:
        parts.append(requirement.text_ar)
    if requirement.text_en:
        parts.append(requirement.text_en)
    return " ".join(parts).strip() or requirement.requirement_id


def hybrid_search(
    requirement: Requirement,
    document_id: str,
    *,
    bm25: Bm25Index,
    dense: DenseIndex | None = None,
    reranker: CrossEncoder | None = None,
    config: HybridConfig | None = None,
) -> tuple[SearchHit, ...]:
    """Run the full retrieval pipeline for one requirement against one document.

    The stages applied, in order:

    1. Build the query from ``requirement`` (with universal cue augmentation
       when ``config.use_cue_augmentation`` is set).
    2. Rank candidates in BM25 (always) and dense (when provided).
    3. Fuse the rankings with reciprocal rank fusion.
    4. If a reranker is given, re-score the top ``config.rerank_candidates``.
    5. If ``config.dedupe_by_page`` is set, keep only the first hit per
       ``(document_id, page)`` pair.
    6. Return the top ``config.final_top_k`` hits.
    """

    settings = config or HybridConfig()
    query = _query_text(requirement, settings)
    bm25_hits = bm25.search(query, k=settings.bm25_top_k, document_id=document_id)
    if dense is None:
        fused: tuple[SearchHit, ...] = bm25_hits[: settings.fused_top_k]
    else:
        dense_hits = dense.search(query, k=settings.dense_top_k, document_id=document_id)
        fused = reciprocal_rank_fusion([bm25_hits, dense_hits], top_k=settings.fused_top_k)
    if reranker is not None and fused:
        reranked = rerank(
            query,
            fused[: settings.rerank_candidates],
            reranker,
            top_k=settings.rerank_candidates,
        )
        # Keep any fused hits beyond the rerank window at the back of the list.
        remainder = tuple(fused[settings.rerank_candidates :])
        fused = reranked + remainder
    if settings.dedupe_by_page:
        return deduplicate_by_page(fused, top_k=settings.final_top_k)
    return tuple(fused[: settings.final_top_k])
