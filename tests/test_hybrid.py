"""Tests for reciprocal rank fusion over BM25 and dense ranked lists."""

from __future__ import annotations

from jsc_review.retrieval.bm25 import SearchHit
from jsc_review.retrieval.hybrid import reciprocal_rank_fusion


def _hit(chunk_id: str, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id="doc1",
        page=1,
        section="Body",
        text=f"text for {chunk_id}",
        extraction_method="pymupdf-text",
        score=score,
    )


def test_rrf_promotes_chunks_that_appear_in_both_rankings() -> None:
    bm25 = [_hit("a", 10.0), _hit("b", 9.0), _hit("c", 8.0)]
    dense = [_hit("c", 0.9), _hit("a", 0.8), _hit("d", 0.7)]
    fused = reciprocal_rank_fusion([bm25, dense], top_k=4)
    chunk_ids = [hit.chunk_id for hit in fused]
    # a and c appear in both lists; b and d only appear in one.
    assert chunk_ids[:2] == ["a", "c"]
    assert set(chunk_ids) == {"a", "b", "c", "d"}


def test_rrf_returns_at_most_top_k() -> None:
    bm25 = [_hit("a", 1.0), _hit("b", 0.9), _hit("c", 0.8)]
    dense = [_hit("d", 1.0), _hit("e", 0.9), _hit("f", 0.8)]
    fused = reciprocal_rank_fusion([bm25, dense], top_k=2)
    assert len(fused) == 2


def test_rrf_handles_an_empty_ranking() -> None:
    bm25 = [_hit("a", 1.0), _hit("b", 0.5)]
    fused = reciprocal_rank_fusion([bm25, []], top_k=3)
    assert [hit.chunk_id for hit in fused] == ["a", "b"]


def test_rrf_scores_are_the_fused_weights() -> None:
    bm25 = [_hit("a", 10.0)]
    dense = [_hit("a", 0.5)]
    fused = reciprocal_rank_fusion([bm25, dense], k=60, top_k=1)
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert fused[0].chunk_id == "a"
    assert abs(fused[0].score - expected) < 1e-12
