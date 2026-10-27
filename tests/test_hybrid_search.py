"""Tests for the composed hybrid retrieval pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from jsc_review.retrieval.bm25 import Bm25Index, IndexedChunk, SearchHit, build_index
from jsc_review.retrieval.hybrid import deduplicate_by_page
from jsc_review.retrieval.query import build_query, cue_terms_for
from jsc_review.retrieval.search import HybridConfig, hybrid_search
from jsc_review.review.findings import Requirement


def _chunk(doc: str, page: int, suffix: str, text: str) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=f"{doc}:p{page}:{suffix}",
        document_id=doc,
        page=page,
        section="",
        text=text,
        extraction_method="pymupdf-text",
    )


def _hit(doc: str, page: int, suffix: str, score: float = 1.0) -> SearchHit:
    return SearchHit(
        chunk_id=f"{doc}:p{page}:{suffix}",
        document_id=doc,
        page=page,
        section="",
        text="",
        extraction_method="pymupdf-text",
        score=score,
    )


def _requirement(
    requirement_id: str, text_ar: str = "", text_en: str = "", method: str = "llm"
) -> Requirement:
    return Requirement(
        requirement_id=requirement_id,
        section="بند",
        text_ar=text_ar,
        text_en=text_en,
        evaluation_method=method,
    )


# --- page deduplication ----------------------------------------------------


def test_deduplicate_by_page_keeps_first_hit_per_page() -> None:
    hits = [
        _hit("doc", 108, "c01"),
        _hit("doc", 109, "c01"),
        _hit("doc", 109, "c02"),
        _hit("doc", 108, "c02"),
        _hit("doc", 110, "c01"),
    ]

    deduped = deduplicate_by_page(hits, top_k=3)

    assert tuple(hit.page for hit in deduped) == (108, 109, 110)


def test_deduplicate_by_page_respects_document_boundary() -> None:
    hits = [
        _hit("a", 1, "c01"),
        _hit("b", 1, "c01"),
        _hit("a", 1, "c02"),  # duplicate page within doc a
    ]

    deduped = deduplicate_by_page(hits, top_k=5)

    assert [(hit.document_id, hit.page) for hit in deduped] == [("a", 1), ("b", 1)]


# --- query augmentation ----------------------------------------------------


def test_cue_terms_for_heading_requirement() -> None:
    requirement = _requirement("DIS-064", "قائمة التدفقات النقدية", "cash flow statement")

    cues = cue_terms_for(requirement)

    assert "قائمة التدفقات النقدية" in cues
    assert "statement of cash flows" in cues


def test_cue_terms_for_threshold_requirement_pulls_owner_cues() -> None:
    requirement = _requirement("DIS-010", "كبار المساهمين", "major shareholders")

    cues = cue_terms_for(requirement)

    assert any("كبار المساهم" in cue for cue in cues)


def test_cue_terms_for_customer_requirement_pulls_customer_cues() -> None:
    requirement = _requirement("DIS-017", "الاعتماد على الموردين", "customer dependence")

    cues = cue_terms_for(requirement)

    assert any("عملاء" in cue or "موردين" in cue for cue in cues)


def test_cue_terms_for_unknown_requirement_is_empty() -> None:
    # DIS-999 is not registered under any deterministic or cue rule, so the
    # universal cue vocabulary should be empty. Picking a reserved id keeps
    # the test independent of future HEADING_VARIANTS additions.
    requirement = _requirement("DIS-999", "متطلب غير معرّف", "unregistered requirement")

    assert cue_terms_for(requirement) == ()


def test_build_query_appends_cues_once() -> None:
    requirement = _requirement("DIS-064", "قائمة التدفقات النقدية", "cash flow statement")

    query = build_query(requirement)

    # Base text appears before the first cue; the universal English cue is present.
    assert query.index("قائمة التدفقات النقدية") < query.index("statement of cash flows")


# --- hybrid pipeline -------------------------------------------------------


def _page_flood_index() -> Bm25Index:
    """A page with many near-duplicate chunks on the same topic.

    This is the pattern that caused the `164_10163 DIS-011` miss in the
    real retrieval gate: a single long page carries many chunks that all
    match the query, so without dedup they occupy every top-k slot.
    """

    return build_index(
        [
            _chunk("doc", 108, "c01", "notes on operations and procurement"),
            _chunk("doc", 108, "c02", "notes continued on operations and procurement"),
            _chunk("doc", 108, "c03", "notes continued further on operations"),
            _chunk("doc", 109, "c01", "notes on segment reporting"),
            _chunk("doc", 109, "c02", "notes on segment reporting continued"),
            _chunk("doc", 44, "c01", "shareholder notes on page forty-four"),
            _chunk("doc", 55, "c01", "customer notes on page fifty-five"),
        ]
    )


def test_hybrid_search_dedupe_shrinks_page_flooding() -> None:
    index = _page_flood_index()
    requirement = _requirement("GOV-999", "notes", "notes")

    without = hybrid_search(
        requirement,
        "doc",
        bm25=index,
        dense=None,
        config=HybridConfig(dedupe_by_page=False, final_top_k=5),
    )
    with_dedup = hybrid_search(
        requirement,
        "doc",
        bm25=index,
        dense=None,
        config=HybridConfig(dedupe_by_page=True, final_top_k=5),
    )

    # Without dedup, 108 and 109 flood the top slots.
    assert len({hit.page for hit in without}) < len({hit.page for hit in with_dedup})
    # With dedup, every slot is a distinct page.
    assert len({hit.page for hit in with_dedup}) == len(with_dedup)


class _StubReranker:
    """A reranker that boosts hits whose passage text contains ``prefer``."""

    def __init__(self, prefer: str) -> None:
        self._prefer = prefer

    def predict(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        scores = [10.0 if self._prefer in passage else 0.0 for _, passage in pairs]
        return np.asarray(scores, dtype=np.float32)


def test_hybrid_search_reranker_promotes_the_preferred_passage() -> None:
    """The reranker must be able to pull a buried-but-present hit up to rank 1."""

    index = build_index(
        [
            _chunk("doc", 108, "c01", "shareholders own stake notes on operations"),
            _chunk("doc", 108, "c02", "notes continued on operations"),
            _chunk("doc", 109, "c01", "notes on segment reporting"),
            _chunk("doc", 44, "c01", "shareholders table KEYSTONE"),
            _chunk("doc", 44, "c02", "shareholders table continued"),
        ]
    )
    requirement = _requirement("GOV-999", "shareholders notes", "shareholders notes")

    without = hybrid_search(
        requirement,
        "doc",
        bm25=index,
        dense=None,
        config=HybridConfig(dedupe_by_page=True, final_top_k=3),
    )
    assert without[0].page != 44  # the longer page 108 dominates BM25

    with_rerank = hybrid_search(
        requirement,
        "doc",
        bm25=index,
        dense=None,
        reranker=_StubReranker("KEYSTONE"),
        config=HybridConfig(dedupe_by_page=True, final_top_k=3, rerank_candidates=10),
    )
    assert with_rerank[0].page == 44  # KEYSTONE boost promotes p44


def test_hybrid_search_falls_back_to_bm25_when_dense_is_none() -> None:
    index = _page_flood_index()
    requirement = _requirement("GOV-999", "shareholder forty-four", "shareholder forty-four")

    hits = hybrid_search(
        requirement,
        "doc",
        bm25=index,
        dense=None,
        config=HybridConfig(dedupe_by_page=True, final_top_k=5),
    )

    assert hits
    assert hits[0].page == 44
