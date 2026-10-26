"""Append-chunks helpers for BM25 and dense indexes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from jsc_review.retrieval.bm25 import IndexedChunk, append_chunks, build_index
from jsc_review.retrieval.dense import DenseIndex
from jsc_review.retrieval.dense import append_chunks as dense_append
from jsc_review.retrieval.dense import build_index as dense_build


class _StubEncoder:
    """Deterministic encoder that produces unique 4D vectors per text."""

    def __init__(self) -> None:
        self._cache: dict[str, NDArray[np.float32]] = {}

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        rows: list[NDArray[np.float32]] = []
        for text in texts:
            vector = self._cache.get(text)
            if vector is None:
                hashed = abs(hash(text)) % (10**6)
                vector = np.array(
                    [
                        (hashed % 97) / 97.0,
                        ((hashed // 97) % 97) / 97.0,
                        ((hashed // (97 * 97)) % 97) / 97.0,
                        1.0,
                    ],
                    dtype=np.float32,
                )
                self._cache[text] = vector
            rows.append(vector)
        return np.vstack(rows).astype(np.float32)


def _chunk(chunk_id: str, text: str, document_id: str = "doc_a", page: int = 1) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page=page,
        section="",
        text=text,
        extraction_method="pymupdf-text",
    )


# --- BM25 -------------------------------------------------------------------


def test_bm25_append_adds_new_chunks_and_rebuilds_postings() -> None:
    """A new chunk becomes searchable and the chunk list grows."""

    base = build_index([_chunk("c1", "shareholders own ten percent")])
    assert base.search("shareholders", k=1)[0].chunk_id == "c1"

    updated = append_chunks(base, [_chunk("c2", "cash flow statement for the year")])
    assert len(updated.chunks) == 2
    hit = updated.search("cash flow", k=1)
    assert hit and hit[0].chunk_id == "c2"


def test_bm25_append_is_idempotent_on_duplicate_chunk_ids() -> None:
    """Repeating the same chunk id leaves the index unchanged."""

    base = build_index([_chunk("c1", "major shareholders disclosure")])
    unchanged = append_chunks(base, [_chunk("c1", "different text under same id")])
    assert len(unchanged.chunks) == 1
    assert unchanged.chunks[0].text == "major shareholders disclosure"


def test_bm25_append_on_empty_input_returns_the_same_object() -> None:
    """Appending an empty sequence short-circuits."""

    base = build_index([_chunk("c1", "hello world")])
    assert append_chunks(base, []) is base


# --- Dense ------------------------------------------------------------------


def test_dense_append_adds_vectors_and_keeps_them_searchable() -> None:
    """A new chunk becomes retrievable through the FAISS index."""

    encoder = _StubEncoder()
    base = dense_build([_chunk("c1", "shareholders ten percent")], encoder, batch_size=4)
    base = base.with_encoder(encoder)
    assert base.search("shareholders ten percent", k=1)[0].chunk_id == "c1"

    updated = dense_append(
        base,
        [_chunk("c2", "cash flow statement for the year")],
        encoder,
        batch_size=4,
    )
    assert len(updated.chunks) == 2
    hit = updated.search("cash flow statement for the year", k=1)
    assert hit and hit[0].chunk_id == "c2"


def test_dense_append_is_idempotent_on_duplicate_chunk_ids() -> None:
    """Repeating the same chunk id does not duplicate vectors."""

    encoder = _StubEncoder()
    base = dense_build([_chunk("c1", "hello")], encoder, batch_size=4)
    base = base.with_encoder(encoder)
    unchanged = dense_append(base, [_chunk("c1", "hello again")], encoder, batch_size=4)
    assert len(unchanged.chunks) == 1
    assert base.faiss_index.ntotal == 1


def test_dense_append_rejects_mismatched_vector_dimensions() -> None:
    """A different-dim encoder output surfaces as a ValueError."""

    import faiss

    base = DenseIndex(
        chunks=(_chunk("c1", "a"),),
        faiss_index=faiss.IndexFlatIP(4),
    )
    base.faiss_index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    class BadEncoder:
        def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
            return np.zeros((len(texts), 7), dtype=np.float32)

    with pytest.raises(ValueError, match="dim"):
        dense_append(base, [_chunk("c2", "b")], BadEncoder(), batch_size=4)
