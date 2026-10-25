"""Tests for the local dense index. A fake encoder keeps the tests offline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from jsc_review.retrieval.bm25 import IndexedChunk
from jsc_review.retrieval.dense import build_index, load_index, save_index


class _FakeEncoder:
    """Deterministic three-dimensional encoder for test chunks."""

    _TABLE = {
        "chunk_a": [1.0, 0.0, 0.0],
        "chunk_b": [0.0, 1.0, 0.0],
        "chunk_c": [0.0, 0.0, 1.0],
        "chunk_d": [0.9, 0.1, 0.0],
        "query_a": [1.0, 0.0, 0.0],
        "query_b": [0.0, 1.0, 0.0],
    }

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        rows = [self._TABLE.get(text, [0.0, 0.0, 0.0]) for text in texts]
        return np.asarray(rows, dtype=np.float32)


def _chunk(chunk_id: str, document_id: str = "doc1", page: int = 1) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page=page,
        section="Body",
        text=chunk_id,
        extraction_method="pymupdf-text",
    )


def test_dense_search_returns_the_closest_vector() -> None:
    encoder = _FakeEncoder()
    chunks = [_chunk("chunk_a"), _chunk("chunk_b"), _chunk("chunk_c")]
    index = build_index(chunks, encoder)
    hits = index.search("query_a", k=2)
    assert hits
    assert hits[0].chunk_id == "chunk_a"


def test_dense_search_filters_by_document_id() -> None:
    encoder = _FakeEncoder()
    chunks = [
        _chunk("chunk_a", document_id="doc1"),
        _chunk("chunk_b", document_id="doc2"),
        _chunk("chunk_d", document_id="doc2"),
    ]
    index = build_index(chunks, encoder)
    hits = index.search("query_a", k=5, document_id="doc2")
    chunk_ids = [hit.chunk_id for hit in hits]
    assert chunk_ids[0] == "chunk_d"
    assert "chunk_a" not in chunk_ids
    assert set(chunk_ids).issubset({"chunk_b", "chunk_d"})


def test_dense_search_rejects_an_empty_query() -> None:
    encoder = _FakeEncoder()
    index = build_index([_chunk("chunk_a")], encoder)
    assert index.search("", k=5) == ()


def test_dense_search_raises_when_no_encoder_is_attached() -> None:
    encoder = _FakeEncoder()
    index = build_index([_chunk("chunk_a")], encoder)
    stripped = index.with_encoder(None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        stripped.search("query_a", k=1)


def test_dense_index_round_trips_through_disk(tmp_path: Path) -> None:
    encoder = _FakeEncoder()
    chunks = [_chunk("chunk_a"), _chunk("chunk_b"), _chunk("chunk_c")]
    index = build_index(chunks, encoder)
    saved = save_index(index, tmp_path / "dense.faiss")
    reloaded = load_index(saved).with_encoder(encoder)
    hits = reloaded.search("query_b", k=1)
    assert hits and hits[0].chunk_id == "chunk_b"
