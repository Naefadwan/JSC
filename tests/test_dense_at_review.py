"""Tests for the opt-in dense-retrieval hook in the review pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from jsc_review.documents.pipeline import _maybe_load_dense_for_review


def test_disabled_returns_none(tmp_path: Path) -> None:
    assert (
        _maybe_load_dense_for_review(
            use_dense_at_review=False,
            dense_index_path=tmp_path / "does-not-matter.faiss",
            document_id="doc1",
            progress=None,
        )
        is None
    )


def test_missing_index_returns_none(tmp_path: Path) -> None:
    assert (
        _maybe_load_dense_for_review(
            use_dense_at_review=True,
            dense_index_path=tmp_path / "nope.faiss",
            document_id="doc1",
            progress=None,
        )
        is None
    )


def test_corrupt_index_degrades_to_none(tmp_path: Path) -> None:
    """A broken index file must not raise; dense-at-review is best-effort."""

    bad_path = tmp_path / "broken.faiss"
    bad_path.write_bytes(b"not a faiss index")
    assert (
        _maybe_load_dense_for_review(
            use_dense_at_review=True,
            dense_index_path=bad_path,
            document_id="doc1",
            progress=None,
        )
        is None
    )


def test_document_not_in_index_returns_none(tmp_path: Path) -> None:
    """If the dense index has no chunks for this document, skip the lookup."""

    from jsc_review.retrieval import dense as dense_module
    from jsc_review.retrieval.bm25 import IndexedChunk

    # Build a fake dense index with a single chunk from a different document.
    chunk = IndexedChunk(
        chunk_id="other-1",
        document_id="other-doc",
        page=1,
        section="",
        text="hello",
        extraction_method="embedded",
    )

    class _FakeEncoder:
        def encode(self, texts):  # type: ignore[no-untyped-def]
            import numpy as np

            return np.ones((len(texts), 4), dtype="float32")

    index = dense_module.build_index([chunk], _FakeEncoder())
    index_path = tmp_path / "dense.faiss"
    dense_module.save_index(index, index_path)

    # Even with the encoder reachable, we must bail because the queried
    # document has no chunks in the index.
    result = _maybe_load_dense_for_review(
        use_dense_at_review=True,
        dense_index_path=index_path,
        document_id="doc-not-in-index",
        progress=None,
    )
    assert result is None


def test_encoder_load_failure_degrades_to_none(tmp_path: Path) -> None:
    """A missing or failing encoder must not break the review loop."""

    from jsc_review.retrieval import dense as dense_module
    from jsc_review.retrieval.bm25 import IndexedChunk

    chunk = IndexedChunk(
        chunk_id="doc1-1",
        document_id="doc1",
        page=1,
        section="",
        text="hello",
        extraction_method="embedded",
    )

    class _FakeEncoder:
        def encode(self, texts):  # type: ignore[no-untyped-def]
            import numpy as np

            return np.ones((len(texts), 4), dtype="float32")

    dense_module.save_index(
        dense_module.build_index([chunk], _FakeEncoder()),
        tmp_path / "dense.faiss",
    )

    with patch(
        "jsc_review.retrieval.embedding.SentenceTransformerEncoder",
        side_effect=RuntimeError("model missing"),
    ):
        result = _maybe_load_dense_for_review(
            use_dense_at_review=True,
            dense_index_path=tmp_path / "dense.faiss",
            document_id="doc1",
            progress=None,
        )
    assert result is None
