"""Tests for the global-index backfill helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval import dense as dense_module
from jsc_review.retrieval.backfill import (
    apply_backfill,
    chunks_by_document,
    document_ids,
    plan_backfill,
)
from jsc_review.retrieval.bm25 import IndexedChunk, build_index, save_index
from jsc_review.retrieval.dense import save_index as save_dense


class _StubEncoder:
    """Deterministic encoder that returns the same 4D vector per text."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        rows = [
            np.array(
                [
                    (abs(hash(text)) % 97) / 97.0,
                    ((abs(hash(text)) // 97) % 97) / 97.0,
                    ((abs(hash(text)) // (97 * 97)) % 97) / 97.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            for text in texts
        ]
        return np.vstack(rows).astype(np.float32)


def _chunk(doc: str, page: int, suffix: str, text: str) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=f"{doc}:p{page}:{suffix}",
        document_id=doc,
        page=page,
        section="",
        text=text,
        extraction_method="pymupdf-text",
    )


def _write_chunks(path: Path, document_id: str, chunks: list[IndexedChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "chunks": [chunk.__dict__ for chunk in chunks],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_document_ids_and_grouping() -> None:
    chunks = [
        _chunk("a", 1, "c01", "alpha"),
        _chunk("a", 2, "c01", "beta"),
        _chunk("b", 1, "c01", "gamma"),
    ]

    assert document_ids(chunks) == {"a", "b"}
    grouped = chunks_by_document(chunks)
    assert list(grouped) == ["a", "b"]
    assert [chunk.page for chunk in grouped["a"]] == [1, 2]


def test_plan_backfill_lists_only_missing_documents(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir / "a.json", "a", [_chunk("a", 1, "c01", "alpha")])
    _write_chunks(chunks_dir / "b.json", "b", [_chunk("b", 1, "c01", "beta")])

    bm25_index = build_index([_chunk("a", 1, "c01", "alpha")])

    plan = plan_backfill(
        chunks_dir=chunks_dir,
        bm25_index=bm25_index,
        dense_index=None,
    )

    assert plan.bm25_missing_documents == ("b",)
    assert plan.bm25_chunk_count == 1
    assert plan.dense_missing_documents == ()
    assert plan.dense_chunk_count == 0


def test_plan_backfill_respects_only_documents_filter(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir / "a.json", "a", [_chunk("a", 1, "c01", "alpha")])
    _write_chunks(chunks_dir / "b.json", "b", [_chunk("b", 1, "c01", "beta")])
    _write_chunks(chunks_dir / "c.json", "c", [_chunk("c", 1, "c01", "gamma")])

    bm25_index = build_index([])

    plan = plan_backfill(
        chunks_dir=chunks_dir,
        bm25_index=bm25_index,
        dense_index=None,
        only_documents=["b"],
    )

    assert plan.bm25_missing_documents == ("b",)
    assert plan.bm25_chunk_count == 1


def test_apply_backfill_appends_only_missing_to_bm25(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir / "a.json", "a", [_chunk("a", 1, "c01", "alpha text")])
    _write_chunks(chunks_dir / "b.json", "b", [_chunk("b", 1, "c01", "beta disclosure")])

    bm25_path = tmp_path / "chunk_bm25.json"
    save_index(build_index([_chunk("a", 1, "c01", "alpha text")]), bm25_path)

    report = apply_backfill(
        chunks_dir=chunks_dir,
        bm25_index_path=bm25_path,
        dense_index_path=None,
        encoder=None,
    )

    assert report.bm25_added_documents == ("b",)
    assert report.bm25_added_chunks == 1
    reloaded = bm25_module.load_index(bm25_path)
    assert document_ids(reloaded.chunks) == {"a", "b"}
    assert reloaded.search("disclosure", k=1)[0].document_id == "b"


def test_apply_backfill_is_a_no_op_when_everything_is_present(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir / "a.json", "a", [_chunk("a", 1, "c01", "alpha text")])

    bm25_path = tmp_path / "chunk_bm25.json"
    save_index(build_index([_chunk("a", 1, "c01", "alpha text")]), bm25_path)

    report = apply_backfill(
        chunks_dir=chunks_dir,
        bm25_index_path=bm25_path,
        dense_index_path=None,
        encoder=None,
    )

    assert report.bm25_added_documents == ()
    assert report.bm25_added_chunks == 0


def test_apply_backfill_appends_to_dense_index_when_encoder_present(
    tmp_path: Path,
) -> None:
    encoder = _StubEncoder()
    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir / "a.json", "a", [_chunk("a", 1, "c01", "alpha text")])
    _write_chunks(chunks_dir / "b.json", "b", [_chunk("b", 1, "c01", "beta text")])

    bm25_path = tmp_path / "chunk_bm25.json"
    save_index(build_index([_chunk("a", 1, "c01", "alpha text")]), bm25_path)
    dense_path = tmp_path / "chunk_dense.faiss"
    dense_index = dense_module.build_index(
        [_chunk("a", 1, "c01", "alpha text")], encoder, batch_size=4
    )
    save_dense(dense_index, dense_path)

    report = apply_backfill(
        chunks_dir=chunks_dir,
        bm25_index_path=bm25_path,
        dense_index_path=dense_path,
        encoder=encoder,
    )

    assert report.bm25_added_documents == ("b",)
    assert report.dense_added_documents == ("b",)
    assert report.dense_added_chunks == 1
    reloaded = dense_module.load_index(dense_path)
    assert document_ids(reloaded.chunks) == {"a", "b"}


def test_apply_backfill_dense_requires_encoder_when_dense_path_exists(
    tmp_path: Path,
) -> None:
    encoder = _StubEncoder()
    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir / "a.json", "a", [_chunk("a", 1, "c01", "alpha text")])

    bm25_path = tmp_path / "chunk_bm25.json"
    save_index(build_index([_chunk("a", 1, "c01", "alpha text")]), bm25_path)
    dense_path = tmp_path / "chunk_dense.faiss"
    dense_index = dense_module.build_index(
        [_chunk("a", 1, "c01", "alpha text")], encoder, batch_size=4
    )
    save_dense(dense_index, dense_path)

    try:
        apply_backfill(
            chunks_dir=chunks_dir,
            bm25_index_path=bm25_path,
            dense_index_path=dense_path,
            encoder=None,
        )
    except ValueError as exc:
        assert "encoder" in str(exc).lower()
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError when encoder is missing")
