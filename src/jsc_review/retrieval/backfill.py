"""Append missing documents to the global BM25 and dense retrieval indexes.

The upload pipeline builds a per-document BM25 index at ingest time and
best-effort appends its chunks to the global BM25 and dense indexes. When
that best-effort append is skipped (for example when the dense encoder
cannot load) the global indexes end up missing the newly uploaded report
and the batch runner returns no passages for it.

This module scans ``chunks_with_ocr`` for ready-to-index chunks per
document, compares the set of document ids against the global indexes, and
appends only the missing ones. The design is intentionally idempotent: a
rerun on a complete index is a no-op.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval import dense as dense_module
from jsc_review.retrieval.bm25 import Bm25Index, IndexedChunk, load_chunk_directory
from jsc_review.retrieval.dense import DenseIndex, TextEncoder


@dataclass(frozen=True)
class BackfillPlan:
    """What a backfill run would add to each global index."""

    bm25_missing_documents: tuple[str, ...]
    bm25_chunk_count: int
    dense_missing_documents: tuple[str, ...]
    dense_chunk_count: int


@dataclass(frozen=True)
class BackfillReport:
    """Counts after an append-and-save run."""

    bm25_added_documents: tuple[str, ...]
    bm25_added_chunks: int
    dense_added_documents: tuple[str, ...]
    dense_added_chunks: int


def document_ids(chunks: Iterable[IndexedChunk]) -> set[str]:
    """Return the distinct document ids in a chunk iterable."""

    return {chunk.document_id for chunk in chunks}


def chunks_by_document(
    chunks: Iterable[IndexedChunk],
) -> dict[str, list[IndexedChunk]]:
    """Group chunks by document id, preserving input order within each group."""

    grouped: dict[str, list[IndexedChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)
    return grouped


def _select_missing(
    grouped: dict[str, list[IndexedChunk]],
    present: set[str],
    only_documents: Iterable[str] | None,
) -> list[IndexedChunk]:
    wanted = set(only_documents) if only_documents is not None else None
    selected: list[IndexedChunk] = []
    for document_id in sorted(grouped):
        if wanted is not None and document_id not in wanted:
            continue
        if document_id in present:
            continue
        selected.extend(grouped[document_id])
    return selected


def plan_backfill(
    *,
    chunks_dir: Path,
    bm25_index: Bm25Index,
    dense_index: DenseIndex | None,
    only_documents: Iterable[str] | None = None,
) -> BackfillPlan:
    """Describe which documents and how many chunks a backfill would add."""

    all_chunks = load_chunk_directory(chunks_dir)
    grouped = chunks_by_document(all_chunks)
    bm25_present = document_ids(bm25_index.chunks)
    bm25_missing = _select_missing(grouped, bm25_present, only_documents)
    bm25_docs = tuple(sorted({chunk.document_id for chunk in bm25_missing}))

    dense_docs: tuple[str, ...] = ()
    dense_missing_count = 0
    if dense_index is not None:
        dense_present = document_ids(dense_index.chunks)
        dense_missing = _select_missing(grouped, dense_present, only_documents)
        dense_docs = tuple(sorted({chunk.document_id for chunk in dense_missing}))
        dense_missing_count = len(dense_missing)

    return BackfillPlan(
        bm25_missing_documents=bm25_docs,
        bm25_chunk_count=len(bm25_missing),
        dense_missing_documents=dense_docs,
        dense_chunk_count=dense_missing_count,
    )


def apply_backfill(
    *,
    chunks_dir: Path,
    bm25_index_path: Path,
    dense_index_path: Path | None,
    encoder: TextEncoder | None,
    only_documents: Iterable[str] | None = None,
    bm25_module_: object = bm25_module,
    dense_module_: object = dense_module,
) -> BackfillReport:
    """Append missing chunks to the on-disk indexes and persist both.

    ``encoder`` is required when ``dense_index_path`` is given. When it is
    ``None`` the dense index is left untouched. The ``*_module_`` kwargs
    exist to let tests inject fake load/save/append callables.
    """

    all_chunks = load_chunk_directory(chunks_dir)
    grouped = chunks_by_document(all_chunks)

    bm25_index: Bm25Index = bm25_module_.load_index(bm25_index_path)  # type: ignore[attr-defined]
    bm25_present = document_ids(bm25_index.chunks)
    bm25_additions = _select_missing(grouped, bm25_present, only_documents)
    bm25_added_docs: tuple[str, ...] = ()
    bm25_added_chunks = 0
    if bm25_additions:
        updated_bm25 = bm25_module_.append_chunks(bm25_index, bm25_additions)  # type: ignore[attr-defined]
        bm25_module_.save_index(updated_bm25, bm25_index_path)  # type: ignore[attr-defined]
        bm25_added_docs = tuple(sorted({chunk.document_id for chunk in bm25_additions}))
        bm25_added_chunks = len(bm25_additions)

    dense_added_docs: tuple[str, ...] = ()
    dense_added_chunks = 0
    if dense_index_path is not None and dense_index_path.exists():
        if encoder is None:
            raise ValueError("An encoder is required to backfill the dense index")
        dense_index: DenseIndex = dense_module_.load_index(dense_index_path)  # type: ignore[attr-defined]
        dense_present = document_ids(dense_index.chunks)
        dense_additions = _select_missing(grouped, dense_present, only_documents)
        if dense_additions:
            updated_dense = dense_module_.append_chunks(  # type: ignore[attr-defined]
                dense_index, dense_additions, encoder
            )
            dense_module_.save_index(updated_dense, dense_index_path)  # type: ignore[attr-defined]
            dense_added_docs = tuple(sorted({chunk.document_id for chunk in dense_additions}))
            dense_added_chunks = len(dense_additions)

    return BackfillReport(
        bm25_added_documents=bm25_added_docs,
        bm25_added_chunks=bm25_added_chunks,
        dense_added_documents=dense_added_docs,
        dense_added_chunks=dense_added_chunks,
    )
