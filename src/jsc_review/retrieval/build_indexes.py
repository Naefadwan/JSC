"""Rebuild the global BM25 and dense indexes from the on-disk chunk directory.

Usage from the repository root::

    python -m jsc_review.retrieval.build_indexes --help
    python -m jsc_review.retrieval.build_indexes --bm25
    python -m jsc_review.retrieval.build_indexes --dense
    python -m jsc_review.retrieval.build_indexes --bm25 --dense --force

Nothing in this CLI calls out to a remote service. The dense encoder is a
locally cached sentence-transformers model (see ``dense_config.json``).

Default behaviour is incremental: when an index already exists on disk, the
CLI delegates to :func:`jsc_review.retrieval.backfill.apply_backfill` so only
missing documents are added. ``--force`` wipes the on-disk index first and
rebuilds from every chunk the directory holds, which is the right choice
after a chunk-format change or a reproducibility rerun.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from jsc_review.core.settings import get_settings
from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval import dense as dense_module
from jsc_review.retrieval.backfill import apply_backfill
from jsc_review.retrieval.bm25 import load_chunk_directory

_LOGGER = logging.getLogger("jsc_review.retrieval.build_indexes")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jsc_review.retrieval.build_indexes",
        description=(
            "Rebuild or extend the global BM25 and dense retrieval indexes "
            "from the local chunks-with-OCR directory. Nothing leaves this "
            "machine."
        ),
    )
    parser.add_argument(
        "--bm25", action="store_true", help="Build or extend the BM25 index."
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Build or extend the dense FAISS index (needs the 'retrieval' extra).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete the on-disk index first and rebuild from every chunk. "
            "Default is incremental: only missing documents are added."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        help="Override Settings.chunks_with_ocr_dir.",
    )
    parser.add_argument(
        "--bm25-path",
        type=Path,
        help="Override Settings.bm25_index_path.",
    )
    parser.add_argument(
        "--dense-path",
        type=Path,
        help="Override Settings.dense_index_path.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help=(
            "Restrict the operation to one document id. May be repeated to "
            "cover several ids. Default is every document in the chunks dir."
        ),
    )
    return parser.parse_args(argv)


def _rebuild_bm25(chunks_dir: Path, index_path: Path) -> int:
    chunks = load_chunk_directory(chunks_dir)
    if not chunks:
        _LOGGER.warning("No chunks found in %s. BM25 index not written.", chunks_dir)
        return 0
    index = bm25_module.build_index(chunks)
    bm25_module.save_index(index, index_path)
    _LOGGER.info(
        "BM25 index rewritten: %d chunks across %d documents at %s",
        len(index.chunks),
        len({chunk.document_id for chunk in index.chunks}),
        index_path,
    )
    return len(index.chunks)


def _rebuild_dense(chunks_dir: Path, index_path: Path) -> int:
    # Lazy import so a workstation without the retrieval extra can still use
    # the BM25-only path above.
    from jsc_review.retrieval.embedding import (
        SentenceTransformerEncoder,
        embedding_settings,
    )

    chunks = load_chunk_directory(chunks_dir)
    if not chunks:
        _LOGGER.warning("No chunks found in %s. Dense index not written.", chunks_dir)
        return 0
    encoder = SentenceTransformerEncoder(embedding_settings())
    index = dense_module.build_index(chunks, encoder, batch_size=8)
    dense_module.save_index(index, index_path)
    _LOGGER.info(
        "Dense index rewritten: %d chunks across %d documents at %s",
        len(index.chunks),
        len({chunk.document_id for chunk in index.chunks}),
        index_path,
    )
    return len(index.chunks)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a shell-style exit code."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if not args.bm25 and not args.dense:
        _LOGGER.error("At least one of --bm25 or --dense must be given.")
        return 2

    settings = get_settings()
    chunks_dir: Path = args.chunks_dir or settings.chunks_with_ocr_dir
    if not chunks_dir.exists():
        _LOGGER.error("Chunks directory %s does not exist.", chunks_dir)
        return 1

    bm25_path: Path = args.bm25_path or settings.bm25_index_path
    dense_path: Path = args.dense_path or settings.dense_index_path
    only = args.only or None

    # --force: wipe before rebuild. Deliberate; the user explicitly asked.
    if args.force and args.bm25 and bm25_path.exists():
        bm25_path.unlink()
    if args.force and args.dense and dense_path.exists():
        dense_path.unlink()
        meta_path = dense_path.with_suffix(dense_path.suffix + ".meta.json")
        if meta_path.exists():
            meta_path.unlink()

    # Full rebuild path: when there is no existing index, build from scratch
    # (apply_backfill expects an existing index to extend).
    if args.bm25 and not bm25_path.exists():
        _rebuild_bm25(chunks_dir, bm25_path)
    if args.dense and not dense_path.exists():
        _rebuild_dense(chunks_dir, dense_path)

    # Incremental path: both indexes already exist on disk and we only need
    # to append the documents they are missing.
    needs_backfill = (args.bm25 and bm25_path.exists()) or (
        args.dense and dense_path.exists()
    )
    if needs_backfill:
        encoder = None
        if args.dense and dense_path.exists():
            from jsc_review.retrieval.embedding import (
                SentenceTransformerEncoder,
                embedding_settings,
            )

            encoder = SentenceTransformerEncoder(embedding_settings())
        report = apply_backfill(
            chunks_dir=chunks_dir,
            bm25_index_path=bm25_path,
            dense_index_path=dense_path if args.dense else None,
            encoder=encoder,
            only_documents=only,
        )
        _LOGGER.info(
            "Backfill: BM25 +%d chunks (%d docs), dense +%d chunks (%d docs)",
            report.bm25_added_chunks,
            len(report.bm25_added_documents),
            report.dense_added_chunks,
            len(report.dense_added_documents),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
