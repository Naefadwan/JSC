"""Tests for the ``python -m jsc_review.retrieval.build_indexes`` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval.bm25 import IndexedChunk, save_indexed_chunks
from jsc_review.retrieval.build_indexes import main


def _chunks(document_id: str, n: int = 3) -> list[IndexedChunk]:
    return [
        IndexedChunk(
            chunk_id=f"{document_id}-{i}",
            document_id=document_id,
            page=i + 1,
            section=f"section-{i}",
            text=f"sample passage {i} for {document_id}",
            extraction_method="embedded",
        )
        for i in range(n)
    ]


def test_cli_requires_at_least_one_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--chunks-dir",
            str(tmp_path / "chunks"),
            "--bm25-path",
            str(tmp_path / "bm25.json"),
        ]
    )
    assert exit_code == 2


def test_cli_bm25_builds_fresh_index(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    save_indexed_chunks(_chunks("doc1"), chunks_dir, "doc1")
    save_indexed_chunks(_chunks("doc2"), chunks_dir, "doc2")
    bm25_path = tmp_path / "bm25.json"
    exit_code = main(
        [
            "--bm25",
            "--chunks-dir",
            str(chunks_dir),
            "--bm25-path",
            str(bm25_path),
        ]
    )
    assert exit_code == 0
    assert bm25_path.exists()
    rebuilt = bm25_module.load_index(bm25_path)
    assert len({chunk.document_id for chunk in rebuilt.chunks}) == 2


def test_cli_bm25_incremental_adds_only_missing_documents(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    save_indexed_chunks(_chunks("doc1"), chunks_dir, "doc1")
    bm25_path = tmp_path / "bm25.json"

    # First pass: build with doc1 only.
    main(["--bm25", "--chunks-dir", str(chunks_dir), "--bm25-path", str(bm25_path)])

    # Add doc2 to the chunks dir and run again.
    save_indexed_chunks(_chunks("doc2"), chunks_dir, "doc2")
    exit_code = main(
        [
            "--bm25",
            "--chunks-dir",
            str(chunks_dir),
            "--bm25-path",
            str(bm25_path),
        ]
    )
    assert exit_code == 0
    rebuilt = bm25_module.load_index(bm25_path)
    assert {chunk.document_id for chunk in rebuilt.chunks} == {"doc1", "doc2"}


def test_cli_force_wipes_and_rebuilds(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    save_indexed_chunks(_chunks("doc1"), chunks_dir, "doc1")
    bm25_path = tmp_path / "bm25.json"
    bm25_path.write_text("{broken}", encoding="utf-8")  # unreadable sentinel
    exit_code = main(
        [
            "--bm25",
            "--force",
            "--chunks-dir",
            str(chunks_dir),
            "--bm25-path",
            str(bm25_path),
        ]
    )
    assert exit_code == 0
    rebuilt = bm25_module.load_index(bm25_path)
    assert {chunk.document_id for chunk in rebuilt.chunks} == {"doc1"}


def test_cli_errors_when_chunks_dir_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--bm25",
            "--chunks-dir",
            str(tmp_path / "does-not-exist"),
            "--bm25-path",
            str(tmp_path / "bm25.json"),
        ]
    )
    assert exit_code == 1
