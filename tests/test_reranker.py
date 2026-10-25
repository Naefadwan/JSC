"""Tests for cross-encoder reranking. A fake cross-encoder keeps tests offline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from jsc_review.retrieval.bm25 import SearchHit
from jsc_review.retrieval.reranker import rerank, reranker_settings


class _FakeCrossEncoder:
    """Returns a higher score when the passage text contains the query."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        scores = [1.0 if query.lower() in passage.lower() else 0.1 for query, passage in pairs]
        return np.asarray(scores, dtype=np.float32)


def _hit(chunk_id: str, text: str, score: float = 0.0) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id="doc1",
        page=1,
        section="Body",
        text=text,
        extraction_method="pymupdf-text",
        score=score,
    )


def test_rerank_promotes_the_matching_passage() -> None:
    hits = [
        _hit("a", "the fox jumps over the lazy dog", score=5.0),
        _hit("b", "major shareholders own 28%", score=4.0),
        _hit("c", "financial statements for the year", score=3.0),
    ]
    reranked = rerank("shareholders", hits, _FakeCrossEncoder(), top_k=3)
    assert reranked[0].chunk_id == "b"
    assert reranked[0].score > reranked[1].score


def test_rerank_returns_at_most_top_k() -> None:
    hits = [_hit(cid, "text") for cid in ("a", "b", "c", "d", "e")]
    reranked = rerank("text", hits, _FakeCrossEncoder(), top_k=2)
    assert len(reranked) == 2


def test_rerank_handles_an_empty_input() -> None:
    assert rerank("q", (), _FakeCrossEncoder(), top_k=5) == ()
    assert rerank("", [_hit("a", "t")], _FakeCrossEncoder(), top_k=5) == ()


def test_reranker_settings_names_the_local_model() -> None:
    settings = reranker_settings()
    assert settings.model_name
    assert settings.device == "cpu"
    assert settings.max_length >= 256


def test_rerank_config_is_versioned_on_disk() -> None:
    from jsc_review import retrieval

    config_path = Path(retrieval.__file__).with_name("rerank_config.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert "model_name" in payload
    assert payload["device"] == "cpu"


def test_reranker_settings_rejects_missing_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt config raises a KeyError, not a silent default."""

    from jsc_review.retrieval import reranker

    config = tmp_path / "rerank_config.json"
    config.write_text(json.dumps({"device": "cpu"}), encoding="utf-8")

    monkeypatch.setattr(
        reranker, "reranker_settings", reranker.reranker_settings.__wrapped__
    )
    monkeypatch.setattr(reranker, "__file__", str(tmp_path / "reranker.py"))
    with pytest.raises(KeyError):
        reranker.reranker_settings()
