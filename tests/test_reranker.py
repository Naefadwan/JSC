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

    monkeypatch.setattr(reranker, "reranker_settings", reranker.reranker_settings.__wrapped__)
    monkeypatch.setattr(reranker, "__file__", str(tmp_path / "reranker.py"))
    with pytest.raises(KeyError):
        reranker.reranker_settings()


def test_shared_cross_encoder_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared loader builds one cross-encoder and reuses it.

    The reranker is opt-in on the production path. When a reviewer flips the
    setting on, every upload grabs the same encoder instance so weights are
    not reloaded per requirement. This test uses a lightweight stand-in to
    avoid loading sentence-transformers in CI.
    """

    from jsc_review.retrieval import reranker

    build_count = {"n": 0}

    class _Stub:
        def predict(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
            del pairs
            return np.asarray([], dtype=np.float32)

    def fake_builder(_settings: object) -> _Stub:
        build_count["n"] += 1
        return _Stub()

    reranker.shared_cross_encoder.cache_clear()
    monkeypatch.setattr(reranker, "SentenceTransformerCrossEncoder", fake_builder)

    try:
        first = reranker.shared_cross_encoder()
        second = reranker.shared_cross_encoder()
    finally:
        reranker.shared_cross_encoder.cache_clear()

    assert first is second
    assert build_count["n"] == 1
