"""Cross-encoder reranking of candidate passages.

A reranker re-scores the top candidates from a first-stage retriever (BM25,
dense, or hybrid) with a model that reads the query and the passage together.
This catches semantic mismatches that bi-encoders cannot see.

The wrapper runs on CPU and loads weights from the local Hugging Face cache.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from jsc_review.retrieval.bm25 import SearchHit


class CrossEncoder(Protocol):
    """A cross-encoder that scores (query, passage) pairs."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class RerankerSettings:
    """Local reranker configuration."""

    model_name: str
    cache_dir: Path | None
    device: str
    max_length: int


def rerank(
    query: str,
    hits: Sequence[SearchHit],
    cross_encoder: CrossEncoder,
    *,
    top_k: int = 10,
) -> tuple[SearchHit, ...]:
    """Re-score the hits with a cross-encoder and return the new top-k."""

    if top_k < 1 or not hits or not query.strip():
        return ()
    pairs = [(query, hit.text) for hit in hits]
    scores = cross_encoder.predict(pairs)
    scored = sorted(
        ((float(score), hit) for score, hit in zip(scores, hits, strict=False)),
        key=lambda item: (-item[0], item[1].chunk_id),
    )
    return tuple(replace(hit, score=score) for score, hit in scored[:top_k])


class SentenceTransformerCrossEncoder:
    """Wraps sentence-transformers.CrossEncoder loaded from the local cache."""

    def __init__(self, settings: RerankerSettings) -> None:
        from sentence_transformers import CrossEncoder as _ST_CrossEncoder

        kwargs: dict[str, Any] = {"device": settings.device, "max_length": settings.max_length}
        if settings.cache_dir is not None:
            kwargs["cache_folder"] = str(settings.cache_dir)
        self._model: Any = _ST_CrossEncoder(settings.model_name, **kwargs)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        scores = self._model.predict(
            list(pairs),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32)


@lru_cache
def reranker_settings() -> RerankerSettings:
    """Read the versioned reranker configuration."""

    config_path = Path(__file__).with_name("rerank_config.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_cache = payload.get("cache_dir")
    cache_dir = Path(raw_cache) if raw_cache else None
    return RerankerSettings(
        model_name=str(payload["model_name"]),
        cache_dir=cache_dir,
        device=str(payload.get("device", "cpu")),
        max_length=int(payload.get("max_length", 512)),
    )
