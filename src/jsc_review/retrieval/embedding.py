"""Thin wrapper over a local sentence-transformers multilingual model.

The weights live in the local Hugging Face cache on this machine. The wrapper
runs on CPU and does not call a remote service at encode time.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EmbeddingSettings:
    """Local embedding configuration."""

    model_name: str
    cache_dir: Path | None
    device: str
    normalize: bool


class SentenceTransformerEncoder:
    """Encode text with a sentence-transformers model loaded from local cache."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"device": settings.device}
        if settings.cache_dir is not None:
            kwargs["cache_folder"] = str(settings.cache_dir)
        self._model: Any = SentenceTransformer(settings.model_name, **kwargs)
        self._normalize = settings.normalize

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


@lru_cache
def embedding_settings() -> EmbeddingSettings:
    """Read the versioned embedding configuration."""

    config_path = Path(__file__).with_name("dense_config.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_cache = payload.get("cache_dir")
    cache_dir = Path(raw_cache) if raw_cache else None
    return EmbeddingSettings(
        model_name=str(payload["model_name"]),
        cache_dir=cache_dir,
        device=str(payload.get("device", "cpu")),
        normalize=bool(payload.get("normalize", False)),
    )
