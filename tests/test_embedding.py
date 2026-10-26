"""Tests for the versioned embedding configuration.

The real model download runs offline, so this file only covers the
configuration contract. The index builder script exercises the full encoder
path on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jsc_review.retrieval.embedding import embedding_settings


def test_embedding_settings_names_the_local_model() -> None:
    settings = embedding_settings()
    assert settings.model_name
    assert settings.device == "cpu"


def test_embedding_config_is_versioned_on_disk() -> None:
    from jsc_review import retrieval

    config_path = Path(retrieval.__file__).with_name("dense_config.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert "model_name" in payload
    assert "device" in payload
    assert payload["device"] == "cpu"


def test_sentence_transformer_import_is_lazy() -> None:
    """The module exposes the encoder class without touching heavy packages."""

    import importlib
    import sys

    sys.modules.pop("jsc_review.retrieval.embedding", None)
    module = importlib.import_module("jsc_review.retrieval.embedding")
    assert hasattr(module, "SentenceTransformerEncoder")


def test_embedding_settings_rejects_missing_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt config raises a KeyError, not a silent default."""

    from jsc_review.retrieval import embedding

    config = tmp_path / "dense_config.json"
    config.write_text(json.dumps({"device": "cpu"}), encoding="utf-8")

    monkeypatch.setattr(embedding, "embedding_settings", embedding.embedding_settings.__wrapped__)
    monkeypatch.setattr(embedding, "__file__", str(tmp_path / "embedding.py"))
    with pytest.raises(KeyError):
        embedding.embedding_settings()
