"""Settings defaults — reranker opt-in and gate path."""

from __future__ import annotations

from pathlib import Path

from jsc_review.core.settings import Settings


def test_reranker_is_off_by_default() -> None:
    """``use_reranker`` defaults to ``False`` so uploads skip the heavy model."""

    settings = Settings()
    assert settings.use_reranker is False


def test_reranker_toggle_can_be_opted_in() -> None:
    """A caller can turn the reranker on by overriding the field."""

    settings = Settings(use_reranker=True)
    assert settings.use_reranker is True


def test_retrieval_gate_measurements_path_default() -> None:
    """The brief and nightly check both read from the same default gate file."""

    settings = Settings()
    assert settings.retrieval_gate_measurements_path == Path(
        "Data/reports/retrieval_gate/measurements.json"
    )
