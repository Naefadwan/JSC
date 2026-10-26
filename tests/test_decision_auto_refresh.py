"""Integration tests for the brief auto-refresh hook on the decision endpoint.

The reviewer endpoint always records the decision first and then refreshes the
brief. The brief refresh is a best-effort side effect: it must update the
file when enabled, must leave the file alone when the toggle is off, and must
never fail the endpoint even when the brief file is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jsc_review.api.app import create_app
from jsc_review.core.settings import Settings, get_settings

_BRIEF_TEMPLATE = (
    "# Owner brief\n\n"
    "Prose that must survive.\n\n"
    "<!-- auto:begin:counts -->\n"
    "- `Met`: 999 (deterministic checks: 0, local-model: 0).\n"
    "- `Not Met`: 0.\n"
    "- `Needs Review`: 0.\n"
    "- Reviewer decisions recorded: 0 (accepted: 0, changed: 0, deferred: 0).\n"
    "<!-- auto:end:counts -->\n"
)


def _write_findings(path: Path) -> None:
    findings = [
        {
            "document_id": "doc1",
            "result": {
                "requirement_id": "GOV-001",
                "status": "Needs Review",
                "reason": "No supporting passage.",
                "evidence": None,
                "confidence": 0.4,
                "human_review": "Required",
                "evaluation_method": "llm",
            },
        }
    ]
    path.write_text(json.dumps(findings, ensure_ascii=False), encoding="utf-8")


def _build_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    auto_refresh: bool,
    brief_exists: bool,
) -> tuple[TestClient, Path]:
    findings_path = tmp_path / "findings.json"
    decisions_path = tmp_path / "decisions.json"
    brief_path = tmp_path / "brief.md"
    _write_findings(findings_path)
    if brief_exists:
        brief_path.write_text(_BRIEF_TEMPLATE, encoding="utf-8")

    def _settings_override() -> Settings:
        return Settings(
            findings_path=findings_path,
            reviewer_decisions_path=decisions_path,
            brief_path=brief_path,
            auto_refresh_brief=auto_refresh,
        )

    get_settings.cache_clear()
    monkeypatch.setattr("jsc_review.api.routes.reviewer.get_settings", _settings_override)
    client = TestClient(create_app())
    return client, brief_path


def _accept_payload() -> dict[str, str]:
    return {
        "document_id": "doc1",
        "requirement_id": "GOV-001",
        "action": "accepted",
        "final_status": "Needs Review",
        "reviewer_name": "A. Reviewer",
        "reviewer_notes": "",
    }


def test_decision_refreshes_brief_when_toggle_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, brief_path = _build_client(tmp_path, monkeypatch, auto_refresh=True, brief_exists=True)

    response = client.post("/reviewer/decisions", json=_accept_payload())

    assert response.status_code == 200
    body = brief_path.read_text(encoding="utf-8")
    assert "Prose that must survive." in body
    # The brief now reflects the single Needs Review finding and one accepted decision.
    assert "`Met`: 0" in body
    assert "`Needs Review`: 1" in body
    assert "accepted: 1" in body
    assert "999" not in body  # stale sentinel body replaced


def test_decision_does_not_touch_brief_when_toggle_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, brief_path = _build_client(tmp_path, monkeypatch, auto_refresh=False, brief_exists=True)
    before = brief_path.read_text(encoding="utf-8")

    response = client.post("/reviewer/decisions", json=_accept_payload())

    assert response.status_code == 200
    assert brief_path.read_text(encoding="utf-8") == before


def test_decision_succeeds_even_when_brief_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing brief must never fail the endpoint; the decision is still saved."""

    client, brief_path = _build_client(tmp_path, monkeypatch, auto_refresh=True, brief_exists=False)

    response = client.post("/reviewer/decisions", json=_accept_payload())

    assert response.status_code == 200
    assert not brief_path.exists()
    # The decision is still recorded on disk.
    decisions_path = tmp_path / "decisions.json"
    saved = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["requirement_id"] == "GOV-001"
