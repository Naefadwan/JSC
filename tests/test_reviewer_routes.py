"""Tests for the local reviewer HTTP routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jsc_review.api.app import create_app
from jsc_review.core.settings import Settings, get_settings


@pytest.fixture
def app_with_tmp_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    findings_path = tmp_path / "findings.json"
    decisions_path = tmp_path / "decisions.json"
    findings = [
        {
            "document_id": "doc1",
            "result": {
                "requirement_id": "GOV-001",
                "status": "Met",
                "reason": "Verified quote on the embedded page.",
                "evidence": {
                    "text": "سياسة الشركة في إدارة المخاطر",
                    "source_document": "doc1",
                    "page": 38,
                    "section": "Risk policy",
                    "chunk_id": "doc1:p38:c02",
                    "verified": True,
                },
                "confidence": 0.9,
                "human_review": "Required",
                "evaluation_method": "llm",
            },
        },
        {
            "document_id": "doc1",
            "result": {
                "requirement_id": "DIS-010",
                "status": "Needs Review",
                "reason": "No major shareholder paragraph was found.",
                "evidence": None,
                "confidence": 0.3,
                "human_review": "Required",
                "evaluation_method": "deterministic",
            },
        },
    ]
    findings_path.write_text(json.dumps(findings, ensure_ascii=False), encoding="utf-8")

    def _settings_override() -> Settings:
        return Settings(
            findings_path=findings_path,
            reviewer_decisions_path=decisions_path,
        )

    get_settings.cache_clear()
    monkeypatch.setattr("jsc_review.api.routes.reviewer.get_settings", _settings_override)
    app = create_app()
    return TestClient(app)


def test_list_findings_returns_both(app_with_tmp_paths: TestClient) -> None:
    response = app_with_tmp_paths.get("/reviewer/findings")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["findings"]) == 2
    assert payload["findings"][0]["result"]["requirement_id"] == "GOV-001"
    assert payload["findings"][0]["decision"] is None


def test_submit_decision_persists_and_appears(app_with_tmp_paths: TestClient) -> None:
    submission = {
        "document_id": "doc1",
        "requirement_id": "GOV-001",
        "action": "accepted",
        "final_status": "Met",
        "reviewer_name": "A. Reviewer",
        "reviewer_notes": "",
    }
    response = app_with_tmp_paths.post("/reviewer/decisions", json=submission)
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "accepted"
    assert body["final_status"] == "Met"

    listing = app_with_tmp_paths.get("/reviewer/findings").json()
    first = listing["findings"][0]
    assert first["decision"] is not None
    assert first["decision"]["reviewer_name"] == "A. Reviewer"


def test_change_without_notes_is_rejected(app_with_tmp_paths: TestClient) -> None:
    response = app_with_tmp_paths.post(
        "/reviewer/decisions",
        json={
            "document_id": "doc1",
            "requirement_id": "GOV-001",
            "action": "changed",
            "final_status": "Not Met",
            "reviewer_name": "A. Reviewer",
            "reviewer_notes": "",
        },
    )
    assert response.status_code == 422


def test_defer_must_resolve_to_needs_review(app_with_tmp_paths: TestClient) -> None:
    response = app_with_tmp_paths.post(
        "/reviewer/decisions",
        json={
            "document_id": "doc1",
            "requirement_id": "GOV-001",
            "action": "deferred",
            "final_status": "Met",
            "reviewer_name": "A. Reviewer",
            "reviewer_notes": "",
        },
    )
    assert response.status_code == 422


def test_second_decision_replaces_first(app_with_tmp_paths: TestClient) -> None:
    first = {
        "document_id": "doc1",
        "requirement_id": "GOV-001",
        "action": "accepted",
        "final_status": "Met",
        "reviewer_name": "A. Reviewer",
        "reviewer_notes": "",
    }
    second = {
        **first,
        "action": "changed",
        "final_status": "Not Met",
        "reviewer_notes": "second look",
    }
    app_with_tmp_paths.post("/reviewer/decisions", json=first)
    app_with_tmp_paths.post("/reviewer/decisions", json=second)

    listing = app_with_tmp_paths.get("/reviewer/findings").json()
    first_finding = listing["findings"][0]
    assert first_finding["decision"]["action"] == "changed"
    assert first_finding["decision"]["final_status"] == "Not Met"


def test_page_serves_html(app_with_tmp_paths: TestClient) -> None:
    response = app_with_tmp_paths.get("/reviewer/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "JSC Reviewer" in response.text
