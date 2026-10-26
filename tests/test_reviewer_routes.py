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
    register_path = tmp_path / "requirements_register.csv"
    register_path.write_text(
        "requirement_id,checklist,section,requirement_text_ar,requirement_text_en,"
        "requirement_type,evaluation_method,poc_selected,owner,notes\n"
        "GOV-001,governance,s1,نص,text,presence,llm,yes,,\n"
        "GOV-002,governance,s1,نص2,text 2,completeness,llm,,,\n"
        "DIS-001,disclosure,s2,قائمة,statement,presence,deterministic,yes,,\n",
        encoding="utf-8",
    )
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
            requirements_register_path=register_path,
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


# ----------------------------------------- requirements register endpoints ---


def test_list_requirements_returns_rows_and_counts(app_with_tmp_paths: TestClient) -> None:
    """GET /reviewer/requirements lists every row with the current selection state."""

    response = app_with_tmp_paths.get("/reviewer/requirements")
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 3
    assert body["selected_count"] == 2
    ids = [row["requirement_id"] for row in body["rows"]]
    assert ids == ["GOV-001", "GOV-002", "DIS-001"]
    by_id = {row["requirement_id"]: row for row in body["rows"]}
    assert by_id["GOV-001"]["poc_selected"] is True
    assert by_id["GOV-002"]["poc_selected"] is False


def test_patch_requirements_flips_selection_and_persists(
    app_with_tmp_paths: TestClient,
) -> None:
    """PATCH /reviewer/requirements updates poc_selected on disk."""

    response = app_with_tmp_paths.patch(
        "/reviewer/requirements",
        json={"selections": {"GOV-002": True, "DIS-001": False}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["selected_count"] == 2
    assert body["unknown_requirement_ids"] == []
    by_id = {row["requirement_id"]: row for row in body["rows"]}
    assert by_id["GOV-001"]["poc_selected"] is True
    assert by_id["GOV-002"]["poc_selected"] is True
    assert by_id["DIS-001"]["poc_selected"] is False

    # Follow-up GET sees the saved state.
    fresh = app_with_tmp_paths.get("/reviewer/requirements").json()
    by_id_fresh = {row["requirement_id"]: row for row in fresh["rows"]}
    assert by_id_fresh["GOV-002"]["poc_selected"] is True
    assert by_id_fresh["DIS-001"]["poc_selected"] is False


def test_patch_requirements_reports_unknown_ids(app_with_tmp_paths: TestClient) -> None:
    """Unknown ids come back in ``unknown_requirement_ids`` without raising."""

    response = app_with_tmp_paths.patch(
        "/reviewer/requirements",
        json={"selections": {"NEW-999": True, "GOV-001": False}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["unknown_requirement_ids"] == ["NEW-999"]
    by_id = {row["requirement_id"]: row for row in body["rows"]}
    assert by_id["GOV-001"]["poc_selected"] is False  # the known one still landed


def test_patch_requirements_rejects_wrong_payload_shape(
    app_with_tmp_paths: TestClient,
) -> None:
    """Extra fields fail validation so the client cannot touch other columns."""

    response = app_with_tmp_paths.patch(
        "/reviewer/requirements",
        json={"selections": {"GOV-001": True}, "evil": {"requirement_text_ar": "pwned"}},
    )
    assert response.status_code == 422
