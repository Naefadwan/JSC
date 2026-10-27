"""Route tests for the Golden-labels drawer endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jsc_review.api.app import create_app
from jsc_review.core.settings import Settings, get_settings


def _finding(doc_id: str, req_id: str, status: str) -> dict[str, object]:
    evidence: dict[str, object] | None = None
    if status in {"Met", "Not Met"}:
        evidence = {
            "text": "verified passage",
            "source_document": doc_id,
            "page": 1,
            "section": "fixture",
            "chunk_id": f"{doc_id}-{req_id}",
            "verified": True,
        }
    return {
        "document_id": doc_id,
        "result": {
            "requirement_id": req_id,
            "status": status,
            "reason": "fixture",
            "evidence": evidence,
            "confidence": 0.5,
            "human_review": "Required",
            "evaluation_method": "llm",
        },
    }


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient whose golden + findings paths live under tmp_path."""

    findings_path = tmp_path / "findings.json"
    golden_path = tmp_path / "golden.json"
    findings_path.write_text(
        json.dumps(
            [
                _finding("doc1", "R1", "Met"),
                _finding("doc1", "R2", "Not Met"),
                _finding("doc1", "R3", "Needs Review"),
                _finding("doc2", "R1", "Needs Review"),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def _settings_override() -> Settings:
        return Settings(
            findings_path=findings_path,
            golden_labels_path=golden_path,
        )

    get_settings.cache_clear()
    monkeypatch.setattr(
        "jsc_review.api.routes.golden.get_settings", _settings_override
    )
    app = create_app()
    return TestClient(app)


def test_list_golden_returns_empty_set_initially(client: TestClient) -> None:
    response = client.get("/reviewer/golden")
    assert response.status_code == 200
    body = response.json()
    assert body["candidates"] == []
    assert body["summary"]["total"] == 0


def test_sample_adds_candidates_and_is_idempotent(client: TestClient) -> None:
    first = client.post("/reviewer/golden/sample", json={"count": 10, "seed": 0})
    assert first.status_code == 201
    body = first.json()
    assert body["added"] > 0

    # Second sample should see every candidate as a duplicate.
    second = client.post("/reviewer/golden/sample", json={"count": 10, "seed": 0})
    assert second.status_code == 201
    assert second.json()["added"] == 0
    assert second.json()["skipped_duplicates"] >= 0


def test_label_full_roundtrip_and_agreement_counters(client: TestClient) -> None:
    client.post("/reviewer/golden/sample", json={"count": 3, "seed": 0})
    listing = client.get("/reviewer/golden").json()
    cand = listing["candidates"][0]

    r1 = client.post(
        "/reviewer/golden/label",
        json={
            "document_id": cand["document_id"],
            "requirement_id": cand["requirement_id"],
            "reviewer_name": "A",
            "label": "Met",
            "reason": "a-reason",
        },
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/reviewer/golden/label",
        json={
            "document_id": cand["document_id"],
            "requirement_id": cand["requirement_id"],
            "reviewer_name": "B",
            "label": "Not Met",
            "reason": "b-reason",
        },
    )
    assert r2.status_code == 200
    final = client.get("/reviewer/golden").json()
    assert final["summary"]["disagree"] == 1
    assert final["summary"]["agree"] == 0
    assert final["summary"]["agreement_rate"] == 0.0


def test_label_rejects_unknown_candidate(client: TestClient) -> None:
    response = client.post(
        "/reviewer/golden/label",
        json={
            "document_id": "doesnt-exist",
            "requirement_id": "R999",
            "reviewer_name": "A",
            "label": "Met",
        },
    )
    assert response.status_code == 404


def test_label_rejects_bad_status(client: TestClient) -> None:
    client.post("/reviewer/golden/sample", json={"count": 1, "seed": 0})
    listing = client.get("/reviewer/golden").json()
    cand = listing["candidates"][0]
    response = client.post(
        "/reviewer/golden/label",
        json={
            "document_id": cand["document_id"],
            "requirement_id": cand["requirement_id"],
            "reviewer_name": "A",
            "label": "Partially Met",  # Rejected throughout the project.
        },
    )
    assert response.status_code == 422


def test_sample_rejects_extra_fields(client: TestClient) -> None:
    response = client.post(
        "/reviewer/golden/sample", json={"count": 5, "seed": 0, "note": "nope"}
    )
    assert response.status_code == 422


def test_delete_candidate_204_and_404(client: TestClient) -> None:
    client.post("/reviewer/golden/sample", json={"count": 1, "seed": 0})
    listing = client.get("/reviewer/golden").json()
    cand = listing["candidates"][0]
    deleted = client.delete(
        f"/reviewer/golden/{cand['document_id']}/{cand['requirement_id']}"
    )
    assert deleted.status_code == 204
    missing = client.delete(
        f"/reviewer/golden/{cand['document_id']}/{cand['requirement_id']}"
    )
    assert missing.status_code == 404
