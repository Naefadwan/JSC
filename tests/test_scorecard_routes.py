"""Route tests for the Scorecard drawer endpoint and the evidence-pin route."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jsc_review.api.app import create_app
from jsc_review.core.settings import Settings, get_settings


def _finding(
    doc_id: str, req_id: str, status: str, page: int = 1
) -> dict[str, object]:
    evidence: dict[str, object] | None = None
    if status in {"Met", "Not Met"}:
        evidence = {
            "text": "verified passage",
            "source_document": doc_id,
            "page": page,
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
    findings_path = tmp_path / "findings.json"
    golden_path = tmp_path / "golden.json"
    findings_path.write_text(
        json.dumps(
            [
                _finding("d1", "R1", "Met"),
                _finding("d1", "R2", "Not Met"),
                _finding("d1", "R3", "Needs Review"),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def _settings_override() -> Settings:
        return Settings(
            findings_path=findings_path,
            golden_labels_path=golden_path,
            bm25_index_path=tmp_path / "bm25-missing.json",
            dense_index_path=tmp_path / "dense-missing.faiss",
        )

    get_settings.cache_clear()
    monkeypatch.setattr(
        "jsc_review.api.routes.scorecard.get_settings", _settings_override
    )
    monkeypatch.setattr(
        "jsc_review.api.routes.golden.get_settings", _settings_override
    )
    app = create_app()
    return TestClient(app)


def test_scorecard_endpoint_renders_with_empty_golden(client: TestClient) -> None:
    response = client.get("/reviewer/scorecard")
    assert response.status_code == 200
    body = response.json()
    assert body["total_golden"] == 0
    # Even with no golden labels, referral rate uses all findings.
    rr = next(m for m in body["metrics"] if m["name"] == "referral_rate")
    assert rr["value"] == 1.0  # all three are human_review=Required


def test_scorecard_computes_status_accuracy_after_double_review(
    client: TestClient,
) -> None:
    # Sample and label two candidates with agreement.
    client.post("/reviewer/golden/sample", json={"count": 3, "seed": 0})
    listing = client.get("/reviewer/golden").json()
    for cand in listing["candidates"][:2]:
        for reviewer in ("A", "B"):
            client.post(
                "/reviewer/golden/label",
                json={
                    "document_id": cand["document_id"],
                    "requirement_id": cand["requirement_id"],
                    "reviewer_name": reviewer,
                    "label": cand["system_status"],
                },
            )
    response = client.get("/reviewer/scorecard")
    body = response.json()
    assert body["scored_golden"] == 2
    status = next(m for m in body["metrics"] if m["name"] == "status_accuracy")
    assert status["sample_size"] == 2
    assert status["value"] == 1.0


def test_scorecard_hit5_degrades_when_bm25_missing(client: TestClient) -> None:
    response = client.get("/reviewer/scorecard?include_hit_at_5=true")
    assert response.status_code == 200
    body = response.json()
    hit = next(m for m in body["metrics"] if m["name"] == "hit_at_5")
    assert hit["value"] is None
    assert "BM25" in hit["notes"] or "golden" in hit["notes"] or "page" in hit["notes"]


def test_pin_evidence_endpoint_sets_ground_truth(client: TestClient) -> None:
    client.post("/reviewer/golden/sample", json={"count": 1, "seed": 0})
    listing = client.get("/reviewer/golden").json()
    cand = listing["candidates"][0]
    response = client.post(
        "/reviewer/golden/evidence",
        json={
            "document_id": cand["document_id"],
            "requirement_id": cand["requirement_id"],
            "page": 11,
            "text": "ground-truth quote",
            "pinned_by": "Nayef",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ground_truth_page"] == 11
    assert body["ground_truth_pinned_by"] == "Nayef"


def test_pin_evidence_rejects_unknown_candidate(client: TestClient) -> None:
    response = client.post(
        "/reviewer/golden/evidence",
        json={
            "document_id": "nope",
            "requirement_id": "nope",
            "page": 1,
            "text": "",
            "pinned_by": "Nayef",
        },
    )
    assert response.status_code == 404


def test_pin_evidence_rejects_bad_page(client: TestClient) -> None:
    response = client.post(
        "/reviewer/golden/evidence",
        json={
            "document_id": "d1",
            "requirement_id": "R1",
            "page": 0,  # must be >= 1
            "text": "",
            "pinned_by": "A",
        },
    )
    assert response.status_code == 422
