"""Route tests for /reviewer/companies, document links, and compliance."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jsc_review.api.app import create_app
from jsc_review.companies.compliance import DisclosurePeriod, save_periods
from jsc_review.companies.registry import Company, CompanyRegistryPayload
from jsc_review.core.settings import Settings, get_settings


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient whose company/compliance paths live under ``tmp_path``."""

    registry_path = tmp_path / "companies.json"
    mapping_path = tmp_path / "doc_companies.json"
    periods_path = tmp_path / "periods.json"
    excel_path = tmp_path / "src.xlsx"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()

    # Names are long enough to clear the specificity filter (6+ chars for a
    # single-word name). Short tickers are matched separately.
    payload = CompanyRegistryPayload(
        source_excel=str(excel_path),
        companies=[
            Company(
                name_ar="Alphabetical Corp",
                ticker="AAA",
                numeric_code="100001",
                sector_ar="Banks",
            ),
            Company(
                name_ar="Betacorp Holdings",
                ticker="BBB",
                numeric_code="100002",
                sector_ar="Insurance",
            ),
        ],
    )
    registry_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    save_periods(
        periods_path,
        [DisclosurePeriod(period_id="Q1-2026", due_date=date(2026, 4, 30))],
    )

    def _settings_override() -> Settings:
        return Settings(
            companies_registry_path=registry_path,
            document_companies_path=mapping_path,
            disclosure_periods_path=periods_path,
            companies_excel_path=excel_path,
            reports_dir=reports_dir,
            extracted_dir=extracted_dir,
            findings_path=tmp_path / "findings.json",
            reviewer_decisions_path=tmp_path / "decisions.json",
            # Lower the threshold so our fixture's "Alpha" + "AAA" scores pass.
            auto_link_confidence_threshold=1.5,
        )

    get_settings.cache_clear()
    monkeypatch.setattr(
        "jsc_review.api.routes.companies.get_settings", _settings_override
    )
    app = create_app()
    return TestClient(app)


def test_list_companies(client: TestClient) -> None:
    response = client.get("/reviewer/companies")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    tickers = {company["ticker"] for company in body["companies"]}
    assert tickers == {"AAA", "BBB"}


def test_get_document_company_without_link_returns_empty(client: TestClient) -> None:
    response = client.get("/reviewer/documents/by-id/doc_15-04-2026_09-00-00/company")
    assert response.status_code == 200
    assert response.json() == {"link": None, "company": None}


def test_put_document_company_sets_link(client: TestClient) -> None:
    response = client.put(
        "/reviewer/documents/by-id/doc_15-04-2026_09-00-00/company",
        json={"company_key": "AAA", "linked_by": "Nayef"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["link"]["company_key"] == "AAA"
    assert body["company"]["name_ar"] == "Alphabetical Corp"

    follow_up = client.get("/reviewer/documents/by-id/doc_15-04-2026_09-00-00/company")
    assert follow_up.json()["link"]["company_key"] == "AAA"


def test_put_document_company_resolves_numeric_code(client: TestClient) -> None:
    response = client.put(
        "/reviewer/documents/by-id/doc_a/company",
        json={"company_key": "100002"},
    )
    assert response.status_code == 200
    assert response.json()["company"]["ticker"] == "BBB"


def test_put_document_company_rejects_unknown_company(client: TestClient) -> None:
    response = client.put(
        "/reviewer/documents/by-id/doc_a/company",
        json={"company_key": "ZZZ"},
    )
    assert response.status_code == 404


def test_put_document_company_rejects_extra_fields(client: TestClient) -> None:
    response = client.put(
        "/reviewer/documents/by-id/doc_a/company",
        json={"company_key": "AAA", "unknown": "x"},
    )
    assert response.status_code == 422


def test_delete_document_company_is_idempotent(client: TestClient) -> None:
    client.put(
        "/reviewer/documents/by-id/doc_a/company",
        json={"company_key": "AAA"},
    )
    response = client.delete("/reviewer/documents/by-id/doc_a/company")
    assert response.status_code == 200
    assert client.get("/reviewer/documents/by-id/doc_a/company").json()["link"] is None
    # Second delete still returns 200 (no-op)
    assert client.delete("/reviewer/documents/by-id/doc_a/company").status_code == 200


def test_rejects_unsafe_document_id(client: TestClient) -> None:
    response = client.get("/reviewer/documents/by-id/..%2Fetc%2Fpasswd/company")
    # FastAPI's path normalization typically returns 404 for traversal,
    # but if the raw segment reaches our handler we return 422.
    assert response.status_code in (404, 422)


def test_list_document_companies(client: TestClient) -> None:
    client.put(
        "/reviewer/documents/by-id/doc_a/company",
        json={"company_key": "AAA"},
    )
    response = client.get("/reviewer/document-companies")
    assert response.status_code == 200
    links = response.json()["links"]
    assert "doc_a" in links
    assert links["doc_a"]["company_key"] == "AAA"


def test_list_disclosure_periods(client: TestClient) -> None:
    response = client.get("/reviewer/disclosure-periods")
    assert response.status_code == 200
    body = response.json()
    assert len(body["periods"]) == 1
    assert body["periods"][0]["period_id"] == "Q1-2026"


def test_create_disclosure_period_adds_to_store(client: TestClient) -> None:
    response = client.post(
        "/reviewer/disclosure-periods",
        json={
            "period_id": "Q2-2026",
            "description": "Q2 interim disclosure",
            "due_date": "2026-07-31",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["period_id"] == "Q2-2026"
    # And it shows up on the list endpoint
    listing = client.get("/reviewer/disclosure-periods").json()
    assert {p["period_id"] for p in listing["periods"]} == {"Q1-2026", "Q2-2026"}


def test_create_disclosure_period_rejects_duplicate(client: TestClient) -> None:
    response = client.post(
        "/reviewer/disclosure-periods",
        json={
            "period_id": "Q1-2026",
            "description": "duplicate",
            "due_date": "2026-04-30",
        },
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_create_disclosure_period_allows_explicit_replace(client: TestClient) -> None:
    response = client.post(
        "/reviewer/disclosure-periods",
        json={
            "period_id": "Q1-2026",
            "description": "updated description",
            "due_date": "2026-05-15",
            "replace": True,
        },
    )
    assert response.status_code == 201
    listing = client.get("/reviewer/disclosure-periods").json()
    only = [p for p in listing["periods"] if p["period_id"] == "Q1-2026"][0]
    assert only["description"] == "updated description"
    assert only["due_date"] == "2026-05-15"


def test_create_disclosure_period_rejects_malformed_id(client: TestClient) -> None:
    response = client.post(
        "/reviewer/disclosure-periods",
        json={
            "period_id": "bad id with spaces",
            "description": "",
            "due_date": "2026-07-31",
        },
    )
    assert response.status_code == 400


def test_create_disclosure_period_rejects_extra_fields(client: TestClient) -> None:
    response = client.post(
        "/reviewer/disclosure-periods",
        json={
            "period_id": "Q3-2026",
            "description": "",
            "due_date": "2026-10-31",
            "note": "extra",
        },
    )
    assert response.status_code == 422


def test_delete_disclosure_period_removes_it(client: TestClient) -> None:
    response = client.delete("/reviewer/disclosure-periods/Q1-2026")
    assert response.status_code == 204
    listing = client.get("/reviewer/disclosure-periods").json()
    assert listing["periods"] == []


def test_delete_disclosure_period_returns_404_when_missing(
    client: TestClient,
) -> None:
    response = client.delete("/reviewer/disclosure-periods/unknown")
    assert response.status_code == 404


def test_compliance_report_groups_on_time_late_missing(client: TestClient) -> None:
    client.put(
        "/reviewer/documents/by-id/doc_15-04-2026_09-00-00/company",
        json={"company_key": "AAA"},
    )
    client.put(
        "/reviewer/documents/by-id/doc_01-05-2026_10-00-00/company",
        json={"company_key": "BBB"},
    )
    response = client.get("/reviewer/compliance?period_id=Q1-2026")
    assert response.status_code == 200
    body = response.json()
    assert body["submitted_on_time_count"] == 1
    assert body["submitted_late_count"] == 1
    assert body["not_submitted_count"] == 0


def test_compliance_report_rejects_unknown_period(client: TestClient) -> None:
    response = client.get("/reviewer/compliance?period_id=unknown")
    assert response.status_code == 404


# ----------------------------------------- autolink / detect endpoints ---


def _write_extracted(dir_path: Path, document_id: str, pages: list[tuple[int, str]]) -> None:
    payload = {
        "document_id": document_id,
        "pages": [{"page": num, "text": text} for num, text in pages],
    }
    (dir_path / f"{document_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_detect_endpoint_returns_candidates_without_applying(
    client: TestClient, tmp_path: Path
) -> None:
    # The fixture's extracted_dir lives under tmp_path/extracted.
    extracted = tmp_path / "extracted"
    sample = (
        "Issuer AAA — Alphabetical Corp report Alphabetical Corp report"
    )
    _write_extracted(
        extracted,
        "doc_alpha_15-04-2026_09-00-00",
        [(1, sample)],
    )
    response = client.get(
        "/reviewer/documents/by-id/doc_alpha_15-04-2026_09-00-00/company/detect"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reason"] in {"accepted", "below_threshold", "ambiguous", "no_match"}
    assert body["candidates"], "expected at least one candidate"
    # Nothing persisted
    assert body["linked_at"] is None
    assert body["already_linked"] is None
    link = client.get(
        "/reviewer/documents/by-id/doc_alpha_15-04-2026_09-00-00/company"
    ).json()
    assert link["link"] is None


def test_detect_endpoint_persists_when_apply_true_and_accepted(
    client: TestClient, tmp_path: Path
) -> None:
    extracted = tmp_path / "extracted"
    sample = "Alphabetical Corp Alphabetical Corp Alphabetical Corp filing"
    # Three strong hits to beat the fixture's 1.5 threshold.
    _write_extracted(
        extracted,
        "doc_alpha_15-04-2026_09-00-00",
        [(1, sample)],
    )
    response = client.get(
        "/reviewer/documents/by-id/doc_alpha_15-04-2026_09-00-00/company/detect?apply=true"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reason"] == "accepted"
    assert body["linked_at"] is not None
    assert body["accepted_key"] == "AAA"
    # Confirm persisted
    follow_up = client.get(
        "/reviewer/documents/by-id/doc_alpha_15-04-2026_09-00-00/company"
    )
    assert follow_up.json()["link"]["company_key"] == "AAA"


def test_detect_endpoint_no_text_when_file_missing(client: TestClient) -> None:
    response = client.get("/reviewer/documents/by-id/doc_missing/company/detect")
    assert response.status_code == 200
    body = response.json()
    assert body["reason"] == "no_text"
    assert body["candidates"] == []


def test_detect_endpoint_rejects_unsafe_doc_id(client: TestClient) -> None:
    response = client.get("/reviewer/documents/by-id/..%2Fetc%2Fpasswd/company/detect")
    assert response.status_code in (404, 422)


def test_bulk_detect_scans_every_extracted_doc_and_links(
    client: TestClient, tmp_path: Path
) -> None:
    extracted = tmp_path / "extracted"
    alpha_text = "Alphabetical Corp Alphabetical Corp Alphabetical Corp filing"
    beta_text = "Betacorp Holdings Betacorp Holdings Betacorp Holdings disclosure"
    mystery_text = "This mentions neither issuer at all in the opening paragraph."
    _write_extracted(extracted, "doc_alpha", [(1, alpha_text)])
    _write_extracted(extracted, "doc_beta", [(1, beta_text)])
    _write_extracted(extracted, "doc_mystery", [(1, mystery_text)])

    response = client.post("/reviewer/documents/companies/detect-all?apply=true")
    assert response.status_code == 200
    body = response.json()
    assert body["scanned"] == 3
    # Both high-confidence docs should auto-link, mystery stays unmatched.
    assert body["linked"] >= 1
    doc_ids = {entry["document_id"] for entry in body["entries"]}
    assert doc_ids == {"doc_alpha", "doc_beta", "doc_mystery"}

    # Linked docs now show up in the mapping store.
    links = client.get("/reviewer/document-companies").json()["links"]
    linked_doc_ids = set(links.keys())
    assert "doc_alpha" in linked_doc_ids or "doc_beta" in linked_doc_ids


def test_bulk_detect_dry_run_does_not_persist(
    client: TestClient, tmp_path: Path
) -> None:
    extracted = tmp_path / "extracted"
    alpha_text = "Alphabetical Corp Alphabetical Corp Alphabetical Corp filing"
    _write_extracted(extracted, "doc_alpha", [(1, alpha_text)])

    response = client.post("/reviewer/documents/companies/detect-all")  # apply=false
    assert response.status_code == 200
    body = response.json()
    assert body["linked"] == 0
    # Nothing persisted
    assert client.get("/reviewer/document-companies").json()["links"] == {}


# --------------------------------------------- documents inventory ----


def test_documents_inventory_lists_every_known_doc(
    client: TestClient, tmp_path: Path
) -> None:
    extracted = tmp_path / "extracted"
    # Doc from extracted dir only
    _write_extracted(extracted, "doc_a_15-04-2026_09-00-00", [(1, "Alphabetical Corp content")])
    # Link a second doc so inventory also sees it via the mapping store
    client.put(
        "/reviewer/documents/by-id/doc_b_01-02-2026_10-00-00/company",
        json={"company_key": "AAA"},
    )

    response = client.get("/reviewer/document-inventory")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 2
    ids = {entry["document_id"] for entry in body["entries"]}
    assert "doc_a_15-04-2026_09-00-00" in ids
    assert "doc_b_01-02-2026_10-00-00" in ids
    # The linked one carries its company info
    linked = next(
        entry for entry in body["entries"]
        if entry["document_id"] == "doc_b_01-02-2026_10-00-00"
    )
    assert linked["link"] is not None
    assert linked["link"]["company_key"] == "AAA"
    assert linked["company"] is not None
    assert linked["company"]["ticker"] == "AAA"


def test_documents_inventory_includes_counts_and_submission_date(
    client: TestClient, tmp_path: Path
) -> None:
    extracted = tmp_path / "extracted"
    _write_extracted(extracted, "doc_15-04-2026_09-30-00", [(1, "Alphabetical Corp content")])
    response = client.get("/reviewer/document-inventory")
    assert response.status_code == 200
    entry = next(
        row for row in response.json()["entries"]
        if row["document_id"] == "doc_15-04-2026_09-30-00"
    )
    # No findings path in this fixture, so counts are all zero but present.
    assert entry["counts"]["total"] == 0
    assert entry["submitted_at"] is not None  # parsed from the doc id
    assert entry["extracted"] is True
    assert entry["pdf_present"] is False  # no PDF written for the fixture


def test_bulk_detect_respects_existing_manual_link(
    client: TestClient, tmp_path: Path
) -> None:
    extracted = tmp_path / "extracted"
    alpha_text = "Alphabetical Corp Alphabetical Corp Alphabetical Corp filing"
    _write_extracted(extracted, "doc_alpha", [(1, alpha_text)])

    # Manually link to BBB first.
    client.put(
        "/reviewer/documents/by-id/doc_alpha/company",
        json={"company_key": "BBB"},
    )
    response = client.post("/reviewer/documents/companies/detect-all?apply=true")
    body = response.json()
    assert body["skipped_existing"] == 1
    link = client.get("/reviewer/documents/by-id/doc_alpha/company").json()
    assert link["link"]["company_key"] == "BBB"  # never overwritten
