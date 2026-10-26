"""Tests for the disclosure-period compliance assessor."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from jsc_review.companies.compliance import (
    DisclosurePeriod,
    PeriodValidationError,
    assess_period,
    assess_period_from_disk,
    classify_submission,
    delete_period,
    load_periods,
    parse_submission_timestamp,
    save_periods,
    upsert_period,
)
from jsc_review.companies.mapping import DocumentCompanyLink, set_link
from jsc_review.companies.registry import Company, CompanyRegistry


def _registry() -> CompanyRegistry:
    return CompanyRegistry(
        companies=(
            Company(name_ar="Alpha", ticker="AAA", numeric_code="100001", sector_ar="Banks"),
            Company(name_ar="Beta", ticker="BBB", numeric_code="100002", sector_ar="Insurance"),
            Company(name_ar="Gamma", ticker="CCC", numeric_code="100003", sector_ar="Real Estate"),
        )
    )


def _links(mapping: dict[str, str]) -> dict[str, DocumentCompanyLink]:
    return {
        doc_id: DocumentCompanyLink(document_id=doc_id, company_key=key)
        for doc_id, key in mapping.items()
    }


def _period() -> DisclosurePeriod:
    return DisclosurePeriod(period_id="Q1-2026", due_date=date(2026, 4, 30))


# ---- parse_submission_timestamp -------------------------------------------


def test_parse_submission_timestamp_full_suffix() -> None:
    stamp = parse_submission_timestamp("131_10130_889_31-03-2026_12-52-11")
    assert stamp is not None
    assert stamp == datetime(2026, 3, 31, 12, 52, 11, tzinfo=UTC)


def test_parse_submission_timestamp_date_only() -> None:
    stamp = parse_submission_timestamp("foo_bar_31-03-2026")
    assert stamp is not None
    assert stamp.date() == date(2026, 3, 31)


def test_parse_submission_timestamp_falls_back_to_mtime(tmp_path: Path) -> None:
    pdf = tmp_path / "nothing-parseable.pdf"
    pdf.write_bytes(b"x")
    stamp = parse_submission_timestamp("nothing-parseable", pdf_path=pdf)
    assert stamp is not None
    assert stamp.tzinfo is UTC


def test_parse_submission_timestamp_returns_none_when_nothing_available() -> None:
    assert parse_submission_timestamp("nothing_at_all") is None


# ---- classify_submission --------------------------------------------------


def test_classify_submission_on_time() -> None:
    submission = datetime(2026, 4, 30, 10, 0, 0, tzinfo=UTC)
    assert classify_submission(submission, date(2026, 4, 30)) == "submitted_on_time"


def test_classify_submission_late() -> None:
    submission = datetime(2026, 5, 1, 0, 1, 0, tzinfo=UTC)
    assert classify_submission(submission, date(2026, 4, 30)) == "submitted_late"


def test_classify_submission_missing() -> None:
    assert classify_submission(None, date(2026, 4, 30)) == "not_submitted"


# ---- assess_period --------------------------------------------------------


def test_assess_period_classifies_every_company() -> None:
    links = _links({
        "131_10130_889_15-04-2026_09-00-00": "AAA",
        "131_10130_890_01-05-2026_09-00-00": "BBB",
    })
    report = assess_period(_period(), registry=_registry(), links=links)

    assert report.total == 3
    assert report.submitted_on_time_count == 1
    assert report.submitted_late_count == 1
    assert report.not_submitted_count == 1

    by_key = {entry.company_key: entry for entry in report.entries}
    assert by_key["AAA"].status == "submitted_on_time"
    alpha_doc = by_key["AAA"].document_id
    assert alpha_doc is not None and alpha_doc.endswith("15-04-2026_09-00-00")
    assert by_key["AAA"].days_from_due == -15
    assert by_key["BBB"].status == "submitted_late"
    assert by_key["BBB"].days_from_due == 1
    assert by_key["CCC"].status == "not_submitted"
    assert by_key["CCC"].submitted_at is None


def test_assess_period_picks_latest_submission_when_many(tmp_path: Path) -> None:
    links = _links({
        "131_10130_889_01-04-2026_09-00-00": "AAA",
        "131_10130_890_15-04-2026_09-00-00": "AAA",
        "131_10130_891_28-04-2026_09-00-00": "AAA",
    })
    report = assess_period(_period(), registry=_registry(), links=links)
    alpha = next(entry for entry in report.entries if entry.company_key == "AAA")
    assert alpha.document_id is not None
    assert alpha.document_id.endswith("28-04-2026_09-00-00")


def test_assess_period_ignores_links_to_unknown_companies() -> None:
    links = _links({"doc_15-04-2026_09-00-00": "ZZZ"})
    report = assess_period(_period(), registry=_registry(), links=links)
    assert report.submitted_on_time_count == 0
    assert report.not_submitted_count == 3


def test_assess_period_resolves_links_by_ticker_or_code() -> None:
    links = _links({
        "doc_15-04-2026_09-00-00": "100001",  # numeric code of AAA
    })
    report = assess_period(_period(), registry=_registry(), links=links)
    alpha = next(entry for entry in report.entries if entry.company_key == "AAA")
    assert alpha.status == "submitted_on_time"


# ---- periods persistence --------------------------------------------------


def test_save_and_load_periods_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "periods.json"
    save_periods(path, [_period()])
    assert path.exists()

    loaded = load_periods(path)
    assert len(loaded) == 1
    assert loaded[0].period_id == "Q1-2026"
    assert loaded[0].due_date == date(2026, 4, 30)


def test_load_periods_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_periods(tmp_path / "nope.json") == []


def test_load_periods_returns_empty_when_malformed(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    assert load_periods(path) == []


# ---- upsert_period / delete_period ---------------------------------------


def test_upsert_period_creates_new_entry(tmp_path: Path) -> None:
    path = tmp_path / "periods.json"
    created = upsert_period(
        path,
        period_id="Q1-2026",
        description="",
        due_date=date(2026, 4, 30),
    )
    assert created.period_id == "Q1-2026"
    assert [p.period_id for p in load_periods(path)] == ["Q1-2026"]


def test_upsert_period_rejects_duplicate_without_replace(tmp_path: Path) -> None:
    path = tmp_path / "periods.json"
    save_periods(path, [DisclosurePeriod(period_id="Q1-2026", due_date=date(2026, 4, 30))])
    try:
        upsert_period(
            path,
            period_id="Q1-2026",
            description="",
            due_date=date(2026, 5, 1),
        )
    except PeriodValidationError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected PeriodValidationError")


def test_upsert_period_replaces_when_allowed(tmp_path: Path) -> None:
    path = tmp_path / "periods.json"
    save_periods(path, [DisclosurePeriod(period_id="Q1-2026", due_date=date(2026, 4, 30))])
    upsert_period(
        path,
        period_id="Q1-2026",
        description="updated",
        due_date=date(2026, 5, 15),
        allow_replace=True,
    )
    loaded = load_periods(path)
    assert len(loaded) == 1
    assert loaded[0].description == "updated"
    assert loaded[0].due_date == date(2026, 5, 15)


def test_upsert_period_rejects_malformed_id(tmp_path: Path) -> None:
    path = tmp_path / "periods.json"
    try:
        upsert_period(
            path, period_id="bad id", description="", due_date=date(2026, 4, 30)
        )
    except PeriodValidationError:
        pass
    else:
        raise AssertionError("expected PeriodValidationError")


def test_delete_period_returns_true_on_hit_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "periods.json"
    save_periods(
        path,
        [
            DisclosurePeriod(period_id="Q1-2026", due_date=date(2026, 4, 30)),
            DisclosurePeriod(period_id="Q2-2026", due_date=date(2026, 7, 31)),
        ],
    )
    assert delete_period(path, "Q1-2026") is True
    assert [p.period_id for p in load_periods(path)] == ["Q2-2026"]


def test_delete_period_returns_false_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "periods.json"
    save_periods(path, [DisclosurePeriod(period_id="Q1-2026", due_date=date(2026, 4, 30))])
    assert delete_period(path, "nope") is False
    assert len(load_periods(path)) == 1


# ---- assess_period_from_disk ---------------------------------------------


def test_assess_period_from_disk(tmp_path: Path) -> None:
    registry_path = tmp_path / "companies.json"
    mapping_path = tmp_path / "map.json"
    from jsc_review.companies.registry import CompanyRegistryPayload

    registry_path.write_text(
        CompanyRegistryPayload(companies=list(_registry().companies)).model_dump_json(indent=2),
        encoding="utf-8",
    )
    set_link(mapping_path, document_id="doc_15-04-2026_09-00-00", company_key="AAA")

    report = assess_period_from_disk(
        _period(),
        registry_path=registry_path,
        mapping_path=mapping_path,
    )
    assert report.submitted_on_time_count == 1
