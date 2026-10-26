"""Unit tests for the company registry parser and cache round-trip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jsc_review.companies.registry import (
    Company,
    CompanyRegistry,
    CompanyRegistryPayload,
    _completeness_score,
    _deduplicate_companies,
    build_registry,
    load_registry,
    load_registry_payload,
    parse_company_rows,
)


def _rows() -> list[tuple[Any, ...]]:
    """Return a small two-sheet-style grid with a banner + header + rows."""

    return [
        ("البنوك",),  # sector banner
        (
            "اسم الشركة",
            "الاسم المختصر للشركة",
            "الرمز الحرفي",
            "الرمز الرقمي",
            "السوق",
            "الأسهم المدرجة",
            "الحالة",
            "ملاحظات",
        ),
        (
            "البنك الإسلامي الأردني",
            "بنك الإسلامي",
            "JOIB",
            111001,
            "1",
            200000000,
            "",
            "",
        ),
        ("التأمين",),  # new banner
        (
            "اسم الشركة",
            "الاسم المختصر للشركة",
            "الرمز الحرفي",
            "الرمز الرقمي",
            "السوق",
            "الأسهم المدرجة",
            "الحالة",
            "ملاحظات",
        ),
        (
            "التأمين الاردنية",
            "التأمين الاردنية",
            "JOIN",
            121004,
            "1",
            30000000,
            "",
            "",
        ),
        (None, None, None, None, None, None, None, None),  # blank row
    ]


def test_parse_company_rows_handles_banners_and_headers() -> None:
    companies = parse_company_rows(_rows())

    assert len(companies) == 2
    bank, insurance = companies
    assert bank.name_ar == "البنك الإسلامي الأردني"
    assert bank.ticker == "JOIB"
    assert bank.numeric_code == "111001"
    assert bank.sector_ar == "البنوك"
    assert bank.listed_shares == 200_000_000
    assert insurance.sector_ar == "التأمين"
    assert insurance.ticker == "JOIN"


def test_parse_company_rows_skips_rows_without_name() -> None:
    rows = [
        ("البنوك",),
        ("اسم الشركة", "الاسم المختصر للشركة", "الرمز الحرفي", "الرمز الرقمي"),
        ("", "", "FOO", 111111),  # no name
    ]

    assert parse_company_rows(rows) == []


def test_parse_company_rows_skips_rows_without_ticker_or_code() -> None:
    rows = [
        (
            "اسم الشركة",
            "الاسم المختصر للشركة",
            "الرمز الحرفي",
            "الرمز الرقمي",
        ),
        ("شركة بدون رموز", "", "", ""),
    ]

    assert parse_company_rows(rows) == []


def test_deduplicate_prefers_the_more_complete_record() -> None:
    sparse = Company(name_ar="x", ticker="AAA")
    full = Company(
        name_ar="x",
        short_name_ar="X",
        ticker="AAA",
        numeric_code="123",
        market="1",
        listed_shares=10,
        sector_ar="S",
        status="active",
        notes="note",
    )
    assert _completeness_score(full) > _completeness_score(sparse)
    result = _deduplicate_companies([sparse, full])
    assert len(result) == 1
    assert result[0] is full


def test_deduplicate_preserves_first_appearance_order_across_keys() -> None:
    a = Company(name_ar="a", ticker="AAA")
    b = Company(name_ar="b", ticker="BBB")
    c = Company(name_ar="a", ticker="AAA", sector_ar="richer")
    out = _deduplicate_companies([a, b, c])
    assert [company.ticker for company in out] == ["AAA", "BBB"]
    assert out[0].sector_ar == "richer"


def test_build_and_load_registry_round_trip(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    payload = CompanyRegistryPayload(
        source_excel="dummy.xlsx",
        companies=[
            Company(name_ar="A", ticker="AAA", numeric_code="100001", sector_ar="S1"),
            Company(name_ar="B", ticker="BBB", numeric_code="100002", sector_ar="S2"),
        ],
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    registry = load_registry(cache)
    assert isinstance(registry, CompanyRegistry)
    first = registry.by_ticker("aaa")
    second = registry.by_numeric_code("100002")
    direct = registry.by_key("AAA")
    assert first is not None and first.numeric_code == "100001"
    assert second is not None and second.ticker == "BBB"
    assert direct is not None and direct.name_ar == "A"
    assert registry.by_ticker("ZZZ") is None

    read_payload = load_registry_payload(cache)
    assert read_payload.source_excel == "dummy.xlsx"
    assert len(read_payload.companies) == 2


def test_load_registry_returns_empty_when_missing(tmp_path: Path) -> None:
    cache = tmp_path / "missing.json"
    registry = load_registry(cache)
    assert registry.companies == ()
    assert load_registry_payload(cache).companies == []


def test_build_registry_writes_cache_from_excel(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    from openpyxl import Workbook

    excel_path = tmp_path / "src.xlsx"
    cache_path = tmp_path / "out.json"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    for row in _rows():
        sheet.append(list(row))
    workbook.save(excel_path)

    payload = build_registry(excel_path, cache_path)
    assert cache_path.exists()
    assert payload.source_excel == str(excel_path)
    assert {company.ticker for company in payload.companies} == {"JOIB", "JOIN"}
