"""Rebuild the local company registry JSON from the source Excel file.

Run this after the Excel changes (e.g. a new sector or company is added) so
the API serves the fresh directory. The JSON is checked into
``Data/companies/companies.json`` and read by every route that needs the
company list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jsc_review.companies.registry import build_registry
from jsc_review.core.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the company registry JSON.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Path to the source Excel file (defaults to Settings.companies_excel_path).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to the JSON cache (defaults to Settings.companies_registry_path).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    excel = args.excel or settings.companies_excel_path
    cache = args.out or settings.companies_registry_path

    if not excel.exists():
        print(f"Excel file not found: {excel}", file=sys.stderr)
        return 1

    payload = build_registry(excel, cache)
    print(f"source: {excel}")
    print(f"wrote: {cache}")
    print(f"companies: {len(payload.companies)}")
    if payload.companies:
        sectors: dict[str, int] = {}
        for company in payload.companies:
            sectors[company.sector_ar or "(no sector)"] = (
                sectors.get(company.sector_ar or "(no sector)", 0) + 1
            )
        for sector, count in sorted(sectors.items(), key=lambda pair: (-pair[1], pair[0])):
            print(f"  {sector}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
