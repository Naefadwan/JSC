"""Scan every extracted document and auto-link the clear-cut matches.

Run after rebuilding the registry (or whenever the extracted pile grows) to
backfill ``Data/documents/document_companies.json``. Documents that already
have a manual link are left untouched. Ambiguous or low-score matches are
printed so a human can resolve them in the UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsc_review.companies.autolink import (
    decide_autolink,
    iter_pages_from_extracted,
    iter_pages_from_ocr,
    merge_pages_with_ocr,
    rank_candidates_with_fallback,
)

_DEEP_SCAN_PAGES_CLI = 40
from jsc_review.companies.mapping import get_link, set_link
from jsc_review.companies.registry import load_registry
from jsc_review.core.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-link extracted documents to companies.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the accepted links. Without this flag the run is a dry report.",
    )
    parser.add_argument(
        "--include-linked",
        action="store_true",
        help="Also re-detect documents that already have a manual link (never overwrites).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the Settings threshold (default from Settings).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    threshold = (
        args.threshold if args.threshold is not None else settings.auto_link_confidence_threshold
    )
    min_lead_margin = settings.auto_link_min_lead_margin
    scan_pages = settings.auto_link_scan_pages

    registry = load_registry(settings.companies_registry_path)
    if not registry.companies:
        print("Registry is empty. Run Data/companies/_build_registry.py first.", file=sys.stderr)
        return 1

    extracted_dir = settings.extracted_dir
    if not extracted_dir.exists():
        print(f"No extracted directory at {extracted_dir}.", file=sys.stderr)
        return 1

    scanned = linked = skipped = ambiguous = below = unmatched = 0
    attention: list[str] = []

    for path in sorted(extracted_dir.glob("*.json")):
        scanned += 1
        document_id = path.stem
        existing = get_link(settings.document_companies_path, document_id)
        if existing is not None and not args.include_linked:
            skipped += 1
            print(f"  SKIP     {document_id:<45s} already linked to {existing.company_key}")
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"  ERR      {document_id:<45s} could not read extracted JSON: {exc}", file=sys.stderr)
            unmatched += 1
            continue
        native_pages = list(iter_pages_from_extracted(payload))
        ocr_path = settings.ocr_dir / f"{document_id}.json"
        ocr_pages: list[tuple[int, str]] = []
        if ocr_path.exists():
            try:
                ocr_payload = json.loads(ocr_path.read_text(encoding="utf-8"))
                ocr_pages = list(iter_pages_from_ocr(ocr_payload))
            except (ValueError, OSError):
                pass
        pages = tuple(merge_pages_with_ocr(native_pages, ocr_pages))
        if not pages:
            print(f"  NO-TEXT  {document_id:<45s} extracted file has no text")
            unmatched += 1
            continue

        scan_result = rank_candidates_with_fallback(
            pages,
            registry,
            scan_pages=scan_pages,
            deep_scan_pages=_DEEP_SCAN_PAGES_CLI,
        )
        decision = decide_autolink(
            scan_result.candidates,
            threshold=threshold,
            min_lead_margin=min_lead_margin,
            from_deep_scan=scan_result.deep,
        )
        if decision.accepted is not None:
            company = decision.accepted.company
            if args.apply and existing is None:
                set_link(
                    settings.document_companies_path,
                    document_id=document_id,
                    company_key=company.key,
                    linked_by="auto-cli",
                )
                linked += 1
                print(
                    f"  LINKED   {document_id:<45s} -> {company.ticker or company.numeric_code:<8s} "
                    f"({company.name_ar}) score={decision.accepted.score:.2f}"
                )
            else:
                linked += 1
                print(
                    f"  WOULD    {document_id:<45s} -> {company.ticker or company.numeric_code:<8s} "
                    f"({company.name_ar}) score={decision.accepted.score:.2f}"
                )
        elif decision.reason == "ambiguous":
            ambiguous += 1
            top3 = ", ".join(
                f"{c.company.ticker or c.company.numeric_code}={c.score:.2f}"
                for c in decision.candidates[:3]
            )
            print(f"  AMBIG    {document_id:<45s} top-3: {top3}")
            attention.append(document_id)
        elif decision.reason == "deep_suggestion":
            below += 1
            top3 = ", ".join(
                f"{c.company.ticker or c.company.numeric_code}={c.score:.2f}"
                for c in decision.candidates[:3]
            )
            print(f"  DEEP     {document_id:<45s} suggest (needs confirm): {top3}")
            attention.append(document_id)
        elif decision.reason == "below_threshold":
            below += 1
            top3 = ", ".join(
                f"{c.company.ticker or c.company.numeric_code}={c.score:.2f}"
                for c in decision.candidates[:3]
            )
            print(f"  LOW      {document_id:<45s} top-3: {top3}")
            attention.append(document_id)
        else:
            unmatched += 1
            print(f"  NONE     {document_id:<45s} no company matched the first {scan_pages} pages")
            attention.append(document_id)

    print()
    print(f"Scanned: {scanned}")
    print(f"{'Linked' if args.apply else 'Would link'}: {linked}")
    print(f"Skipped (already linked): {skipped}")
    print(f"Ambiguous: {ambiguous}")
    print(f"Below threshold: {below}")
    print(f"Unmatched: {unmatched}")
    if attention:
        print()
        print("Needs manual pick in the UI:")
        for doc in attention:
            print(f"  - {doc}")
    if not args.apply:
        print()
        print("(dry run; pass --apply to persist the LINKED entries.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
