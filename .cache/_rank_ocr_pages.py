"""Rank OCR pages by how many deterministic Needs-Review rows they unlock.

For every current Needs-Review finding, check whether its deterministic
rule would match the ANCHOR on any OCR page of that document. If yes,
that OCR page is a transcription candidate — once its CER is measured
and <= 5%, the finding flips to Met.

Prints the top 15 pages ranked by (unlocked_rows, document_id, page).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from jsc_review.companies.autolink import iter_pages_from_ocr
from jsc_review.core.settings import get_settings
from jsc_review.review.findings import (
    HEADING_VARIANTS,
    MULTI_ANCHOR_RULES,
    _find_variant,
    _find_variants_on_same_page,
)
from jsc_review.review.registry import load_selected_requirements


def _ocr_pages_for(ocr_dir: Path, document_id: str) -> dict[int, str]:
    path = ocr_dir / f"{document_id}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(iter_pages_from_ocr(payload))


def main() -> int:
    settings = get_settings()
    rows = json.loads(settings.findings_path.read_text(encoding="utf-8"))
    reqs = {r.requirement_id: r for r in load_selected_requirements(settings.requirements_register_path)}

    # (doc_id, page) -> list of (requirement_id, variant) that would hit it.
    page_hits: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    doc_ocr_cache: dict[str, dict[int, str]] = {}

    needs_review_count = 0
    anchor_in_ocr_count = 0

    for row in rows:
        result = row.get("result", {})
        if not isinstance(result, dict) or result.get("status") != "Needs Review":
            continue
        needs_review_count += 1
        doc = str(row.get("document_id", ""))
        req_id = str(result.get("requirement_id", ""))
        requirement = reqs.get(req_id)
        if requirement is None:
            continue
        if doc not in doc_ocr_cache:
            doc_ocr_cache[doc] = _ocr_pages_for(settings.ocr_dir, doc)
        ocr_pages = doc_ocr_cache[doc]
        if not ocr_pages:
            continue
        # Check if the deterministic rule's anchors match in OCR.
        if req_id in HEADING_VARIANTS:
            variants = HEADING_VARIANTS[req_id]
            hit = _find_variant(ocr_pages, variants)
            if hit is not None:
                page, variant, _ = hit
                page_hits[(doc, page)].append((req_id, variant))
                anchor_in_ocr_count += 1
        elif req_id in MULTI_ANCHOR_RULES:
            hit2 = _find_variants_on_same_page(ocr_pages, MULTI_ANCHOR_RULES[req_id])
            if hit2 is not None:
                page = hit2[0]
                page_hits[(doc, page)].append((req_id, "multi-anchor"))
                anchor_in_ocr_count += 1

    print(f"Needs-Review rows total: {needs_review_count}")
    print(f"Of which the anchor DOES match on an OCR page: {anchor_in_ocr_count}")
    print(f"Unique OCR pages that would unlock >=1 flip: {len(page_hits)}")
    print()

    ranked = sorted(
        page_hits.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    )

    print(f"{'rank':>4}  {'doc':<42} {'page':>5}  unlocks  requirements")
    print("-" * 100)
    for rank, ((doc, page), reqs_hit) in enumerate(ranked[:20], start=1):
        reqs_str = ", ".join(sorted({r for r, _ in reqs_hit}))
        doc_short = (doc[:38] + "...") if len(doc) > 40 else doc
        print(f"{rank:>4}  {doc_short:<42} {page:>5}  {len(reqs_hit):>7}  {reqs_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
