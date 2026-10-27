"""List every deterministic requirement that is currently ``Needs Review``.

For each failure the CLI prints the document, the requirement, the anchor
list we tried, and the top retrieval hits so a reviewer can see which
Arabic variant the report actually uses. It is read-only.

Usage::

    python -m jsc_review.review.diagnose_deterministic
    python -m jsc_review.review.diagnose_deterministic --document 131_10130_889_31-03-2026_12-52-11
    python -m jsc_review.review.diagnose_deterministic --requirement DIS-062

Nothing in this CLI calls out to a remote service.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from jsc_review.core.settings import get_settings
from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval.search import HybridConfig, hybrid_search
from jsc_review.review.findings import (
    EARLY_PAGE_RULES,
    HEADING_VARIANTS,
    MULTI_ANCHOR_RULES,
    TABLE_STRUCTURE_RULES,
    THRESHOLD_RULES,
)
from jsc_review.review.registry import load_selected_requirements
from jsc_review.reviewer.findings import load_findings

_LOGGER = logging.getLogger("jsc_review.review.diagnose_deterministic")


def _deterministic_requirement_ids() -> set[str]:
    """Every requirement id handled by a deterministic rule."""

    return (
        set(HEADING_VARIANTS)
        | set(THRESHOLD_RULES)
        | set(MULTI_ANCHOR_RULES)
        | set(EARLY_PAGE_RULES)
        | set(TABLE_STRUCTURE_RULES)
    )


def _anchors_for(req_id: str) -> tuple[str, ...]:
    """Return the anchor list for a requirement (empty for threshold rules)."""

    if req_id in HEADING_VARIANTS:
        return HEADING_VARIANTS[req_id]
    if req_id in MULTI_ANCHOR_RULES:
        return tuple(a for group in MULTI_ANCHOR_RULES[req_id] for a in group)
    if req_id in EARLY_PAGE_RULES:
        return EARLY_PAGE_RULES[req_id][0]
    if req_id in TABLE_STRUCTURE_RULES:
        return TABLE_STRUCTURE_RULES[req_id][0]
    return ()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jsc_review.review.diagnose_deterministic",
        description=(
            "List deterministic requirements that are currently Needs Review, "
            "plus the anchor list and the top-5 retrieval hits for each case."
        ),
    )
    parser.add_argument("--document", help="Restrict to one document id.")
    parser.add_argument("--requirement", help="Restrict to one requirement id.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many retrieval hits to show per failure (default 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    settings = get_settings()
    if not settings.findings_path.exists():
        _LOGGER.error("Findings file is missing: %s", settings.findings_path)
        return 1
    deterministic_ids = _deterministic_requirement_ids()
    rows = load_findings(settings.findings_path)
    failures = [
        (document_id, result)
        for document_id, result in rows
        if result.requirement_id in deterministic_ids
        and result.status == "Needs Review"
        and (args.document is None or document_id == args.document)
        and (args.requirement is None or result.requirement_id == args.requirement)
    ]
    _LOGGER.info("Deterministic Needs-Review failures: %d", len(failures))
    if not failures:
        return 0

    bm25 = None
    if settings.bm25_index_path.exists():
        try:
            bm25 = bm25_module.load_index(settings.bm25_index_path)
        except (OSError, ValueError, KeyError) as exc:
            _LOGGER.warning("BM25 index could not load: %s", exc)

    requirements_by_id = {
        req.requirement_id: req
        for req in load_selected_requirements(settings.requirements_register_path)
    }

    for document_id, result in failures:
        _LOGGER.info("")
        _LOGGER.info("=" * 70)
        _LOGGER.info("doc:   %s", document_id)
        _LOGGER.info("req:   %s", result.requirement_id)
        _LOGGER.info("reason: %s", result.reason)
        anchors = _anchors_for(result.requirement_id)
        if anchors:
            _LOGGER.info("anchors tried:")
            for anchor in anchors:
                _LOGGER.info("  - %s", anchor)
        requirement = requirements_by_id.get(result.requirement_id)
        if bm25 is None or requirement is None:
            continue
        hits = hybrid_search(
            requirement,
            document_id,
            bm25=bm25,
            config=HybridConfig(final_top_k=args.top_k, dedupe_by_page=True),
        )
        if not hits:
            _LOGGER.info("retrieval: (no hits for this document in the global BM25 index)")
            continue
        _LOGGER.info("top-%d retrieval hits:", min(args.top_k, len(hits)))
        for position, hit in enumerate(hits, start=1):
            snippet = _single_line(hit.text)[:200]
            _LOGGER.info(
                "  %d. page %d · %s · %s",
                position,
                hit.page,
                hit.section or "(no section)",
                snippet,
            )
    return 0


def _single_line(text: str) -> str:
    return " ".join(text.split())


# Re-exported so the CLI can be scripted (``main`` returns an exit code).
__all__ = ["main"]


# -- entry point ------------------------------------------------------------


def _entry_point() -> None:  # pragma: no cover - module entry point
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    _entry_point()


# A small helper for Path-typed callers who want to run against a different
# findings file without changing Settings.
def diagnose(findings_path: Path, requirement_ids: set[str] | None = None) -> list[tuple[str, str]]:
    """Return the list of (document_id, requirement_id) deterministic failures."""

    rows = load_findings(findings_path)
    want = requirement_ids or _deterministic_requirement_ids()
    return [
        (document_id, result.requirement_id)
        for document_id, result in rows
        if result.requirement_id in want and result.status == "Needs Review"
    ]
