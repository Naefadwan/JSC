"""Enrich every Needs-Review finding with the local LLM's reasoning.

For each row with ``status == 'Needs Review'``, the CLI calls the local
model on the existing evidence passage (or re-retrieves the top passage
when no evidence is attached) and appends the model's reasoning + quote
to the finding's ``reason`` field under a ``نموذج محلي:`` prefix.

The row's ``status``, ``human_review``, and ``evaluation_method`` are
*not* changed. Policy says only deterministic rules with verified evidence
can clear a referral; the model's text is additional context for the
human reviewer, not a decision.

Writes back to ``findings.json`` atomically. Nothing leaves 127.0.0.1.

Usage::

    python -m jsc_review.review.enrich_with_model
    python -m jsc_review.review.enrich_with_model --document 131_...
    python -m jsc_review.review.enrich_with_model --requirement DIS-056
    python -m jsc_review.review.enrich_with_model --skip-if-enriched
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from jsc_review.companies.autolink import (
    iter_pages_from_extracted,
    iter_pages_from_ocr,
    merge_pages_with_ocr,
)
from jsc_review.core.settings import get_settings
from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval import dense as dense_module
from jsc_review.review.findings import locate_passage
from jsc_review.review.ollama import LocalModelError, ModelJudgment, judge_passage
from jsc_review.review.registry import load_selected_requirements

_LOGGER = logging.getLogger("jsc_review.review.enrich_with_model")

_ENRICHMENT_MARKER = "نموذج محلي:"


def _load_pages(extracted_dir: Path, ocr_dir: Path, document_id: str) -> dict[int, str]:
    native_file = extracted_dir / f"{document_id}.json"
    ocr_file = ocr_dir / f"{document_id}.json"
    native: list[tuple[int, str]] = []
    ocr: list[tuple[int, str]] = []
    if native_file.exists():
        payload = json.loads(native_file.read_text(encoding="utf-8"))
        native = list(iter_pages_from_extracted(payload))
    if ocr_file.exists():
        payload = json.loads(ocr_file.read_text(encoding="utf-8"))
        ocr = list(iter_pages_from_ocr(payload))
    return dict(merge_pages_with_ocr(native, ocr))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jsc_review.review.enrich_with_model",
        description=(
            "Append the local model's reasoning to every Needs-Review "
            "finding. The row's status is unchanged. Nothing leaves 127.0.0.1."
        ),
    )
    parser.add_argument("--document", help="Restrict to one document id.")
    parser.add_argument("--requirement", help="Restrict to one requirement id.")
    parser.add_argument(
        "--skip-if-enriched",
        action="store_true",
        help="Skip rows whose reason already contains the enrichment marker.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N rows (0 = all).",
    )
    return parser.parse_args(argv)


def _enrich_reason(existing: str, judgment: ModelJudgment) -> str:
    """Append the model's take to the existing reason text."""

    confidence_pct = int(round(judgment.confidence * 100))
    quote = judgment.quote.strip()
    quote_part = f'  اقتباس: "{quote[:200]}"' if quote else ""
    addition = (
        f"\n\n{_ENRICHMENT_MARKER} رأى النموذج أن الحالة {judgment.status} "
        f"بثقة {confidence_pct}٪. {judgment.reason}{quote_part}"
    )
    return existing.rstrip() + addition


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    settings = get_settings()
    if not settings.findings_path.exists():
        _LOGGER.error("Findings file missing: %s", settings.findings_path)
        return 1

    rows: list[dict[str, Any]] = json.loads(settings.findings_path.read_text(encoding="utf-8"))

    candidates: list[tuple[int, dict[str, Any]]] = []
    for position, row in enumerate(rows):
        result = row.get("result", {})
        if not isinstance(result, dict):
            continue
        if result.get("status") != "Needs Review":
            continue
        if args.document and row.get("document_id") != args.document:
            continue
        if args.requirement and result.get("requirement_id") != args.requirement:
            continue
        if args.skip_if_enriched and _ENRICHMENT_MARKER in str(result.get("reason", "")):
            continue
        candidates.append((position, row))

    if args.limit > 0:
        candidates = candidates[: args.limit]
    _LOGGER.info("Needs-Review rows to enrich: %d", len(candidates))
    if not candidates:
        return 0

    requirements_by_id = {
        req.requirement_id: req
        for req in load_selected_requirements(settings.requirements_register_path)
    }

    bm25 = bm25_module.load_index(settings.bm25_index_path)
    dense = None
    if settings.use_dense_at_review and settings.dense_index_path.exists():
        _LOGGER.info("Loading dense FAISS index + BGE-M3 encoder ...")
        bare = dense_module.load_index(settings.dense_index_path)
        from jsc_review.retrieval.embedding import (
            SentenceTransformerEncoder,
            embedding_settings,
        )

        encoder = SentenceTransformerEncoder(embedding_settings())
        dense = bare.with_encoder(encoder)

    enriched = 0
    skipped = 0
    pages_cache: dict[str, dict[int, str]] = {}

    for index, (position, row) in enumerate(candidates, start=1):
        document_id = str(row.get("document_id", ""))
        result = row["result"]
        requirement_id = str(result.get("requirement_id", ""))
        requirement = requirements_by_id.get(requirement_id)
        if requirement is None:
            _LOGGER.warning("[%d/%d] skip: %s unknown", index, len(candidates), requirement_id)
            skipped += 1
            continue

        if document_id not in pages_cache:
            pages_cache[document_id] = _load_pages(
                settings.extracted_dir, settings.ocr_dir, document_id
            )
        pages = pages_cache[document_id]
        if not pages:
            _LOGGER.warning(
                "[%d/%d] skip: %s / %s has no pages",
                index,
                len(candidates),
                document_id,
                requirement_id,
            )
            skipped += 1
            continue

        # Prefer the quote already attached to the finding; it is what the
        # UI shows the reviewer. Fall back to a fresh retrieval only when
        # no evidence is attached, which should be rare.
        existing_evidence = result.get("evidence")
        passage_text: str | None = None
        is_ocr = False
        if isinstance(existing_evidence, dict):
            text = str(existing_evidence.get("text", "")).strip()
            if text:
                passage_text = text
        if passage_text is None:
            located = locate_passage(document_id, requirement, pages, bm25, dense=dense)
            if located is None:
                _LOGGER.info(
                    "[%d/%d] skip: %s / %s has no retrievable passage",
                    index,
                    len(candidates),
                    document_id,
                    requirement_id,
                )
                skipped += 1
                continue
            passage_text = located.text
            is_ocr = located.passage_is_ocr

        try:
            judgment = judge_passage(requirement, passage_text, passage_is_ocr=is_ocr)
        except LocalModelError as exc:
            _LOGGER.warning(
                "[%d/%d] model error on %s / %s: %s",
                index,
                len(candidates),
                document_id,
                requirement_id,
                exc,
            )
            skipped += 1
            continue

        new_reason = _enrich_reason(str(result.get("reason", "")), judgment)
        rows[position]["result"]["reason"] = new_reason
        enriched += 1
        _LOGGER.info(
            "[%d/%d] %s / %s -> model says %s (conf %.2f)",
            index,
            len(candidates),
            document_id,
            requirement_id,
            judgment.status,
            judgment.confidence,
        )

        # Persist after every row so a long run can resume. The atomic
        # rename keeps the file consistent even if the interpreter is
        # killed mid-write.
        tmp = settings.findings_path.with_suffix(settings.findings_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(settings.findings_path)

    _LOGGER.info("Enriched: %d, Skipped: %d", enriched, skipped)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
