"""Phase D OCR CER measurement workflow — scaffold then measure.

Two subcommands:

* ``scaffold`` writes one ``.txt`` file per candidate OCR page under
  ``Data/reports/ocr_rerun/transcribe/{document_id}/p{page:04d}.txt``.
  Each file is pre-populated with the current OCR output so the reviewer
  only has to fix the glitches, not type the page from scratch. A header
  comment names the deterministic anchor that will unlock if the page's
  CER lands at or below the policy threshold.

* ``measure`` reads every ``p*.txt`` file, strips the header comment,
  compares the reviewer's text against the raw OCR output, and writes
  the resulting measurements to
  ``Data/reports/ocr_rerun/cer_measurements.json``.

Nothing in this workflow contacts the network.

Usage::

    python -m jsc_review.documents.ocr_transcribe scaffold
    # ... reviewer edits the generated .txt files in place ...
    python -m jsc_review.documents.ocr_transcribe measure
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from jsc_review.companies.autolink import iter_pages_from_ocr
from jsc_review.core.settings import get_settings
from jsc_review.documents.cer_measurement import (
    load_measurements,
    measure_page,
    save_measurements,
)
from jsc_review.documents.ocr_metrics import ocr_max_character_error_rate
from jsc_review.review.findings import (
    HEADING_VARIANTS,
    MULTI_ANCHOR_RULES,
    _find_variant,
    _find_variants_on_same_page,
)
from jsc_review.review.registry import load_selected_requirements

_LOGGER = logging.getLogger("jsc_review.documents.ocr_transcribe")

_HEADER_FENCE = "---"
_HEADER_PREFIX = "# "


def _transcribe_root(reports_dir: Path) -> Path:
    return reports_dir / "ocr_rerun" / "transcribe"


def _page_filename(document_id: str, page: int) -> str:
    return f"p{page:04d}.txt"


# ---------------------------------------------------------- helpers --


def _ocr_pages_for(ocr_dir: Path, document_id: str) -> dict[int, str]:
    path = ocr_dir / f"{document_id}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(iter_pages_from_ocr(payload))


def candidate_pages() -> dict[tuple[str, int], list[tuple[str, str]]]:
    """Return every (doc, page) whose anchor matches on an OCR page but
    the row is still Needs-Review (so CER measurement would unlock it).

    Exposed publicly so other modules (for example the PaddleOCR rerun)
    can target the exact same page set this scaffolder would hand to a
    reviewer.
    """

    settings = get_settings()
    rows = json.loads(settings.findings_path.read_text(encoding="utf-8"))
    reqs = {
        r.requirement_id: r
        for r in load_selected_requirements(settings.requirements_register_path)
    }
    hits: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    ocr_cache: dict[str, dict[int, str]] = {}

    for row in rows:
        result = row.get("result", {})
        if not isinstance(result, dict) or result.get("status") != "Needs Review":
            continue
        document_id = str(row.get("document_id", ""))
        requirement_id = str(result.get("requirement_id", ""))
        requirement = reqs.get(requirement_id)
        if requirement is None:
            continue
        if document_id not in ocr_cache:
            ocr_cache[document_id] = _ocr_pages_for(settings.ocr_dir, document_id)
        ocr_pages = ocr_cache[document_id]
        if not ocr_pages:
            continue
        if requirement_id in HEADING_VARIANTS:
            hit = _find_variant(ocr_pages, HEADING_VARIANTS[requirement_id])
            if hit is not None:
                page, variant, _ = hit
                hits[(document_id, page)].append((requirement_id, variant))
        elif requirement_id in MULTI_ANCHOR_RULES:
            hit2 = _find_variants_on_same_page(
                ocr_pages, MULTI_ANCHOR_RULES[requirement_id]
            )
            if hit2 is not None:
                page = hit2[0]
                hits[(document_id, page)].append((requirement_id, "multi-anchor"))

    return hits


def _render_header(document_id: str, page: int, reqs_hit: list[tuple[str, str]]) -> str:
    unlock_list = "\n".join(
        f"{_HEADER_PREFIX}  - {req_id}: anchor '{variant}'" for req_id, variant in reqs_hit
    )
    return (
        f"{_HEADER_FENCE}\n"
        f"{_HEADER_PREFIX}document: {document_id}\n"
        f"{_HEADER_PREFIX}page:     {page}\n"
        f"{_HEADER_PREFIX}unlocks these Needs-Review rows if the CER lands "
        f"at or below the policy threshold:\n"
        f"{unlock_list}\n"
        f"{_HEADER_PREFIX}\n"
        f"{_HEADER_PREFIX}Edit everything below the fence to match the printed "
        f"page exactly.\n"
        f"{_HEADER_PREFIX}Delete this header or leave it; the measure step "
        f"strips lines starting with '# '.\n"
        f"{_HEADER_FENCE}\n"
    )


def _strip_header(text: str) -> str:
    """Drop every leading header line marked ``# ``; keep everything else."""

    lines = text.splitlines()
    body_start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == _HEADER_FENCE:
            body_start = index + 1
            continue
        if stripped.startswith(_HEADER_PREFIX.strip()):
            body_start = index + 1
            continue
        break
    return "\n".join(lines[body_start:]).strip()


# ----------------------------------------------------- scaffold -----


def scaffold(force: bool = False) -> int:
    settings = get_settings()
    hits = candidate_pages()
    if not hits:
        _LOGGER.info(
            "No candidate OCR pages. Every Needs-Review row that could be "
            "unlocked by CER measurement is already covered or the anchors "
            "do not match an OCR page."
        )
        return 0

    root = _transcribe_root(settings.reports_dir)
    root.mkdir(parents=True, exist_ok=True)

    ocr_cache: dict[str, dict[int, str]] = {}
    written = 0
    skipped = 0

    for (document_id, page), reqs_hit in sorted(
        hits.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])
    ):
        if document_id not in ocr_cache:
            ocr_cache[document_id] = _ocr_pages_for(settings.ocr_dir, document_id)
        pages = ocr_cache[document_id]
        ocr_text = pages.get(page, "")
        if not ocr_text:
            _LOGGER.warning("No OCR text for %s page %d, skipping.", document_id, page)
            continue

        doc_dir = root / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        target = doc_dir / _page_filename(document_id, page)
        if target.exists() and not force:
            skipped += 1
            continue
        header = _render_header(document_id, page, reqs_hit)
        target.write_text(header + ocr_text, encoding="utf-8")
        written += 1

    _LOGGER.info("Scaffolded transcription files: %d written, %d skipped.", written, skipped)
    _LOGGER.info("Edit the files under %s then run `measure`.", root)
    return 0


# ----------------------------------------------------- measure ------


def _iter_edited_files(root: Path) -> list[tuple[Path, str, int]]:
    """Return (file_path, document_id, page_number) for every transcript."""

    collected: list[tuple[Path, str, int]] = []
    if not root.exists():
        return collected
    for doc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(doc_dir.glob("p*.txt")):
            stem = path.stem  # ``p0088`` etc.
            if not (stem.startswith("p") and stem[1:].isdigit()):
                continue
            page = int(stem[1:])
            collected.append((path, doc_dir.name, page))
    return collected


def measure() -> int:
    settings = get_settings()
    threshold = ocr_max_character_error_rate()
    root = _transcribe_root(settings.reports_dir)
    files = _iter_edited_files(root)
    if not files:
        _LOGGER.warning(
            "No transcription files under %s. Run `scaffold` first.", root
        )
        return 1

    existing = load_measurements(settings.cer_measurements_path)
    kept = [m for m in existing.measurements]
    measured_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")

    ocr_cache: dict[str, dict[int, str]] = {}
    added = 0

    for path, document_id, page in files:
        if document_id not in ocr_cache:
            ocr_cache[document_id] = _ocr_pages_for(settings.ocr_dir, document_id)
        pages = ocr_cache[document_id]
        hypothesis = pages.get(page, "")
        if not hypothesis:
            _LOGGER.warning(
                "No OCR output for %s page %d; cannot measure.", document_id, page
            )
            continue
        reference = _strip_header(path.read_text(encoding="utf-8"))
        if not reference.strip():
            _LOGGER.warning("Transcript %s is empty; skipping.", path.name)
            continue
        measurement = measure_page(
            reference,
            hypothesis,
            document_id=document_id,
            page=page,
            engine="tesseract-ara+eng",
            threshold=threshold,
            measured_at=measured_ts,
        )
        if measurement is None:
            _LOGGER.warning(
                "Measurement failed for %s page %d (empty reference).",
                document_id,
                page,
            )
            continue
        # Replace any existing measurement for the same (doc, page, engine).
        kept = [
            m
            for m in kept
            if not (
                m.document_id == measurement.document_id
                and m.page == measurement.page
                and m.engine == measurement.engine
            )
        ]
        kept.append(measurement)
        added += 1
        status = "OK" if measurement.verified_evidence_eligible else "ABOVE THRESHOLD"
        _LOGGER.info(
            "%s / p%d: CER %.2f%% (%d ref chars) -> %s",
            document_id,
            page,
            measurement.cer * 100,
            measurement.reference_chars,
            status,
        )

    save_measurements(
        kept,
        settings.cer_measurements_path,
        threshold=threshold,
        anchor_consensus=existing.anchor_consensus,
    )
    verified = sum(1 for m in kept if m.verified_evidence_eligible)
    _LOGGER.info(
        "Wrote %d measurements (%d verified-eligible) to %s",
        len(kept),
        verified,
        settings.cer_measurements_path,
    )
    _LOGGER.info("Added or updated this run: %d", added)
    return 0


# ----------------------------------------------------- entrypoint ---


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jsc_review.documents.ocr_transcribe",
        description=(
            "Phase D workflow for OCR CER measurement. Scaffolds editable "
            "transcription files for the OCR pages whose anchors already "
            "match a deterministic Needs-Review row, then measures the "
            "character error rate between your edits and the OCR output."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    scaffold_cmd = sub.add_parser(
        "scaffold",
        help="Write one editable file per candidate OCR page.",
    )
    scaffold_cmd.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing transcription files.",
    )
    sub.add_parser(
        "measure",
        help="Compute CER against every edited transcription and write the store.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.command == "scaffold":
        return scaffold(force=args.force)
    if args.command == "measure":
        return measure()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["main", "scaffold", "measure", "candidate_pages"]
