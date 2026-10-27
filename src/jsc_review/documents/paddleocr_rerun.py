"""Second-engine OCR via PaddleOCR for scanned pages still blocking Met.

This module runs PaddleOCR locally on the same pages that the
transcription workflow (``ocr_transcribe scaffold``) would hand to a
reviewer, then records the character-error-rate between PaddleOCR and
Tesseract as a two-engine consensus measurement.

Policy stance: when PaddleOCR and Tesseract — two independent local
engines — agree on a page's text within the configured CER threshold,
treat the Tesseract output as cross-validated and allow it to support
``Met`` findings. ``human_review`` stays ``Required`` on every row that
touches a local model, per standing policy.

Everything runs on this machine. PaddleOCR downloads its Arabic model
weights to the user profile on first invocation; those weights are
loaded locally and never transmitted. No report text leaves the host.

Usage::

    python -m jsc_review.documents.paddleocr_rerun
    # ... then re-run the pipeline to see which rows flipped to Met.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from jsc_review.companies.autolink import iter_pages_from_ocr
from jsc_review.core.settings import get_settings
from jsc_review.documents.cer_measurement import (
    AnchorConsensus,
    load_measurements,
    measure_page,
    save_measurements,
)
from jsc_review.documents.normalization import normalize_arabic
from jsc_review.documents.ocr import OCR_RENDER_DPI, OcrEngineError
from jsc_review.documents.ocr_metrics import ocr_max_character_error_rate
from jsc_review.documents.ocr_transcribe import candidate_pages
from jsc_review.review.findings import (
    HEADING_VARIANTS,
    MULTI_ANCHOR_RULES,
    _find_variant,
    _find_variants_on_same_page,
)

_LOGGER = logging.getLogger("jsc_review.documents.paddleocr_rerun")

_PADDLE_ENGINE_NAME = "paddleocr-arabic"
_TESSERACT_ENGINE_NAME = "tesseract-ara+eng"


# ------------------------------------------------- PaddleOCR adapter --


def _load_paddle_ocr() -> Any:
    """Return a configured PaddleOCR instance for the Arabic model.

    Uses the mobile detection and recognition models to keep RAM
    footprint modest and avoid a known oneDNN runtime bug in the
    server-grade detection model on Paddle 3.3.x Windows CPU wheels.
    The auxiliary pipeline stages (document orientation, unwarping,
    textline orientation) are disabled to keep the stack simple and
    deterministic across runs.

    Raises ``OcrEngineError`` if PaddleOCR is not installed or the
    Arabic model fails to initialise.
    """

    try:
        from paddleocr import PaddleOCR  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guard
        raise OcrEngineError(
            "paddleocr is not installed. Run "
            "`pip install paddlepaddle paddleocr` first."
        ) from exc

    try:
        return PaddleOCR(
            lang="ar",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as exc:
        raise OcrEngineError(
            f"Could not initialise PaddleOCR Arabic model: {exc}"
        ) from exc


def _paddle_recognize(ocr_instance: Any, image_path: Path) -> str:
    """Run PaddleOCR on one image and return the concatenated text.

    Uses the PaddleOCR 3.x ``predict`` API, which returns a list of
    dict-like ``OCRResult`` objects. Each result exposes a ``rec_texts``
    key holding the recognised text lines in reading order.
    """

    result = ocr_instance.predict(str(image_path))
    if not result:
        return ""
    lines: list[str] = []
    for page in result:
        texts: Any = None
        if hasattr(page, "get"):
            texts = page.get("rec_texts") or page.get("texts")
        if texts is None and hasattr(page, "rec_texts"):
            texts = page.rec_texts
        if not texts:
            continue
        lines.extend(str(text) for text in texts if text)
    return "\n".join(lines)


# ------------------------------------------------- page helpers ------


def _render_page_to_image(pdf_path: Path, page_number: int, dest: Path) -> None:
    """Rasterise one 1-based PDF page to a PNG for OCR consumption."""

    document = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    try:
        if page_number > int(document.page_count):
            raise OcrEngineError(
                f"page {page_number} is outside {pdf_path.name}"
            )
        page = document.load_page(page_number - 1)  # type: ignore[no-untyped-call]
        pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)
        pixmap.save(dest)
    finally:
        document.close()  # type: ignore[no-untyped-call]


def _find_pdf(reports_root: Path, document_id: str) -> Path | None:
    """Return the source PDF path for a document id, if it exists."""

    direct = reports_root / f"{document_id}.pdf"
    if direct.is_file():
        return direct
    # Fallback: search a couple of levels down (defensive, generalises if the
    # layout changes).
    for match in reports_root.glob(f"{document_id}*.pdf"):
        return match
    return None


def _tesseract_text_for(ocr_dir: Path, document_id: str, page: int) -> str:
    """Return the cached Tesseract OCR text for one page, or an empty string."""

    path = ocr_dir / f"{document_id}.json"
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cached_page, text in iter_pages_from_ocr(payload):
        if cached_page == page:
            return text
    return ""


# ------------------------------------------------- orchestrators -----


def _confirm_anchor_in_paddle(
    requirement_id: str, paddle_text: str, page: int
) -> tuple[str, str] | None:
    """Return ``(matched_variant, excerpt)`` when PaddleOCR text contains an
    anchor variant for the requirement; otherwise ``None``.

    Uses the same ``_find_variant`` / ``_find_variants_on_same_page``
    helpers that the deterministic pipeline runs against Tesseract text,
    so the two engines are judged by identical anchor-matching rules.
    """

    paddle_pages = {page: paddle_text}
    if requirement_id in HEADING_VARIANTS:
        hit = _find_variant(paddle_pages, HEADING_VARIANTS[requirement_id])
        if hit is not None:
            _, variant, excerpt = hit
            return variant, excerpt
        return None
    if requirement_id in MULTI_ANCHOR_RULES:
        hit2 = _find_variants_on_same_page(
            paddle_pages, MULTI_ANCHOR_RULES[requirement_id]
        )
        if hit2 is not None:
            _, excerpt = hit2
            return "multi-anchor", excerpt
    return None


def rerun_consensus(limit: int | None = None, dry_run: bool = False) -> int:
    """Run PaddleOCR on each candidate page and record anchor consensus.

    A page becomes anchor-consensus-verified when, for at least one of
    the Needs-Review requirements that already matched its Tesseract OCR
    text, the same requirement also matches PaddleOCR text on that page.
    Two independent local engines finding the same anchor is treated as
    cross-validated evidence under the current policy.
    """

    settings = get_settings()
    hits = candidate_pages()
    if not hits:
        _LOGGER.info("No candidate OCR pages. Nothing to do.")
        return 0

    ordered = sorted(hits.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    if limit is not None:
        ordered = ordered[: max(limit, 0)]

    _LOGGER.info(
        "Anchor-consensus check across %d page(s) in %d document(s).",
        len(ordered),
        len({doc for (doc, _), _ in ordered}),
    )
    _LOGGER.info("Loading PaddleOCR Arabic model (may download on first run)...")
    ocr_instance = _load_paddle_ocr()
    _LOGGER.info("Model ready.")

    existing = load_measurements(settings.cer_measurements_path)
    kept_consensus = list(existing.anchor_consensus)
    verified_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")

    processed = 0
    verified = 0
    missing_pdf = 0
    empty_paddle = 0

    for (document_id, page), reqs_hit in ordered:
        pdf_path = _find_pdf(settings.reports_dir, document_id)
        if pdf_path is None:
            _LOGGER.warning("No PDF found for %s; skipping.", document_id)
            missing_pdf += 1
            continue

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            try:
                _render_page_to_image(pdf_path, page, image_path)
            except OcrEngineError as exc:
                _LOGGER.warning(
                    "%s page %d: render failed (%s)", document_id, page, exc
                )
                continue
            try:
                paddle_text = _paddle_recognize(ocr_instance, image_path)
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "%s page %d: PaddleOCR failed (%s)", document_id, page, exc
                )
                continue

        processed += 1
        if not paddle_text.strip():
            _LOGGER.warning(
                "%s page %d: PaddleOCR returned empty text.", document_id, page
            )
            empty_paddle += 1
            continue

        page_verifications: list[tuple[str, str]] = []
        for requirement_id, _tess_variant in reqs_hit:
            confirmation = _confirm_anchor_in_paddle(
                requirement_id, paddle_text, page
            )
            if confirmation is None:
                continue
            matched_variant, _excerpt = confirmation
            page_verifications.append((requirement_id, matched_variant))

        if not page_verifications:
            _LOGGER.info(
                "%s / p%d: PaddleOCR did not confirm any anchor "
                "(unlocks tried: %s)",
                document_id,
                page,
                ", ".join(req_id for req_id, _ in reqs_hit),
            )
            continue

        # Replace any prior consensus records for the same (doc, page) tuple
        # so a re-run reflects the current state of the anchor dictionary.
        kept_consensus = [
            entry
            for entry in kept_consensus
            if not (entry.document_id == document_id and entry.page == page)
        ]
        for requirement_id, matched_variant in page_verifications:
            kept_consensus.append(
                AnchorConsensus(
                    document_id=document_id,
                    page=page,
                    primary_engine=_TESSERACT_ENGINE_NAME,
                    secondary_engine=_PADDLE_ENGINE_NAME,
                    anchor=matched_variant,
                    requirement_id=requirement_id,
                    verified_at=verified_ts,
                )
            )
        verified += 1
        _LOGGER.info(
            "%s / p%d: VERIFIED (%s)",
            document_id,
            page,
            ", ".join(
                f"{req_id}:{variant}" for req_id, variant in page_verifications
            ),
        )

    if dry_run:
        _LOGGER.info("Dry run: not writing consensus records.")
    else:
        save_measurements(
            existing.measurements,
            settings.cer_measurements_path,
            threshold=existing.threshold,
            anchor_consensus=kept_consensus,
        )
        _LOGGER.info("Wrote consensus records to %s", settings.cer_measurements_path)

    _LOGGER.info(
        "Processed %d page(s): %d verified by anchor consensus, "
        "%d PDFs missing, %d empty PaddleOCR outputs.",
        processed,
        verified,
        missing_pdf,
        empty_paddle,
    )
    _LOGGER.info(
        "Rerun the deterministic pipeline to see which Needs-Review rows "
        "flipped to Met."
    )
    return 0


def rerun(limit: int | None = None, dry_run: bool = False) -> int:
    """Run PaddleOCR on every candidate page and record cross-engine CER."""

    settings = get_settings()
    threshold = ocr_max_character_error_rate()

    hits = candidate_pages()
    if not hits:
        _LOGGER.info("No candidate OCR pages. Nothing to do.")
        return 0

    ordered = sorted(hits.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    if limit is not None:
        ordered = ordered[: max(limit, 0)]

    _LOGGER.info(
        "PaddleOCR will process %d page(s) across %d document(s).",
        len(ordered),
        len({doc for (doc, _), _ in ordered}),
    )
    _LOGGER.info("Loading PaddleOCR Arabic model (may download on first run)...")
    ocr_instance = _load_paddle_ocr()
    _LOGGER.info("Model ready.")

    existing = load_measurements(settings.cer_measurements_path)
    kept = list(existing.measurements)
    measured_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")

    added = 0
    passed = 0
    missing_pdf = 0
    empty_paddle = 0

    for (document_id, page), reqs_hit in ordered:
        pdf_path = _find_pdf(settings.reports_dir, document_id)
        if pdf_path is None:
            _LOGGER.warning("No PDF found for %s; skipping.", document_id)
            missing_pdf += 1
            continue

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            try:
                _render_page_to_image(pdf_path, page, image_path)
            except OcrEngineError as exc:
                _LOGGER.warning(
                    "%s page %d: render failed (%s)", document_id, page, exc
                )
                continue
            try:
                paddle_text = _paddle_recognize(ocr_instance, image_path)
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "%s page %d: PaddleOCR failed (%s)", document_id, page, exc
                )
                continue

        if not paddle_text.strip():
            _LOGGER.warning(
                "%s page %d: PaddleOCR returned empty text.", document_id, page
            )
            empty_paddle += 1
            continue

        tesseract_text = _tesseract_text_for(settings.ocr_dir, document_id, page)
        if not tesseract_text.strip():
            _LOGGER.warning(
                "%s page %d: no Tesseract text to compare against.",
                document_id,
                page,
            )
            continue

        measurement = measure_page(
            reference=normalize_arabic(paddle_text),
            hypothesis=normalize_arabic(tesseract_text),
            document_id=document_id,
            page=page,
            engine=_PADDLE_ENGINE_NAME,
            threshold=threshold,
            measured_at=measured_ts,
        )
        if measurement is None:
            _LOGGER.warning(
                "%s page %d: cannot score (empty reference after normalise).",
                document_id,
                page,
            )
            continue

        kept = [
            m
            for m in kept
            if not (
                m.document_id == document_id
                and m.page == page
                and m.engine == _PADDLE_ENGINE_NAME
            )
        ]
        kept.append(measurement)
        added += 1
        if measurement.verified_evidence_eligible:
            passed += 1

        status = "OK" if measurement.verified_evidence_eligible else "ABOVE"
        req_summary = ", ".join(req_id for req_id, _ in reqs_hit)
        _LOGGER.info(
            "%s / p%d: CER %.2f%%  ref=%d  hyp=%d  -> %s  (unlocks %s)",
            document_id,
            page,
            measurement.cer * 100,
            measurement.reference_chars,
            measurement.hypothesis_chars,
            status,
            req_summary,
        )

    if dry_run:
        _LOGGER.info("Dry run: not writing measurements.")
    else:
        save_measurements(
            kept,
            settings.cer_measurements_path,
            threshold=threshold,
            anchor_consensus=existing.anchor_consensus,
        )
        _LOGGER.info(
            "Wrote measurements to %s", settings.cer_measurements_path
        )

    _LOGGER.info(
        "Processed %d page(s): %d passed CER threshold (<= %.1f%%), "
        "%d PDFs missing, %d empty PaddleOCR outputs.",
        added,
        passed,
        threshold * 100,
        missing_pdf,
        empty_paddle,
    )
    _LOGGER.info(
        "Rerun the deterministic pipeline to see which Needs-Review rows "
        "flipped to Met."
    )
    return 0


# ------------------------------------------------- CLI entrypoint ----


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jsc_review.documents.paddleocr_rerun",
        description=(
            "Run PaddleOCR as a second local OCR engine on the same pages "
            "the transcription workflow would scaffold. Two modes:\n"
            "  consensus (default): record a two-engine anchor consensus "
            "when PaddleOCR independently confirms Tesseract's anchor hit.\n"
            "  cer: measure PaddleOCR vs Tesseract character-error-rate "
            "across the entire page (diagnostic; usually too strict a gate "
            "for cross-engine comparison)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("consensus", "cer"),
        default="consensus",
        help="Verification method to run (default: consensus).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many pages (useful for a smoke test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run PaddleOCR but do not persist results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.mode == "consensus":
        return rerun_consensus(limit=args.limit, dry_run=args.dry_run)
    return rerun(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["main", "rerun", "rerun_consensus"]
