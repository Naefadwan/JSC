"""Remove every trace of one document from the local stores.

Deleting a document is destructive and crosses several stores on disk: the
PDF, the extracted JSON, the OCR JSON, the chunk mirror, the shared findings
file, the reviewer decisions file, the CER measurement file, and the two
global indexes (BM25 and dense).

The orchestrator calls each store's own ``remove_*`` helper and records what
happened to each one in a :class:`DeletionSummary`. Every per-store step is
wrapped in its own try/except so one broken store cannot block the next.
The caller (an HTTP route or a CLI) decides whether a partial removal is
acceptable; this module reports facts rather than policy.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from jsc_review.documents.cer_measurement import (
    load_measurements,
    remove_document_measurements,
    save_measurements,
)
from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval import dense as dense_module
from jsc_review.reviewer.store import remove_decisions_for_document

_LOGGER = logging.getLogger(__name__)

# Serializes global-index rewrites so a concurrent upload cannot read the
# BM25 or dense index mid-delete and clobber the result on save.
_index_lock = threading.Lock()


@dataclass(frozen=True)
class StoreOutcome:
    """One store's removal result."""

    name: str
    ok: bool
    removed: int = 0
    detail: str = ""


@dataclass(frozen=True)
class DeletionSummary:
    """What the orchestrator did to each store."""

    document_id: str
    outcomes: tuple[StoreOutcome, ...] = field(default_factory=tuple)

    @property
    def removed_anything(self) -> bool:
        """True when at least one store reported a positive removal count."""

        return any(outcome.removed > 0 for outcome in self.outcomes)

    @property
    def all_ok(self) -> bool:
        """True when every store succeeded (regardless of whether it had data)."""

        return all(outcome.ok for outcome in self.outcomes)

    def outcome(self, name: str) -> StoreOutcome | None:
        """Return the first outcome with the given name, if any."""

        for outcome in self.outcomes:
            if outcome.name == name:
                return outcome
        return None


def _run_step(name: str, action: object) -> StoreOutcome:
    """Run one store removal and never raise. ``action`` must be a zero-arg callable.

    When ``action`` returns an ``int``, the value is recorded as ``removed``.
    Any other return value is treated as a plain "ok, nothing to count"
    outcome.
    """

    try:
        result = action()  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001  (orchestrator must stay alive)
        _LOGGER.exception("remove step %s failed", name)
        return StoreOutcome(name=name, ok=False, detail=str(exc))
    if isinstance(result, int):
        return StoreOutcome(name=name, ok=True, removed=result)
    return StoreOutcome(name=name, ok=True, detail=str(result or ""))


def remove_pdf(reports_dir: Path | None, document_id: str) -> int:
    """Unlink the PDF for ``document_id`` under ``reports_dir``. Idempotent."""

    if reports_dir is None:
        return 0
    target = reports_dir / f"{document_id}.pdf"
    if not target.exists():
        return 0
    target.unlink()
    return 1


def remove_single_json(directory: Path | None, document_id: str) -> int:
    """Unlink ``{document_id}.json`` under ``directory``. Idempotent."""

    if directory is None:
        return 0
    target = directory / f"{document_id}.json"
    if not target.exists():
        return 0
    target.unlink()
    return 1


def remove_findings_for_document(findings_path: Path | None, document_id: str) -> int:
    """Rewrite ``findings.json`` without any entry for ``document_id``.

    Returns the number of removed finding rows. Writes atomically so a crash
    cannot leave the shared file partially rewritten.
    """

    if findings_path is None or not findings_path.exists():
        return 0
    try:
        payload = json.loads(findings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, list):
        return 0
    kept = [
        row for row in payload if isinstance(row, dict) and row.get("document_id") != document_id
    ]
    removed = len(payload) - len(kept)
    if removed == 0:
        return 0
    tmp = findings_path.with_suffix(findings_path.suffix + ".tmp")
    tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(findings_path)
    return removed


def remove_from_bm25_index(index_path: Path | None, document_id: str) -> int:
    """Rewrite the global BM25 index without ``document_id``. Idempotent."""

    if index_path is None or not index_path.exists():
        return 0
    with _index_lock:
        index = bm25_module.load_index(index_path)
        rebuilt, removed = bm25_module.remove_document(index, document_id)
        if removed:
            bm25_module.save_index(rebuilt, index_path)
    return removed


def remove_from_dense_index(index_path: Path | None, document_id: str) -> int:
    """Rewrite the global dense index without ``document_id``. Idempotent."""

    if index_path is None or not index_path.exists():
        return 0
    with _index_lock:
        index = dense_module.load_index(index_path)
        rebuilt, removed = dense_module.remove_document(index, document_id)
        if removed:
            dense_module.save_index(rebuilt, index_path)
    return removed


def remove_cer_measurements_for_document(measurements_path: Path | None, document_id: str) -> int:
    """Rewrite the CER measurements file without ``document_id``. Idempotent."""

    if measurements_path is None or not measurements_path.exists():
        return 0
    store = load_measurements(measurements_path)
    rebuilt, removed = remove_document_measurements(store, document_id)
    if removed:
        save_measurements(
            rebuilt.measurements,
            measurements_path,
            threshold=rebuilt.threshold,
            anchor_consensus=rebuilt.anchor_consensus,
        )
    return removed


def _remove_company_link(path: Path | None, document_id: str) -> int:
    """Drop the (document -> company) link for ``document_id`` when present."""

    if path is None:
        return 0
    from jsc_review.companies.mapping import clear_link

    return 1 if clear_link(path, document_id=document_id) else 0


def remove_document(
    document_id: str,
    *,
    reports_dir: Path | None = None,
    extracted_dir: Path | None = None,
    ocr_dir: Path | None = None,
    chunks_with_ocr_dir: Path | None = None,
    findings_path: Path | None = None,
    decisions_path: Path | None = None,
    cer_measurements_path: Path | None = None,
    bm25_index_path: Path | None = None,
    dense_index_path: Path | None = None,
    document_companies_path: Path | None = None,
) -> DeletionSummary:
    """Remove every trace of ``document_id`` from each supplied local store.

    A store whose path is ``None`` is skipped silently. Each enabled store
    runs in its own try/except so one failure cannot stop the next; the
    caller inspects :attr:`DeletionSummary.all_ok` to decide whether to
    report success to the user.
    """

    outcomes: list[StoreOutcome] = [
        _run_step("pdf", lambda: remove_pdf(reports_dir, document_id)),
        _run_step("extracted", lambda: remove_single_json(extracted_dir, document_id)),
        _run_step("ocr", lambda: remove_single_json(ocr_dir, document_id)),
        _run_step("chunks_with_ocr", lambda: remove_single_json(chunks_with_ocr_dir, document_id)),
        _run_step("findings", lambda: remove_findings_for_document(findings_path, document_id)),
        _run_step(
            "decisions",
            lambda: (
                remove_decisions_for_document(decisions_path, document_id)
                if decisions_path is not None
                else 0
            ),
        ),
        _run_step(
            "cer_measurements",
            lambda: remove_cer_measurements_for_document(cer_measurements_path, document_id),
        ),
        _run_step("bm25_index", lambda: remove_from_bm25_index(bm25_index_path, document_id)),
        _run_step("dense_index", lambda: remove_from_dense_index(dense_index_path, document_id)),
        _run_step(
            "company_link",
            lambda: _remove_company_link(document_companies_path, document_id),
        ),
    ]
    return DeletionSummary(document_id=document_id, outcomes=tuple(outcomes))
