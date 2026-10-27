"""Atomic read-modify-write for the golden-label store."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from jsc_review.golden.models import GoldenCandidate, GoldenSet, ReviewerLabel

# Serializes writes from the API routes so two concurrent labelings cannot
# corrupt the JSON store. The store itself is a plain file; this lock covers
# the whole load-modify-save cycle.
_STORE_LOCK = threading.Lock()


def load_golden_set(path: Path) -> GoldenSet:
    """Return the on-disk golden set, or an empty set when the file is absent."""

    if not path.exists():
        return GoldenSet()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return GoldenSet()
    return GoldenSet.model_validate_json(text)


def save_golden_set(path: Path, golden_set: GoldenSet) -> None:
    """Write the golden set atomically via a temp file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(golden_set.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def add_candidates(path: Path, new_candidates: list[GoldenCandidate]) -> GoldenSet:
    """Append any candidates whose (document, requirement) key is new."""

    with _STORE_LOCK:
        current = load_golden_set(path)
        existing_keys = {cand.key for cand in current.candidates}
        for candidate in new_candidates:
            if candidate.key in existing_keys:
                continue
            current.candidates.append(candidate)
            existing_keys.add(candidate.key)
        save_golden_set(path, current)
        return current


def upsert_label(
    path: Path,
    *,
    document_id: str,
    requirement_id: str,
    label: ReviewerLabel,
) -> GoldenCandidate:
    """Record one reviewer's label; replaces that reviewer's prior label if any."""

    with _STORE_LOCK:
        current = load_golden_set(path)
        candidate = current.find(document_id, requirement_id)
        if candidate is None:
            raise KeyError(
                f"Unknown candidate: {document_id} / {requirement_id}. "
                "Sample it into the golden set first."
            )
        candidate.labels = [
            existing
            for existing in candidate.labels
            if existing.reviewer_name != label.reviewer_name
        ]
        candidate.labels.append(label)
        save_golden_set(path, current)
        return candidate


def pin_ground_truth(
    path: Path,
    *,
    document_id: str,
    requirement_id: str,
    page: int | None,
    text: str,
    pinned_by: str,
) -> GoldenCandidate:
    """Record the reviewer-chosen evidence for one candidate.

    Pass ``page=None`` and ``text=""`` to clear a prior pin (the UI uses
    this when a reviewer decides there is no evidence after all).
    """

    with _STORE_LOCK:
        current = load_golden_set(path)
        candidate = current.find(document_id, requirement_id)
        if candidate is None:
            raise KeyError(
                f"Unknown candidate: {document_id} / {requirement_id}."
            )
        candidate.ground_truth_page = page
        candidate.ground_truth_text = text
        candidate.ground_truth_pinned_by = pinned_by.strip() if page else ""
        candidate.ground_truth_pinned_at = datetime.now(UTC) if page else None
        save_golden_set(path, current)
        return candidate


def delete_candidate(
    path: Path, *, document_id: str, requirement_id: str
) -> bool:
    """Remove one candidate from the golden set; returns True on hit."""

    with _STORE_LOCK:
        current = load_golden_set(path)
        before = len(current.candidates)
        current.candidates = [
            cand
            for cand in current.candidates
            if cand.key != (document_id, requirement_id)
        ]
        if len(current.candidates) == before:
            return False
        save_golden_set(path, current)
        return True
