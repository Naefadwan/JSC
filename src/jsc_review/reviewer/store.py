"""Local JSON store for reviewer decisions.

Writes are atomic. The file stays on the local disk, never sent to a remote
service. The path is configurable through ``JSC_REVIEWER_DECISIONS_PATH``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from jsc_review.reviewer.models import ReviewerDecision

_adapter = TypeAdapter(list[ReviewerDecision])


def load_decisions(path: Path) -> list[ReviewerDecision]:
    """Read every reviewer decision from the local store."""

    if not path.exists():
        return []
    payload = path.read_text(encoding="utf-8")
    if not payload.strip():
        return []
    return _adapter.validate_json(payload)


def save_decision(path: Path, decision: ReviewerDecision) -> list[ReviewerDecision]:
    """Append or replace a reviewer decision keyed by document and requirement.

    When a decision already exists for the same ``(document_id, requirement_id)``
    pair, the new decision replaces it so the file holds the latest reviewer
    answer rather than a history.
    """

    decisions = load_decisions(path)
    key = (decision.document_id, decision.requirement_id)
    decisions = [
        existing
        for existing in decisions
        if (existing.document_id, existing.requirement_id) != key
    ]
    decisions.append(decision)
    _write_atomic(path, decisions)
    return decisions


def _write_atomic(path: Path, decisions: list[ReviewerDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        _adapter.dump_json(decisions, indent=2).decode("utf-8"),
        encoding="utf-8",
    )
    tmp.replace(path)
