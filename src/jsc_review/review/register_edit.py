"""Read and update the requirements register CSV from the reviewer screen.

The project is run against different documents, companies, and purposes.
Which checklist requirements apply changes per project, so a reviewer
toggles the ``poc_selected`` column from the UI instead of hand-editing
the CSV. Edits are limited to that one column — the other columns describe
the requirement itself and must stay aligned with the source checklist.

The CSV writer preserves row order and the full column schema so a round
trip through this module is a no-op for every column except the toggled
``poc_selected``. Writes go through a ``.tmp`` file and ``replace`` so a
crash mid-write cannot leave the register partially rewritten.
"""

from __future__ import annotations

import csv
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Serializes concurrent writes to the register so two reviewers hitting the
# UI at once cannot clobber each other's edits.
_register_lock = threading.Lock()

# The canonical column schema from the source checklist. Rewrites use this
# order so the file stays diffable against the original.
REGISTER_COLUMNS: tuple[str, ...] = (
    "requirement_id",
    "checklist",
    "section",
    "requirement_text_ar",
    "requirement_text_en",
    "requirement_type",
    "evaluation_method",
    "poc_selected",
    "owner",
    "notes",
)

_TRUE_VALUES = frozenset({"yes", "true", "1", "y", "t"})


class RegisterRow(BaseModel):
    """One row of the requirements register, exposed to the UI."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    checklist: str = ""
    section: str = ""
    requirement_text_ar: str = ""
    requirement_text_en: str = ""
    requirement_type: str = ""
    evaluation_method: str = ""
    poc_selected: bool = False
    owner: str = ""
    notes: str = ""


class SelectionUpdate(BaseModel):
    """A reviewer-provided set of selection toggles keyed by requirement id."""

    model_config = ConfigDict(extra="forbid")

    selections: dict[str, bool] = Field(default_factory=dict)


def _is_selected(raw: str | None) -> bool:
    """Normalize the CSV string value for ``poc_selected`` into a bool."""

    if raw is None:
        return False
    return raw.strip().lower() in _TRUE_VALUES


def _selection_to_cell(value: bool) -> str:
    """Serialize a bool back to the convention used in the source checklist."""

    return "yes" if value else ""


def load_register(path: Path) -> list[RegisterRow]:
    """Return every row in the register, with ``poc_selected`` as a ``bool``."""

    if not path.exists():
        return []
    rows: list[RegisterRow] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            requirement_id = str(raw.get("requirement_id") or "").strip()
            if not requirement_id:
                continue
            rows.append(
                RegisterRow(
                    requirement_id=requirement_id,
                    checklist=str(raw.get("checklist") or ""),
                    section=str(raw.get("section") or ""),
                    requirement_text_ar=str(raw.get("requirement_text_ar") or ""),
                    requirement_text_en=str(raw.get("requirement_text_en") or ""),
                    requirement_type=str(raw.get("requirement_type") or ""),
                    evaluation_method=str(raw.get("evaluation_method") or ""),
                    poc_selected=_is_selected(raw.get("poc_selected")),
                    owner=str(raw.get("owner") or ""),
                    notes=str(raw.get("notes") or ""),
                )
            )
    return rows


def save_register(path: Path, rows: Sequence[RegisterRow]) -> Path:
    """Rewrite the register atomically with the canonical column schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REGISTER_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "requirement_id": row.requirement_id,
                    "checklist": row.checklist,
                    "section": row.section,
                    "requirement_text_ar": row.requirement_text_ar,
                    "requirement_text_en": row.requirement_text_en,
                    "requirement_type": row.requirement_type,
                    "evaluation_method": row.evaluation_method,
                    "poc_selected": _selection_to_cell(row.poc_selected),
                    "owner": row.owner,
                    "notes": row.notes,
                }
            )
    tmp.replace(path)
    return path


def apply_selection_updates(
    rows: Iterable[RegisterRow], selections: Mapping[str, bool]
) -> tuple[list[RegisterRow], list[str]]:
    """Return a new row list with ``poc_selected`` flipped per ``selections``.

    The second element of the tuple lists any requirement ids from
    ``selections`` that do not appear in ``rows`` — the caller surfaces them
    as warnings so a stale UI (e.g. after the register was manually trimmed)
    does not silently drop edits.
    """

    by_id = {row.requirement_id: row for row in rows}
    updated: list[RegisterRow] = []
    for row in by_id.values():
        if row.requirement_id in selections:
            updated.append(row.model_copy(update={"poc_selected": selections[row.requirement_id]}))
        else:
            updated.append(row)
    missing = [requirement_id for requirement_id in selections if requirement_id not in by_id]
    return updated, missing


def update_register_selections(
    path: Path, selections: Mapping[str, bool]
) -> tuple[list[RegisterRow], list[str]]:
    """Load, apply, save in one lock-serialized operation.

    Returns the fresh in-memory row list and the list of unknown requirement
    ids (empty on a clean call). A no-op selection map is still written so
    the caller can trust the file timestamp as the last-edited stamp.
    """

    with _register_lock:
        current = load_register(path)
        updated, missing = apply_selection_updates(current, selections)
        save_register(path, updated)
    return updated, missing
