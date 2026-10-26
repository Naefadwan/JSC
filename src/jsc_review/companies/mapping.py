"""Persistent mapping from an uploaded document to the company it belongs to.

Each entry is keyed by the sanitized document id (the PDF stem) and holds
the company's registry key (ticker or numeric code), the moment the link
was recorded, and who recorded it. The store is a single JSON file on disk,
rewritten atomically via a ``.tmp`` file.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_mapping_lock = threading.Lock()


class DocumentCompanyLink(BaseModel):
    """One (document_id -> company) link."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    company_key: str
    linked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    linked_by: str = ""


class DocumentCompanyStore(BaseModel):
    """The persistent on-disk envelope for every link."""

    model_config = ConfigDict(extra="forbid")

    links: list[DocumentCompanyLink] = Field(default_factory=list)

    def as_mapping(self) -> dict[str, DocumentCompanyLink]:
        """Return ``{document_id: link}`` keeping the latest link per doc."""

        latest: dict[str, DocumentCompanyLink] = {}
        for entry in self.links:
            previous = latest.get(entry.document_id)
            if previous is None or entry.linked_at >= previous.linked_at:
                latest[entry.document_id] = entry
        return latest


def load_store(path: Path) -> DocumentCompanyStore:
    """Return the stored links or an empty store when the file is missing."""

    if not path.exists():
        return DocumentCompanyStore()
    try:
        return DocumentCompanyStore.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError):
        return DocumentCompanyStore()


def _write_store(path: Path, store: DocumentCompanyStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def _apply_single_update(
    store: DocumentCompanyStore,
    *,
    document_id: str,
    company_key: str | None,
    linked_by: str,
) -> DocumentCompanyStore:
    """Return a new store with ``document_id`` either updated or removed."""

    kept = [link for link in store.links if link.document_id != document_id]
    if company_key is not None and company_key.strip():
        kept.append(
            DocumentCompanyLink(
                document_id=document_id,
                company_key=company_key.strip(),
                linked_by=linked_by,
            )
        )
    return DocumentCompanyStore(links=kept)


def set_link(
    path: Path,
    *,
    document_id: str,
    company_key: str,
    linked_by: str = "",
) -> DocumentCompanyLink:
    """Attach ``document_id`` to ``company_key`` and persist atomically.

    Replaces any previous link for the same document. Returns the fresh link
    so the caller can echo it to the UI.
    """

    with _mapping_lock:
        store = load_store(path)
        updated = _apply_single_update(
            store,
            document_id=document_id,
            company_key=company_key,
            linked_by=linked_by,
        )
        _write_store(path, updated)
    # The link we just appended is always the last one for this document.
    return next(link for link in updated.links if link.document_id == document_id)


def clear_link(path: Path, *, document_id: str) -> bool:
    """Drop the link for ``document_id``. Returns True when something changed."""

    with _mapping_lock:
        store = load_store(path)
        updated = _apply_single_update(
            store, document_id=document_id, company_key=None, linked_by=""
        )
        if len(updated.links) == len(store.links):
            return False
        _write_store(path, updated)
        return True


def get_link(path: Path, document_id: str) -> DocumentCompanyLink | None:
    """Return the latest link for ``document_id`` or ``None``."""

    return load_store(path).as_mapping().get(document_id)


def latest_links(path: Path) -> Mapping[str, DocumentCompanyLink]:
    """Return ``{document_id: link}`` for every document that has a company."""

    return load_store(path).as_mapping()
