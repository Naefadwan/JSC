"""Tests for the document-to-company mapping store."""

from __future__ import annotations

import json
from pathlib import Path

from jsc_review.companies.mapping import (
    DocumentCompanyLink,
    clear_link,
    get_link,
    latest_links,
    load_store,
    set_link,
)


def test_load_store_returns_empty_when_missing(tmp_path: Path) -> None:
    store = load_store(tmp_path / "nope.json")
    assert store.links == []


def test_set_link_persists_atomically(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    link = set_link(path, document_id="doc-1", company_key="JOIB", linked_by="Nayef")

    assert path.exists()
    assert link.document_id == "doc-1"
    assert link.company_key == "JOIB"
    assert link.linked_by == "Nayef"

    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["links"]) == 1
    assert data["links"][0]["company_key"] == "JOIB"


def test_set_link_replaces_previous_link_for_same_doc(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    set_link(path, document_id="doc-1", company_key="JOIB")
    set_link(path, document_id="doc-1", company_key="BOJX")

    store = load_store(path)
    assert len(store.links) == 1
    assert store.links[0].company_key == "BOJX"


def test_set_link_preserves_other_links(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    set_link(path, document_id="doc-1", company_key="JOIB")
    set_link(path, document_id="doc-2", company_key="BOJX")
    set_link(path, document_id="doc-1", company_key="JOIN")

    links = latest_links(path)
    assert links["doc-1"].company_key == "JOIN"
    assert links["doc-2"].company_key == "BOJX"


def test_clear_link_returns_true_only_when_something_changes(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    set_link(path, document_id="doc-1", company_key="JOIB")

    assert clear_link(path, document_id="doc-1") is True
    assert clear_link(path, document_id="doc-1") is False
    assert get_link(path, "doc-1") is None


def test_get_link_returns_none_for_unknown_doc(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    set_link(path, document_id="doc-1", company_key="JOIB")
    assert get_link(path, "nope") is None


def test_malformed_file_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    store = load_store(path)
    assert isinstance(store.links, list)
    assert store.links == []


def test_set_link_strips_whitespace_from_key(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    link = set_link(path, document_id="doc-1", company_key="  JOIB  ")
    assert link.company_key == "JOIB"
    assert isinstance(link, DocumentCompanyLink)
