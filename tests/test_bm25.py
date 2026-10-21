"""Tests for the local multilingual BM25 index."""

from pathlib import Path

from jsc_review.retrieval.bm25 import (
    IndexedChunk,
    build_index,
    load_index,
    save_index,
    tokenize,
)


def _chunk(chunk_id: str, text: str, page: int = 1) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk_id,
        document_id="report",
        page=page,
        section="Board",
        text=text,
        extraction_method="pymupdf-text",
    )


def test_tokenize_keeps_arabic_and_english_words() -> None:
    assert tokenize("Risk سياسة management") == ("risk", "سياسة", "management")


def test_search_ranks_the_matching_passage_first() -> None:
    index = build_index(
        [
            _chunk("report:p1:c01", "The board adopted a risk management policy."),
            _chunk("report:p2:c01", "Number of employees in each geographic location."),
            _chunk("report:p3:c01", "اعتماد سياسة إدارة المخاطر التي قد تتعرض لها الشركة."),
        ]
    )

    hits = index.search("risk management policy", k=2)

    assert len(hits) == 1
    assert hits[0].chunk_id == "report:p1:c01"
    assert hits[0].page == 1


def test_arabic_query_finds_the_arabic_passage() -> None:
    index = build_index(
        [
            _chunk("report:p1:c01", "The cash flow statement is presented in the financials."),
            _chunk("report:p4:c01", "اعتماد سياسة إدارة المخاطر التي قد تتعرض لها الشركة."),
        ]
    )

    hits = index.search("سياسة إدارة المخاطر", k=1)

    assert hits[0].chunk_id == "report:p4:c01"


def test_search_can_stay_inside_one_document() -> None:
    index = build_index(
        [
            IndexedChunk(
                chunk_id="other:p1:c01",
                document_id="other",
                page=1,
                section="Board",
                text="The board adopted a risk management policy.",
                extraction_method="pymupdf-text",
            ),
            _chunk("report:p2:c01", "The board adopted a risk management policy.", page=2),
        ]
    )

    hits = index.search("risk management policy", k=3, document_id="report")

    assert [hit.chunk_id for hit in hits] == ["report:p2:c01"]


def test_empty_query_returns_nothing() -> None:
    index = build_index([_chunk("report:p1:c01", "A board policy.")])

    assert index.search("   ") == ()


def test_index_round_trip(tmp_path: Path) -> None:
    original = build_index([_chunk("report:p8:ocr", "Recovered audit committee text.", page=8)])
    path = save_index(original, tmp_path / "index.json")

    loaded = load_index(path)
    hits = loaded.search("audit committee", k=1)

    assert hits[0].chunk_id == "report:p8:ocr"
    assert hits[0].extraction_method == "pymupdf-text"
    assert hits[0].page == 8
