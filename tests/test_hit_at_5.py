"""Unit tests for the Hit@5 scorer against a tiny in-memory BM25 index."""

from __future__ import annotations

from jsc_review.golden.models import GoldenCandidate, GoldenSet, ReviewerLabel
from jsc_review.metrics.hit_at_5 import compute_hit_at_5, hit_at_5_notes
from jsc_review.retrieval.bm25 import IndexedChunk, build_index
from jsc_review.review.findings import Requirement


def _chunk(doc: str, page: int, text: str) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=f"{doc}-{page}",
        document_id=doc,
        page=page,
        section=f"s{page}",
        text=text,
        extraction_method="embedded",
    )


def _agreed(doc: str, req: str, label: str, page: int | None) -> GoldenCandidate:
    cand = GoldenCandidate(
        document_id=doc, requirement_id=req, ground_truth_page=page
    )
    cand.labels.append(ReviewerLabel(reviewer_name="A", label=label))  # type: ignore[arg-type]
    cand.labels.append(ReviewerLabel(reviewer_name="B", label=label))  # type: ignore[arg-type]
    return cand


def _req(req_id: str, keyword: str) -> Requirement:
    return Requirement(
        requirement_id=req_id,
        section="s",
        text_ar="",
        text_en=keyword,
        evaluation_method="llm",
    )


def test_hit_at_5_counts_pages_in_top_k() -> None:
    bm25 = build_index([
        _chunk("doc1", 1, "major shareholders ownership five percent"),
        _chunk("doc1", 2, "statement of financial position balance sheet"),
        _chunk("doc1", 3, "noise unrelated text about weather"),
    ])
    golden = GoldenSet(candidates=[
        _agreed("doc1", "R1", "Met", page=1),
        _agreed("doc1", "R2", "Met", page=2),
        _agreed("doc1", "R3", "Met", page=99),  # not in index
    ])
    requirements = {
        "R1": _req("R1", "major shareholders ownership"),
        "R2": _req("R2", "statement of financial position"),
        "R3": _req("R3", "nonexistent topic"),
    }
    result = compute_hit_at_5(golden, bm25=bm25, requirements=requirements)
    assert result.scored == 3
    assert result.hits == 2  # R1 and R2 land on their pages; R3 misses
    assert result.ratio is not None
    assert abs(result.ratio - (2 / 3)) < 1e-9


def test_hit_at_5_skips_unpinned_candidates() -> None:
    bm25 = build_index([_chunk("doc1", 1, "anything")])
    golden = GoldenSet(candidates=[
        _agreed("doc1", "R1", "Met", page=None),
        _agreed("doc1", "R2", "Met", page=1),
    ])
    requirements = {"R1": _req("R1", "x"), "R2": _req("R2", "anything")}
    result = compute_hit_at_5(golden, bm25=bm25, requirements=requirements)
    assert result.skipped_no_page == 1
    assert result.scored == 1


def test_hit_at_5_skips_unknown_requirements() -> None:
    bm25 = build_index([_chunk("doc1", 1, "text")])
    golden = GoldenSet(candidates=[
        _agreed("doc1", "RX", "Met", page=1),
    ])
    requirements: dict[str, Requirement] = {}  # register has no RX
    result = compute_hit_at_5(golden, bm25=bm25, requirements=requirements)
    assert result.skipped_unknown_requirement == 1
    assert result.scored == 0
    assert result.ratio is None


def test_hit_at_5_notes_are_human_readable() -> None:
    result = compute_hit_at_5(
        GoldenSet(candidates=[_agreed("d", "R1", "Met", None)]),
        bm25=build_index([_chunk("d", 1, "x")]),
        requirements={"R1": _req("R1", "x")},
    )
    note = hit_at_5_notes(result, total_golden=1)
    assert "0/1" in note
    assert "no pinned page" in note
