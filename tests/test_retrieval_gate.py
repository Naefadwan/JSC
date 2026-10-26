"""Tests for the Phase 2 retrieval-gate helpers."""

from __future__ import annotations

import json
from pathlib import Path

from jsc_review.retrieval.bm25 import Bm25Index, IndexedChunk, build_index
from jsc_review.retrieval.retrieval_gate import (
    RetrievalLabel,
    collect_labels_from_findings,
    gate_passes,
    measure_label,
    measure_retrieval,
    summarize_measurements,
)
from jsc_review.review.findings import Requirement

_AR_OWNERS = "كبار المساهمين"
_AR_SECTION = "بند"


def _requirement(requirement_id: str, text_ar: str, text_en: str = "") -> Requirement:
    return Requirement(
        requirement_id=requirement_id,
        section=_AR_SECTION,
        text_ar=text_ar,
        text_en=text_en,
        evaluation_method="llm",
    )


def _chunk(
    doc: str, page: int, suffix: str, text: str, method: str = "pymupdf-text"
) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=f"{doc}:p{page}:{suffix}",
        document_id=doc,
        page=page,
        section="",
        text=text,
        extraction_method=method,
    )


def _write_findings(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# --- labels -----------------------------------------------------------------


def test_collect_labels_pulls_verified_evidence_pages(tmp_path: Path) -> None:
    findings = tmp_path / "findings.json"
    _write_findings(
        findings,
        [
            {
                "document_id": "doc_a",
                "result": {
                    "requirement_id": "DIS-010",
                    "status": "Met",
                    "evidence": {"page": 44, "verified": True},
                },
            },
            {
                "document_id": "doc_a",
                "result": {
                    "requirement_id": "DIS-017",
                    "status": "Needs Review",
                    "evidence": {"page": 20, "verified": True},
                },
            },
        ],
    )

    labels = collect_labels_from_findings(findings)

    assert len(labels) == 2
    by_req = {label.requirement_id: label for label in labels}
    assert by_req["DIS-010"].correct_pages == (44,)
    assert by_req["DIS-017"].correct_pages == (20,)


def test_collect_labels_skips_unverified_evidence(tmp_path: Path) -> None:
    findings = tmp_path / "findings.json"
    _write_findings(
        findings,
        [
            {
                "document_id": "doc_a",
                "result": {
                    "requirement_id": "DIS-026",
                    "status": "Needs Review",
                    "evidence": {"page": 11, "verified": False},
                },
            }
        ],
    )

    assert collect_labels_from_findings(findings) == []


def test_collect_labels_merges_duplicate_pairs(tmp_path: Path) -> None:
    findings = tmp_path / "findings.json"
    _write_findings(
        findings,
        [
            {
                "document_id": "doc_a",
                "result": {
                    "requirement_id": "DIS-010",
                    "status": "Met",
                    "evidence": {"page": 44, "verified": True},
                },
            },
            {
                "document_id": "doc_a",
                "result": {
                    "requirement_id": "DIS-010",
                    "status": "Met",
                    "evidence": {"page": 45, "verified": True},
                },
            },
        ],
    )

    labels = collect_labels_from_findings(findings)

    assert len(labels) == 1
    assert labels[0].correct_pages == (44, 45)


def test_collect_labels_on_missing_file_returns_empty(tmp_path: Path) -> None:
    assert collect_labels_from_findings(tmp_path / "nope.json") == []


# --- measurement ------------------------------------------------------------


def _tiny_bm25() -> Bm25Index:
    return build_index(
        [
            _chunk("doc", 1, "c01", "introduction and general description"),
            _chunk("doc", 5, "c01", "major shareholders own more than ten percent"),
            _chunk("doc", 6, "c01", "cash flow statement for the year"),
        ]
    )


def test_measure_label_finds_correct_page_in_top_1() -> None:
    label = RetrievalLabel(
        document_id="doc",
        requirement_id="DIS-010",
        correct_pages=(5,),
        source="finding:met",
    )
    requirement = _requirement("DIS-010", _AR_OWNERS, "major shareholders")

    result = measure_label(label, requirement, bm25=_tiny_bm25(), dense=None)

    assert result.rank_for("bm25") == 1
    assert result.method_results[0].top_pages[0] == 5


def test_measure_label_misses_when_query_is_unrelated() -> None:
    label = RetrievalLabel(
        document_id="doc",
        requirement_id="GOV-999",
        correct_pages=(99,),
        source="finding:met",
    )
    requirement = _requirement("GOV-999", "نص عربي غير مرتبط", "unrelated english text")

    result = measure_label(label, requirement, bm25=_tiny_bm25(), dense=None)

    assert result.rank_for("bm25") is None


def test_measure_retrieval_skips_unknown_requirement_ids() -> None:
    labels = [
        RetrievalLabel(
            document_id="doc",
            requirement_id="DIS-010",
            correct_pages=(5,),
            source="finding:met",
        ),
        RetrievalLabel(
            document_id="doc",
            requirement_id="DIS-999",  # not in requirements dict
            correct_pages=(1,),
            source="finding:met",
        ),
    ]
    requirements = {
        "DIS-010": _requirement("DIS-010", _AR_OWNERS, "major shareholders"),
    }

    results = measure_retrieval(labels, requirements, bm25=_tiny_bm25(), dense=None)

    assert len(results) == 1
    assert results[0].label.requirement_id == "DIS-010"


def test_summarize_counts_hits_at_each_k() -> None:
    labels = [
        RetrievalLabel(
            document_id="doc",
            requirement_id="DIS-010",
            correct_pages=(5,),
            source="finding:met",
        ),
        RetrievalLabel(
            document_id="doc",
            requirement_id="GOV-999",
            correct_pages=(99,),
            source="finding:met",
        ),
    ]
    requirements = {
        "DIS-010": _requirement("DIS-010", _AR_OWNERS, "major shareholders"),
        "GOV-999": _requirement("GOV-999", "unrelated", "unrelated"),
    }

    results = measure_retrieval(labels, requirements, bm25=_tiny_bm25(), dense=None)
    summaries = summarize_measurements(results, ks=(1, 5))

    bm25 = summaries["bm25"]
    assert bm25.total == 2
    assert bm25.hits[1] == 1
    assert bm25.hits[5] == 1
    assert 0.4 < bm25.rate(5) < 0.6


def test_gate_passes_at_or_above_threshold() -> None:
    passing = summarize_measurements(
        [
            # The tiny index returns the correct page at rank 1 for this label.
            measure_label(
                RetrievalLabel(
                    document_id="doc",
                    requirement_id="DIS-010",
                    correct_pages=(5,),
                    source="finding:met",
                ),
                _requirement("DIS-010", _AR_OWNERS, "major shareholders"),
                bm25=_tiny_bm25(),
                dense=None,
            )
        ],
        ks=(1, 5),
    )

    assert gate_passes(passing["bm25"], k=5, threshold=0.80) is True
    assert gate_passes(passing["bm25"], k=5, threshold=1.0) is True
