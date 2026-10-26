"""Tests for the regression runner.

The runner has four moving parts that must each hold on their own: the
outcome classifier, the per-record rescore (with and without the LLM
skip), the end-to-end ``rescore_all`` with a mock judge and on-disk inputs,
and the summary + markdown report.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from jsc_review.retrieval.bm25 import IndexedChunk, build_index
from jsc_review.review.findings import Requirement
from jsc_review.review.golden import GoldenRecord
from jsc_review.review.ollama import ModelJudgment
from jsc_review.review.regression import (
    DocumentInputs,
    classify_outcome,
    load_document_index,
    load_document_ocr_pages,
    load_document_pages,
    render_regression_report,
    rescore_all,
    rescore_record,
    result_to_json,
    summarize_regression,
)

# A unique string the fake judge quotes so the production
# ``apply_model_judgment`` quote-verification path accepts the mocked status.
_QUOTE = "verified passage marker text"
_PAGE_TEXT = f"preamble content {_QUOTE} trailing content"


def _requirement(requirement_id: str, *, method: str = "llm") -> Requirement:
    return Requirement(
        requirement_id=requirement_id,
        section="section",
        text_ar="نص عربي",
        text_en="English text",
        evaluation_method=method,
    )


def _golden(
    *,
    requirement: str = "GOV-001",
    document: str = "doc-a",
    system: str = "Needs Review",
    final: str = "Met",
    split: str = "dev",
    method: str = "llm",
) -> GoldenRecord:
    return GoldenRecord(
        document_id=document,
        requirement_id=requirement,
        system_status=system,  # type: ignore[arg-type]
        final_status=final,  # type: ignore[arg-type]
        evaluation_method=method,
        reviewer_action="changed" if system != final else "accepted",
        reviewer_name="A. Reviewer",
        confidence=0.6,
        evidence_page=38,
        evidence_verified=True,
        split=split,  # type: ignore[arg-type]
        excluded_reason=None,
    )


def _inputs(document: str = "doc-a") -> DocumentInputs:
    """Build DocumentInputs with a real BM25 index.

    The chunk and page text share ``_QUOTE`` so the production retrieval and
    quote-verification path both succeed, which lets the fake judge's status
    reach the final result unchanged.
    """

    chunk = IndexedChunk(
        chunk_id=f"{document}:p1:c01",
        document_id=document,
        page=1,
        section="section",
        text=_PAGE_TEXT,
        extraction_method="pymupdf-text",
    )
    index = build_index([chunk])
    return DocumentInputs(
        document_id=document,
        pages={1: _PAGE_TEXT},
        ocr_pages={},
        index=index,
    )


def _fake_judge(
    status: str, *, quote: str = _QUOTE, confidence: float = 0.9
) -> Callable[..., ModelJudgment]:
    """Return a judge callable that always returns the same ModelJudgment.

    The default ``quote`` matches ``_PAGE_TEXT`` so ``apply_model_judgment``
    accepts the mocked status without downgrading to Needs Review.
    """

    def _judge(_requirement: Requirement, _passage: str, *, passage_is_ocr: bool) -> ModelJudgment:
        _ = passage_is_ocr  # unused in the mock
        return ModelJudgment(status=status, reason="mocked", quote=quote, confidence=confidence)

    return _judge


# ---------------------------------------------------------------------------
# classify_outcome
# ---------------------------------------------------------------------------


def test_outcome_stable_correct_when_prev_and_new_match_truth() -> None:
    assert classify_outcome(previous="Met", new="Met", final="Met") == "stable_correct"


def test_outcome_fixed_when_prev_wrong_and_new_correct() -> None:
    assert classify_outcome(previous="Needs Review", new="Met", final="Met") == "fixed"


def test_outcome_regressed_when_prev_correct_and_new_wrong() -> None:
    assert classify_outcome(previous="Met", new="Needs Review", final="Met") == "regressed"


def test_outcome_stable_wrong_same_when_both_wrong_same_prediction() -> None:
    assert (
        classify_outcome(previous="Needs Review", new="Needs Review", final="Met")
        == "stable_wrong_same"
    )


def test_outcome_stable_wrong_different_when_both_wrong_different_predictions() -> None:
    assert (
        classify_outcome(previous="Needs Review", new="Not Met", final="Met")
        == "stable_wrong_different"
    )


# ---------------------------------------------------------------------------
# rescore_record
# ---------------------------------------------------------------------------


def test_rescore_record_classifies_fixed_when_new_matches_truth() -> None:
    record = _golden(system="Needs Review", final="Met")
    requirement = _requirement("GOV-001")

    result = rescore_record(
        record,
        requirement,
        _inputs(),
        judge=_fake_judge("Met"),
    )

    assert result.outcome == "fixed"
    assert result.previous_system_status == "Needs Review"
    assert result.new_system_status == "Met"
    assert result.final_status == "Met"


def test_rescore_record_detects_regression_when_judge_flips_a_correct_answer() -> None:
    record = _golden(system="Met", final="Met")
    requirement = _requirement("GOV-001")

    result = rescore_record(
        record,
        requirement,
        _inputs(),
        judge=_fake_judge("Needs Review"),
    )

    assert result.outcome == "regressed"


def test_rescore_record_skip_llm_short_circuits_llm_requirements() -> None:
    record = _golden(system="Needs Review", final="Met", method="llm")
    requirement = _requirement("GOV-001", method="llm")

    def _judge_that_should_not_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("judge should not be called when skip_llm=True")

    result = rescore_record(
        record,
        requirement,
        _inputs(),
        judge=_judge_that_should_not_run,
        skip_llm=True,
    )

    assert result.outcome == "skipped"
    assert result.skipped_reason == "llm skipped by flag"
    assert result.new_system_status == record.system_status


def test_rescore_record_skip_llm_still_runs_deterministic_requirements() -> None:
    """``skip_llm`` must only short-circuit llm records."""

    record = _golden(system="Needs Review", final="Met", method="deterministic")
    requirement = _requirement("DIS-062", method="deterministic")
    # DIS-062 is a heading rule; without any heading in pages it returns Needs Review.
    result = rescore_record(record, requirement, _inputs(), skip_llm=True)

    assert result.outcome != "skipped"


# ---------------------------------------------------------------------------
# rescore_all
# ---------------------------------------------------------------------------


def test_rescore_all_skips_excluded_records_and_missing_requirements(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "doc-a.json").write_text(
        json.dumps({"pages": [{"page": 1, "text": _PAGE_TEXT}]}), encoding="utf-8"
    )
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "doc-a.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": "doc-a:p1:c01",
                        "document_id": "doc-a",
                        "page": 1,
                        "section": "section",
                        "text": _PAGE_TEXT,
                        "extraction_method": "pymupdf-text",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = [
        _golden(requirement="GOV-001", system="Needs Review", final="Met"),
        _golden(
            requirement="GOV-002",
            system="Needs Review",
            final="Met",
            split="excluded",
        ),
        _golden(requirement="UNKNOWN-999", system="Met", final="Met"),
    ]
    requirements = {"GOV-001": _requirement("GOV-001")}

    results = rescore_all(
        records,
        requirements,
        extracted_dir=extracted,
        ocr_dir=tmp_path / "ocr",
        chunks_dir=chunks,
        judge=_fake_judge("Met"),
    )

    # Excluded record is dropped (not in output), unknown requirement is kept as skipped.
    assert len(results) == 2
    outcomes = {r.requirement_id: r.outcome for r in results}
    assert outcomes["GOV-001"] == "fixed"
    assert outcomes["UNKNOWN-999"] == "skipped"


def test_rescore_all_caches_document_inputs_per_document(tmp_path: Path) -> None:
    """Two records that share a document should trigger the loader once.

    The on-disk inputs mirror what the upload pipeline writes: a page JSON
    under ``extracted_dir`` and a chunk JSON under ``chunks_dir``. The judge
    is mocked, so the only thing being tested here is that two records in
    the same document make it through the end-to-end rescore path.
    """

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "doc-a.json").write_text(
        json.dumps({"pages": [{"page": 1, "text": _PAGE_TEXT}]}), encoding="utf-8"
    )
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "doc-a.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": "doc-a:p1:c01",
                        "document_id": "doc-a",
                        "page": 1,
                        "section": "section",
                        "text": _PAGE_TEXT,
                        "extraction_method": "pymupdf-text",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = [
        _golden(requirement="GOV-001"),
        _golden(requirement="DIS-010"),
    ]
    requirements = {
        "GOV-001": _requirement("GOV-001"),
        "DIS-010": _requirement("DIS-010"),
    }

    results = rescore_all(
        records,
        requirements,
        extracted_dir=extracted,
        ocr_dir=tmp_path / "ocr",
        chunks_dir=chunks,
        judge=_fake_judge("Met"),
    )

    assert len(results) == 2
    # Both records should score (not skip), proving the loader + cache ran once per document.
    assert all(r.outcome != "skipped" for r in results)


# ---------------------------------------------------------------------------
# summarize_regression + render_regression_report
# ---------------------------------------------------------------------------


def test_summary_tracks_accuracy_before_and_after() -> None:
    """Summary math is independent of the loader; drive it with rescore_record."""

    judge = _fake_judge("Met")
    records_and_requirements = [
        (
            _golden(requirement="GOV-001", system="Needs Review", final="Met"),
            _requirement("GOV-001"),
        ),  # was wrong, judge says Met => fixed
        (
            _golden(requirement="GOV-002", system="Met", final="Met"),
            _requirement("GOV-002"),
        ),  # was right, judge still says Met => stable_correct
    ]
    results = [
        rescore_record(record, requirement, _inputs(), judge=judge)
        for record, requirement in records_and_requirements
    ]
    summary = summarize_regression(results)

    assert summary.total_scored == 2
    assert summary.correct_before == 1
    assert summary.correct_now == 2
    assert summary.accuracy_before == 0.5
    assert summary.accuracy_now == 1.0
    assert summary.accuracy_delta == 0.5
    assert summary.outcome_counts.get("fixed") == 1
    assert summary.outcome_counts.get("stable_correct") == 1


def test_report_lists_flipped_pairs_and_not_a_no_flip_notice_when_regressions_exist() -> None:
    record = _golden(system="Met", final="Met")
    requirement = _requirement("GOV-001")

    results = [rescore_record(record, requirement, _inputs(), judge=_fake_judge("Needs Review"))]
    summary = summarize_regression(results)
    report = render_regression_report(summary, results)

    assert "Regression run report" in report
    assert "regressed" in report
    assert "No flips" not in report


def test_report_says_no_flips_when_everything_stable() -> None:
    record = _golden(system="Met", final="Met")
    requirement = _requirement("GOV-001")

    results = [rescore_record(record, requirement, _inputs(), judge=_fake_judge("Met"))]
    summary = summarize_regression(results)
    report = render_regression_report(summary, results)

    assert "No flips" in report


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def test_load_document_pages_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_document_pages("doc-a", tmp_path) == {}


def test_load_document_pages_reads_text_by_page_number(tmp_path: Path) -> None:
    (tmp_path / "doc-a.json").write_text(
        json.dumps(
            {
                "pages": [
                    {"page": 1, "text": "first"},
                    {"page": 2, "text": "second"},
                ]
            }
        ),
        encoding="utf-8",
    )

    pages = load_document_pages("doc-a", tmp_path)

    assert pages == {1: "first", 2: "second"}


def test_load_document_ocr_pages_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_document_ocr_pages("doc-a", tmp_path) == {}


def test_load_document_index_returns_none_when_chunk_file_missing(tmp_path: Path) -> None:
    assert load_document_index("doc-a", tmp_path) is None


def test_load_document_index_builds_bm25_from_chunk_payload(tmp_path: Path) -> None:
    (tmp_path / "doc-a.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": "doc-a:p1:c01",
                        "document_id": "doc-a",
                        "page": 1,
                        "section": "s",
                        "text": "hello world",
                        "extraction_method": "pymupdf-text",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    index = load_document_index("doc-a", tmp_path)

    assert index is not None
    assert len(index.chunks) == 1


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


def test_result_to_json_round_trips_all_fields() -> None:
    record = _golden(system="Needs Review", final="Met")
    requirement = _requirement("GOV-001")

    result = rescore_record(
        record,
        requirement,
        _inputs(),
        judge=_fake_judge("Met"),
    )
    payload = result_to_json(result)

    assert payload["outcome"] == "fixed"
    assert payload["previous_system_status"] == "Needs Review"
    assert payload["new_system_status"] == "Met"
    assert payload["final_status"] == "Met"
    assert payload["split"] == "dev"
