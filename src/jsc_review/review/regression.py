"""Regression runner: re-score the system against the golden dataset.

Every code change in retrieval, judging, or threshold logic has a chance of
flipping a reviewer-confirmed answer. This module re-runs the current
production pipeline on each golden record and classifies the outcome. It is
the measurement step after any change that touches the review path.

Design
------
* The runner is pure in-process. It reads the extracted text, OCR text, and
  chunk index that the upload pipeline already wrote to disk; it does not
  re-extract or re-OCR. Reruns are therefore fast enough for a tight loop.
* The LLM call is injected through :data:`JudgeCallable` so tests and the
  ``--skip-llm`` CLI flag can short-circuit it with a deterministic
  stand-in. The default is the real :func:`judge_passage`.
* Outcomes are classified against two reference points: the previously
  stored system status (did a code change flip the answer?) and the
  reviewer's final status (is the current answer right or wrong?).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jsc_review.core.models import ReviewStatus
from jsc_review.documents.cer_measurement import MeasurementStore
from jsc_review.retrieval.bm25 import Bm25Index, IndexedChunk, build_index
from jsc_review.retrieval.reranker import CrossEncoder
from jsc_review.review.findings import JudgeCallable, Requirement, judge_requirement
from jsc_review.review.golden import GoldenRecord

_LOGGER = logging.getLogger(__name__)

RescoreOutcome = Literal[
    "stable_correct",
    "stable_wrong_same",
    "stable_wrong_different",
    "regressed",
    "fixed",
    "skipped",
]


@dataclass(frozen=True)
class DocumentInputs:
    """The three pieces of per-document state the review path needs."""

    document_id: str
    pages: dict[int, str]
    ocr_pages: dict[int, str]
    index: Bm25Index | None


@dataclass(frozen=True)
class RescoreResult:
    """One record after re-scoring."""

    document_id: str
    requirement_id: str
    previous_system_status: ReviewStatus
    new_system_status: ReviewStatus
    final_status: ReviewStatus
    outcome: RescoreOutcome
    split: str
    skipped_reason: str | None = None


# ---------------------------------------------------------------------------
# Loaders: translate on-disk artifacts into DocumentInputs
# ---------------------------------------------------------------------------


def load_document_pages(document_id: str, extracted_dir: Path) -> dict[int, str]:
    """Load the embedded page text for one document.

    Returns an empty dict when the extracted JSON is missing, so the caller
    can distinguish "no document" from "document has no text".
    """

    source = extracted_dir / f"{document_id}.json"
    if not source.exists():
        _LOGGER.warning("regression: extracted file missing for %s at %s", document_id, source)
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    pages: dict[int, str] = {}
    for page in payload.get("pages", []):
        page_number = int(page["page"])
        pages[page_number] = str(page.get("text", ""))
    return pages


def load_document_ocr_pages(document_id: str, ocr_dir: Path) -> dict[int, str]:
    """Load OCR text for one document. Returns ``{}`` if the file is absent."""

    source = ocr_dir / f"{document_id}.json"
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    ocr_pages: dict[int, str] = {}
    for page in payload.get("pages", []):
        page_number = int(page["page"])
        ocr_pages[page_number] = str(page.get("text", ""))
    return ocr_pages


def load_document_index(document_id: str, chunks_dir: Path) -> Bm25Index | None:
    """Build a BM25 index from the per-document chunk file written on upload.

    Returns ``None`` when the chunk file is missing. The pipeline writes one
    file per document to ``chunks_with_ocr_dir`` and that is what we read.
    """

    source = chunks_dir / f"{document_id}.json"
    if not source.exists():
        _LOGGER.warning(
            "regression: chunk file missing for %s at %s; retrieval will be unavailable",
            document_id,
            source,
        )
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    chunks = [
        IndexedChunk(
            chunk_id=str(raw["chunk_id"]),
            document_id=str(raw["document_id"]),
            page=int(raw["page"]),
            section=str(raw["section"]),
            text=str(raw["text"]),
            extraction_method=str(raw["extraction_method"]),
        )
        for raw in payload.get("chunks", [])
    ]
    if not chunks:
        return None
    return build_index(chunks)


def load_document_inputs(
    document_id: str,
    *,
    extracted_dir: Path,
    ocr_dir: Path,
    chunks_dir: Path,
) -> DocumentInputs:
    """Load every per-document input needed to re-score records."""

    return DocumentInputs(
        document_id=document_id,
        pages=load_document_pages(document_id, extracted_dir),
        ocr_pages=load_document_ocr_pages(document_id, ocr_dir),
        index=load_document_index(document_id, chunks_dir),
    )


# ---------------------------------------------------------------------------
# Core rescore + outcome classification
# ---------------------------------------------------------------------------


def classify_outcome(
    *, previous: ReviewStatus, new: ReviewStatus, final: ReviewStatus
) -> RescoreOutcome:
    """Classify one rescore against the reviewer truth and the previous answer."""

    was_correct = previous == final
    is_correct = new == final

    if was_correct and is_correct:
        return "stable_correct"
    if was_correct and not is_correct:
        return "regressed"
    if not was_correct and is_correct:
        return "fixed"
    # Both wrong.
    if previous == new:
        return "stable_wrong_same"
    return "stable_wrong_different"


def rescore_record(
    record: GoldenRecord,
    requirement: Requirement,
    inputs: DocumentInputs,
    *,
    reranker: CrossEncoder | None = None,
    cer_store: MeasurementStore | None = None,
    judge: JudgeCallable | None = None,
    skip_llm: bool = False,
) -> RescoreResult:
    """Re-score one record. ``skip_llm`` short-circuits LLM requirements.

    ``reranker`` mirrors :func:`judge_requirement`. Pass the shared cross
    encoder only when the caller wants to measure the regression against
    the rerank-on branch; production regression defaults to ``None`` to
    match the opt-off setting.
    """

    if skip_llm and requirement.evaluation_method == "llm":
        return RescoreResult(
            document_id=record.document_id,
            requirement_id=record.requirement_id,
            previous_system_status=record.system_status,
            new_system_status=record.system_status,
            final_status=record.final_status,
            outcome="skipped",
            split=record.split,
            skipped_reason="llm skipped by flag",
        )

    new_result = judge_requirement(
        inputs.document_id,
        requirement,
        inputs.pages,
        inputs.ocr_pages,
        inputs.index,
        reranker=reranker,
        cer_store=cer_store,
        judge=judge,
    )
    outcome = classify_outcome(
        previous=record.system_status,
        new=new_result.status,
        final=record.final_status,
    )
    return RescoreResult(
        document_id=record.document_id,
        requirement_id=record.requirement_id,
        previous_system_status=record.system_status,
        new_system_status=new_result.status,
        final_status=record.final_status,
        outcome=outcome,
        split=record.split,
    )


def rescore_all(
    records: Iterable[GoldenRecord],
    requirements_by_id: Mapping[str, Requirement],
    *,
    extracted_dir: Path,
    ocr_dir: Path,
    chunks_dir: Path,
    reranker: CrossEncoder | None = None,
    cer_store: MeasurementStore | None = None,
    judge: JudgeCallable | None = None,
    skip_llm: bool = False,
    include_excluded: bool = False,
) -> list[RescoreResult]:
    """Re-score every labeled golden record.

    Excluded records (``split == "excluded"``) are skipped by default — they
    are not ground truth and cannot be scored. Set ``include_excluded`` only
    when caller wants to inspect them in the output.
    """

    inputs_cache: dict[str, DocumentInputs] = {}
    results: list[RescoreResult] = []
    for record in records:
        if record.split == "excluded" and not include_excluded:
            continue

        requirement = requirements_by_id.get(record.requirement_id)
        if requirement is None:
            _LOGGER.warning(
                "regression: no requirement registered for %s; skipping",
                record.requirement_id,
            )
            results.append(
                RescoreResult(
                    document_id=record.document_id,
                    requirement_id=record.requirement_id,
                    previous_system_status=record.system_status,
                    new_system_status=record.system_status,
                    final_status=record.final_status,
                    outcome="skipped",
                    split=record.split,
                    skipped_reason="requirement not registered",
                )
            )
            continue

        inputs = inputs_cache.get(record.document_id)
        if inputs is None:
            inputs = load_document_inputs(
                record.document_id,
                extracted_dir=extracted_dir,
                ocr_dir=ocr_dir,
                chunks_dir=chunks_dir,
            )
            inputs_cache[record.document_id] = inputs

        results.append(
            rescore_record(
                record,
                requirement,
                inputs,
                reranker=reranker,
                cer_store=cer_store,
                judge=judge,
                skip_llm=skip_llm,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionSummary:
    """Headline counts for a regression run."""

    total_scored: int
    skipped: int
    outcome_counts: Mapping[RescoreOutcome, int]
    correct_before: int
    correct_now: int
    accuracy_before: float
    accuracy_now: float
    accuracy_delta: float


def _accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def summarize_regression(results: Iterable[RescoreResult]) -> RegressionSummary:
    """Reduce the per-record outcomes to a one-page summary."""

    results_list = list(results)
    counts: Counter[RescoreOutcome] = Counter(r.outcome for r in results_list)
    scored = [r for r in results_list if r.outcome != "skipped"]
    skipped = len(results_list) - len(scored)

    correct_before = sum(1 for r in scored if r.previous_system_status == r.final_status)
    correct_now = sum(1 for r in scored if r.new_system_status == r.final_status)

    acc_before = _accuracy(correct_before, len(scored))
    acc_now = _accuracy(correct_now, len(scored))

    return RegressionSummary(
        total_scored=len(scored),
        skipped=skipped,
        outcome_counts=dict(counts),
        correct_before=correct_before,
        correct_now=correct_now,
        accuracy_before=acc_before,
        accuracy_now=acc_now,
        accuracy_delta=acc_now - acc_before,
    )


_OUTCOME_LABELS: dict[RescoreOutcome, str] = {
    "stable_correct": "stable correct (was right, still right)",
    "stable_wrong_same": "stable wrong (same prediction)",
    "stable_wrong_different": "stable wrong (different prediction)",
    "regressed": "regressed (was right, now wrong)",
    "fixed": "fixed (was wrong, now right)",
    "skipped": "skipped (no requirement or llm skipped)",
}


def render_regression_report(
    summary: RegressionSummary,
    results: Iterable[RescoreResult],
) -> str:
    """Produce the markdown report alongside the per-record JSON output."""

    results_list = list(results)
    header = "# Regression run report\n\n"
    overview = (
        f"- Records considered: {summary.total_scored + summary.skipped}.\n"
        f"- Scored: {summary.total_scored}.  Skipped: {summary.skipped}.\n"
        f"- Accuracy before: {summary.accuracy_before:.1%}"
        f" ({summary.correct_before}/{summary.total_scored}).\n"
        f"- Accuracy now:    {summary.accuracy_now:.1%}"
        f" ({summary.correct_now}/{summary.total_scored}).\n"
        f"- Delta: {summary.accuracy_delta:+.1%}.\n"
    )

    outcome_lines = ["", "## Outcome counts", ""]
    for outcome in (
        "stable_correct",
        "fixed",
        "regressed",
        "stable_wrong_same",
        "stable_wrong_different",
        "skipped",
    ):
        count = summary.outcome_counts.get(outcome, 0)
        if count == 0:
            continue
        outcome_lines.append(f"- {_OUTCOME_LABELS[outcome]}: {count}.")
    outcome_block = "\n".join(outcome_lines) + "\n"

    flips = [
        r for r in results_list if r.outcome in ("regressed", "fixed", "stable_wrong_different")
    ]
    if flips:
        flip_lines = [
            "",
            "## Flipped pairs",
            "",
            "| Document | Requirement | Previous | New | Final | Outcome |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for flip in flips:
            flip_lines.append(
                f"| `{flip.document_id}` | `{flip.requirement_id}`"
                f" | {flip.previous_system_status} | {flip.new_system_status}"
                f" | {flip.final_status} | {flip.outcome} |"
            )
        flip_block = "\n".join(flip_lines) + "\n"
    else:
        flip_block = "\n## Flipped pairs\n\n_No flips: every scored record kept its answer._\n"

    return header + overview + outcome_block + flip_block


# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------


def result_to_json(result: RescoreResult) -> dict[str, object]:
    """Convert one result to a JSON-friendly dict."""

    return {
        "document_id": result.document_id,
        "requirement_id": result.requirement_id,
        "previous_system_status": result.previous_system_status,
        "new_system_status": result.new_system_status,
        "final_status": result.final_status,
        "outcome": result.outcome,
        "split": result.split,
        "skipped_reason": result.skipped_reason,
    }
