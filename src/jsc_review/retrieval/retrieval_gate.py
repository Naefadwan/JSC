"""Measure retrieval quality against labeled (document, requirement) pairs.

The Phase 2 gate asks that the correct passage for the selected requirements
appears in the top five retrieval results at least 80% of the time. "Correct
passage" is operationalized here as any page already verified by the
pipeline: a Met result's cited page, or a Needs Review result whose evidence
quote was found on an embedded page by the verification check.

This is an honest but limited measurement. It answers "can retrieval
reproduce its own good results across engines?" rather than "does retrieval
find absolute truth?" A hand-labeled set would strengthen the gate; this
measurement is a floor, not a ceiling.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jsc_review.retrieval.bm25 import Bm25Index, SearchHit
from jsc_review.retrieval.dense import DenseIndex
from jsc_review.retrieval.hybrid import reciprocal_rank_fusion
from jsc_review.retrieval.reranker import CrossEncoder
from jsc_review.retrieval.search import HybridConfig, hybrid_search
from jsc_review.review.findings import Requirement

RetrievalMethod = str  # "bm25" | "dense" | "fused"


@dataclass(frozen=True)
class RetrievalLabel:
    """One ground-truth entry: pages considered correct for a (doc, req) pair."""

    document_id: str
    requirement_id: str
    correct_pages: tuple[int, ...]
    source: str


@dataclass(frozen=True)
class MethodResult:
    """Per-method retrieval outcome for one label."""

    method: RetrievalMethod
    top_pages: tuple[int, ...]
    rank: int | None  # 1-indexed rank of the first correct page, or None


@dataclass(frozen=True)
class MeasurementResult:
    """Retrieval quality for one labeled pair across every evaluated method."""

    label: RetrievalLabel
    requirement: Requirement
    method_results: tuple[MethodResult, ...]

    def rank_for(self, method: RetrievalMethod) -> int | None:
        for entry in self.method_results:
            if entry.method == method:
                return entry.rank
        return None


@dataclass(frozen=True)
class GateSummary:
    """Hit@k numbers for one method across all measurements."""

    method: RetrievalMethod
    total: int
    hits: dict[int, int] = field(default_factory=dict)

    def rate(self, k: int) -> float:
        if self.total == 0:
            return 0.0
        return self.hits.get(k, 0) / self.total


def collect_labels_from_findings(
    findings_path: Path,
    *,
    accepted_statuses: Iterable[str] = ("Met", "Needs Review"),
    require_verified: bool = True,
) -> list[RetrievalLabel]:
    """Extract (document, requirement, page) triples from a findings file.

    By default only evidence with ``verified=True`` is kept. Repeating a
    (document_id, requirement_id) merges the pages into one label.
    """

    if not findings_path.exists():
        return []
    payload = json.loads(findings_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    accepted = set(accepted_statuses)
    pages_by_pair: dict[tuple[str, str], set[int]] = {}
    sources_by_pair: dict[tuple[str, str], str] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        document_id = str(row.get("document_id", "")).strip()
        result = row.get("result")
        if not document_id or not isinstance(result, dict):
            continue
        requirement_id = str(result.get("requirement_id", "")).strip()
        status = str(result.get("status", "")).strip()
        if not requirement_id or status not in accepted:
            continue
        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            continue
        if require_verified and not bool(evidence.get("verified", False)):
            continue
        page_value = evidence.get("page")
        if not isinstance(page_value, int) or page_value <= 0:
            continue
        key = (document_id, requirement_id)
        pages_by_pair.setdefault(key, set()).add(page_value)
        sources_by_pair.setdefault(key, f"finding:{status.lower().replace(' ', '_')}")
    labels: list[RetrievalLabel] = []
    for key, pages in sorted(pages_by_pair.items()):
        document_id, requirement_id = key
        labels.append(
            RetrievalLabel(
                document_id=document_id,
                requirement_id=requirement_id,
                correct_pages=tuple(sorted(pages)),
                source=sources_by_pair[key],
            )
        )
    return labels


def _rank_of_correct_page(
    hits: Sequence[SearchHit], correct_pages: Iterable[int]
) -> tuple[int | None, tuple[int, ...]]:
    """Return the 1-indexed rank of the first correct page and the top page list."""

    wanted = set(correct_pages)
    top_pages: list[int] = []
    first_rank: int | None = None
    for position, hit in enumerate(hits, start=1):
        top_pages.append(hit.page)
        if first_rank is None and hit.page in wanted:
            first_rank = position
    return first_rank, tuple(top_pages)


def _query_for(requirement: Requirement) -> str:
    parts: list[str] = []
    if requirement.text_ar:
        parts.append(requirement.text_ar)
    if requirement.text_en:
        parts.append(requirement.text_en)
    return " ".join(parts).strip() or requirement.requirement_id


def measure_label(
    label: RetrievalLabel,
    requirement: Requirement,
    *,
    bm25: Bm25Index,
    dense: DenseIndex | None,
    reranker: CrossEncoder | None = None,
    pipelines: Mapping[str, HybridConfig] | None = None,
    k_search: int = 100,
) -> MeasurementResult:
    """Run retrieval for one label across every raw engine and named pipeline.

    The three raw methods are always evaluated as a baseline: ``bm25``,
    ``dense`` (when present), and ``fused`` RRF. Entries in ``pipelines``
    add one row per named configuration using the full composed hybrid
    pipeline (optional query augmentation + reranker + page dedup).
    """

    query = _query_for(requirement)
    bm25_hits = bm25.search(query, k=k_search, document_id=label.document_id)
    bm25_rank, bm25_pages = _rank_of_correct_page(bm25_hits, label.correct_pages)
    method_entries: list[MethodResult] = [
        MethodResult(method="bm25", top_pages=bm25_pages, rank=bm25_rank)
    ]
    if dense is not None:
        dense_hits = dense.search(query, k=k_search, document_id=label.document_id)
        dense_rank, dense_pages = _rank_of_correct_page(dense_hits, label.correct_pages)
        method_entries.append(MethodResult(method="dense", top_pages=dense_pages, rank=dense_rank))
        fused_hits = reciprocal_rank_fusion([bm25_hits, dense_hits], top_k=k_search)
        fused_rank, fused_pages = _rank_of_correct_page(fused_hits, label.correct_pages)
        method_entries.append(MethodResult(method="fused", top_pages=fused_pages, rank=fused_rank))

    if pipelines:
        for name, config in pipelines.items():
            hits = hybrid_search(
                requirement,
                label.document_id,
                bm25=bm25,
                dense=dense,
                reranker=reranker,
                config=config,
            )
            rank, pages = _rank_of_correct_page(hits, label.correct_pages)
            method_entries.append(MethodResult(method=name, top_pages=pages, rank=rank))

    return MeasurementResult(
        label=label,
        requirement=requirement,
        method_results=tuple(method_entries),
    )


def measure_retrieval(
    labels: Sequence[RetrievalLabel],
    requirements_by_id: Mapping[str, Requirement],
    *,
    bm25: Bm25Index,
    dense: DenseIndex | None,
    reranker: CrossEncoder | None = None,
    pipelines: Mapping[str, HybridConfig] | None = None,
    k_search: int = 100,
) -> list[MeasurementResult]:
    """Measure every label whose requirement is known."""

    measurements: list[MeasurementResult] = []
    for label in labels:
        requirement = requirements_by_id.get(label.requirement_id)
        if requirement is None:
            continue
        measurements.append(
            measure_label(
                label,
                requirement,
                bm25=bm25,
                dense=dense,
                reranker=reranker,
                pipelines=pipelines,
                k_search=k_search,
            )
        )
    return measurements


def summarize_measurements(
    measurements: Sequence[MeasurementResult],
    *,
    ks: Iterable[int] = (1, 3, 5, 10),
) -> dict[RetrievalMethod, GateSummary]:
    """Compute Hit@k for every method represented in the measurements."""

    k_values = sorted(set(ks))
    totals: dict[RetrievalMethod, int] = {}
    hits: dict[RetrievalMethod, dict[int, int]] = {}
    for measurement in measurements:
        for entry in measurement.method_results:
            totals[entry.method] = totals.get(entry.method, 0) + 1
            per_k = hits.setdefault(entry.method, {k: 0 for k in k_values})
            if entry.rank is not None:
                for k in k_values:
                    if entry.rank <= k:
                        per_k[k] += 1
    return {
        method: GateSummary(method=method, total=totals[method], hits=hits[method])
        for method in totals
    }


def gate_passes(summary: GateSummary, *, k: int = 5, threshold: float = 0.80) -> bool:
    """Return True when the method's Hit@k rate meets the configured threshold."""

    return summary.rate(k) >= threshold
