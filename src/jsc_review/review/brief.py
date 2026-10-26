"""Regenerate the data-driven sections of ``Data/owner_review/brief.md``.

The brief mixes editorial prose about the system with numeric statements about
the current state of the review. The prose is stable; the numbers drift every
time a new document is uploaded or a reviewer records a decision. This module
regenerates only the drifting parts, inside sentinel-marked blocks:

    <!-- auto:begin:<name> -->
    ...generated markdown...
    <!-- auto:end:<name> -->

Every number here is derived from ``findings.json`` and ``decisions.json``
alone, so the brief stays correct for any uploaded document from any company.
The reviewer's decision wins over the system's recommendation: ``Met`` in the
brief means the effective status is ``Met`` after the reviewer has spoken, or
the system said ``Met`` and the reviewer has not yet responded.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jsc_review.core.models import ReviewResult, ReviewStatus
    from jsc_review.reviewer.models import ReviewerDecision


BLOCK_NAMES: tuple[str, ...] = (
    "scope",
    "counts",
    "met_table",
    "retrieval_gate",
    "owner_items",
    "footer",
)

_BLOCK_PATTERN = re.compile(
    r"<!--\s*auto:begin:(?P<name>[a-z_]+)\s*-->(?P<body>.*?)<!--\s*auto:end:(?P=name)\s*-->",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Snapshot: pure data derived from findings + decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectiveFinding:
    """One finding reduced to the fields the brief actually uses.

    ``status`` is the effective status after applying the reviewer decision.
    ``source`` records where the effective status came from so the brief can
    show whether the reviewer confirmed, changed, or has not yet spoken.
    """

    document_id: str
    requirement_id: str
    status: ReviewStatus
    evaluation_method: str
    page: int | None
    human_review: str
    source: str  # "system" | "reviewer:accepted" | "reviewer:changed" | "reviewer:deferred"
    reviewer_name: str | None = None
    reviewer_notes: str | None = None


@dataclass(frozen=True)
class GateStatus:
    """Snapshot of the measured retrieval gate.

    ``measured`` is ``False`` when no measurements file was supplied; the
    other fields are then placeholders. The ``method_rates`` dict preserves
    insertion order so the report can render pipelines in the same order the
    measurement script wrote them.
    """

    measured: bool
    k: int = 0
    threshold: float = 0.0
    passed: bool = False
    best_method: str = ""
    best_rate: float = 0.0
    best_total: int = 0
    method_rates: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BriefSnapshot:
    """Everything the brief needs, computed from findings + decisions."""

    document_ids: tuple[str, ...]
    requirement_ids: tuple[str, ...]
    findings: tuple[EffectiveFinding, ...]
    status_counts: Mapping[ReviewStatus, int]
    met_by_method: Mapping[str, int]  # "deterministic" | "llm"
    met_awaiting_review: tuple[EffectiveFinding, ...]
    decision_counts: Mapping[str, int] = field(default_factory=dict)
    total_findings: int = 0
    total_decisions: int = 0
    gate: GateStatus = field(default_factory=lambda: GateStatus(measured=False))


_ACTION_SOURCE: dict[str, str] = {
    "accepted": "reviewer:accepted",
    "changed": "reviewer:changed",
    "deferred": "reviewer:deferred",
}


def _effective_source(action: str) -> str:
    """Map a reviewer action to the brief's source label."""

    return _ACTION_SOURCE.get(action, "reviewer:accepted")


def parse_gate_measurements(payload: Mapping[str, object] | None) -> GateStatus:
    """Build a :class:`GateStatus` from the raw ``measurements.json`` payload.

    Returns an unmeasured status when ``payload`` is ``None``. The best
    pipeline is the one with the highest ``rate_at_k``; ties are broken by
    insertion order so the measurement script's preferred ordering wins.
    """

    if payload is None:
        return GateStatus(measured=False)

    gate_block = payload.get("gate")
    summaries = payload.get("summaries")
    if not isinstance(gate_block, Mapping) or not isinstance(summaries, Mapping):
        return GateStatus(measured=False)

    k = int(gate_block.get("k", 5))
    threshold = float(gate_block.get("threshold", 0.8))
    passed = bool(gate_block.get("passed", False))

    rate_key = f"rate_at_{k}"
    best_method = ""
    best_rate = 0.0
    best_total = 0
    method_rates: dict[str, float] = {}
    for method, summary in summaries.items():
        if not isinstance(summary, Mapping):
            continue
        rate_value = summary.get(rate_key)
        if not isinstance(rate_value, (int, float)):
            continue
        rate = float(rate_value)
        method_rates[str(method)] = rate
        if rate > best_rate:
            best_rate = rate
            best_method = str(method)
            total_value = summary.get("total")
            best_total = int(total_value) if isinstance(total_value, (int, float)) else 0

    return GateStatus(
        measured=True,
        k=k,
        threshold=threshold,
        passed=passed,
        best_method=best_method,
        best_rate=best_rate,
        best_total=best_total,
        method_rates=method_rates,
    )


def compute_snapshot(
    findings: Iterable[tuple[str, ReviewResult]],
    decisions: Iterable[ReviewerDecision],
    *,
    gate_measurements: Mapping[str, object] | None = None,
) -> BriefSnapshot:
    """Fold findings and decisions into a BriefSnapshot.

    The function is pure. It opens no files. It makes no assumptions about
    which companies or documents are present, so it stays correct when a new
    report is uploaded.
    """

    decisions_list = list(decisions)
    by_key: dict[tuple[str, str], ReviewerDecision] = {
        (d.document_id, d.requirement_id): d for d in decisions_list
    }

    effective: list[EffectiveFinding] = []
    status_counter: Counter[ReviewStatus] = Counter()
    met_by_method: Counter[str] = Counter()
    met_awaiting: list[EffectiveFinding] = []
    doc_order: list[str] = []
    req_order: list[str] = []
    seen_docs: set[str] = set()
    seen_reqs: set[str] = set()

    total = 0
    for document_id, result in findings:
        total += 1
        if document_id not in seen_docs:
            seen_docs.add(document_id)
            doc_order.append(document_id)
        if result.requirement_id not in seen_reqs:
            seen_reqs.add(result.requirement_id)
            req_order.append(result.requirement_id)

        decision = by_key.get((document_id, result.requirement_id))
        if decision is not None:
            status = decision.final_status
            source = _effective_source(decision.action)
            reviewer_name: str | None = decision.reviewer_name
            reviewer_notes: str | None = decision.reviewer_notes or None
        else:
            status = result.status
            source = "system"
            reviewer_name = None
            reviewer_notes = None

        page = result.evidence.page if result.evidence is not None else None
        finding = EffectiveFinding(
            document_id=document_id,
            requirement_id=result.requirement_id,
            status=status,
            evaluation_method=result.evaluation_method,
            page=page,
            human_review=result.human_review,
            source=source,
            reviewer_name=reviewer_name,
            reviewer_notes=reviewer_notes,
        )
        effective.append(finding)
        status_counter[status] += 1

        if status == "Met":
            met_by_method[result.evaluation_method] += 1
            # Awaiting review = system said Met, reviewer has not spoken, human_review Required.
            if source == "system" and result.human_review == "Required":
                met_awaiting.append(finding)

    decision_counter: Counter[str] = Counter(d.action for d in decisions_list)

    return BriefSnapshot(
        document_ids=tuple(doc_order),
        requirement_ids=tuple(req_order),
        findings=tuple(effective),
        status_counts=dict(status_counter),
        met_by_method=dict(met_by_method),
        met_awaiting_review=tuple(met_awaiting),
        decision_counts=dict(decision_counter),
        total_findings=total,
        total_decisions=len(decisions_list),
        gate=parse_gate_measurements(gate_measurements),
    )


# ---------------------------------------------------------------------------
# Render: snapshot -> markdown blocks (one per sentinel name)
# ---------------------------------------------------------------------------


def render_sections(snapshot: BriefSnapshot, *, now: datetime | None = None) -> dict[str, str]:
    """Render each sentinel-named block as a markdown string.

    The returned dict is keyed by sentinel name. The ``footer`` block holds the
    "last refreshed" line.
    """

    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    return {
        "scope": _render_scope(snapshot),
        "counts": _render_counts(snapshot),
        "met_table": _render_met_table(snapshot),
        "retrieval_gate": _render_retrieval_gate(snapshot),
        "owner_items": _render_owner_items(snapshot),
        "footer": _render_footer(snapshot, stamp),
    }


def _render_scope(snapshot: BriefSnapshot) -> str:
    reports = len(snapshot.document_ids)
    requirements = len(snapshot.requirement_ids)
    findings = snapshot.total_findings
    lines = [
        f"- Reports in scope: {reports}.",
        f"- Requirements covered: {requirements}.",
        f"- Recommendations on file: {findings}.",
    ]
    return "\n".join(lines)


def _render_counts(snapshot: BriefSnapshot) -> str:
    met = snapshot.status_counts.get("Met", 0)
    not_met = snapshot.status_counts.get("Not Met", 0)
    needs_review = snapshot.status_counts.get("Needs Review", 0)
    met_det = snapshot.met_by_method.get("deterministic", 0)
    met_llm = snapshot.met_by_method.get("llm", 0)

    decided = snapshot.total_decisions
    accepted = snapshot.decision_counts.get("accepted", 0)
    changed = snapshot.decision_counts.get("changed", 0)
    deferred = snapshot.decision_counts.get("deferred", 0)

    lines = [
        f"- `Met`: {met} (deterministic checks: {met_det}, local-model: {met_llm}).",
        f"- `Not Met`: {not_met}.",
        f"- `Needs Review`: {needs_review}.",
        f"- Reviewer decisions recorded: {decided}"
        f" (accepted: {accepted}, changed: {changed}, deferred: {deferred}).",
    ]
    return "\n".join(lines)


def _render_met_table(snapshot: BriefSnapshot) -> str:
    met = [f for f in snapshot.findings if f.status == "Met"]
    if not met:
        return "_No `Met` recommendations in the current review._"

    # Stable order: by document (first-seen), then requirement id.
    doc_order = {doc: index for index, doc in enumerate(snapshot.document_ids)}
    met_sorted = sorted(met, key=lambda f: (doc_order.get(f.document_id, 10**9), f.requirement_id))

    header = "| Document | Requirement | Method | Page | Source |\n| --- | --- | --- | --- | --- |"
    rows = [
        f"| `{f.document_id}` | `{f.requirement_id}` | {f.evaluation_method} |"
        f" {f.page if f.page is not None else '—'} | {f.source} |"
        for f in met_sorted
    ]
    return "\n".join([header, *rows])


def _render_retrieval_gate(snapshot: BriefSnapshot) -> str:
    """Render the measured retrieval gate as a short pass/fail block.

    When no measurement file was supplied, the block says so and names the
    local script that produces one, so an operator reading the brief knows
    what to run.

    When measurements are present, the block also contrasts the best pipeline
    against the next-best so a reviewer can see at a glance whether the
    reranker helps or hurts on the current label set. ``Settings.use_reranker``
    is driven by that comparison.
    """

    gate = snapshot.gate
    if not gate.measured:
        return (
            "- Retrieval gate: not measured yet. Run"
            " `python Data/reports/_measure_retrieval_gate.py` locally to produce"
            " the measurement."
        )

    verdict = "PASS" if gate.passed else "FAIL"
    headline = (
        f"- Retrieval gate ({verdict}): best Hit@{gate.k} = {gate.best_rate:.1%}"
        f" via `{gate.best_method}` on {gate.best_total} labeled requirements."
        f" Threshold is {gate.threshold:.0%}."
    )

    recommendation = _render_rerank_recommendation(gate)

    if not gate.method_rates:
        return "\n".join([headline, *recommendation])

    pipeline_lines = ["", "  Pipelines measured:"]
    for method, rate in gate.method_rates.items():
        pipeline_lines.append(f"  - `{method}`: {rate:.1%}")
    return "\n".join([headline, *recommendation, *pipeline_lines])


def _render_rerank_recommendation(gate: GateStatus) -> list[str]:
    """Return a one-line note comparing the best rerank-off vs rerank-on rates.

    Returns an empty list when the gate did not measure both branches, so the
    brief stays quiet instead of asserting a recommendation it can't support.
    """

    rates = gate.method_rates
    if not rates:
        return []
    off = max(
        (rate for name, rate in rates.items() if "rerank" not in name),
        default=None,
    )
    on = max(
        (rate for name, rate in rates.items() if "rerank" in name),
        default=None,
    )
    if off is None or on is None:
        return []
    if off >= on:
        delta = off - on
        return [
            f"  - Recommended: keep `Settings.use_reranker=False`."
            f" Rerank-off tops out at {off:.1%} vs {on:.1%} with rerank"
            f" (gain {delta:+.1%})."
        ]
    delta = on - off
    return [
        f"  - Consider setting `Settings.use_reranker=True`."
        f" Rerank-on reaches {on:.1%} vs {off:.1%} without it"
        f" (gain {delta:+.1%})."
    ]


def _render_owner_items(snapshot: BriefSnapshot) -> str:
    awaiting = snapshot.met_awaiting_review
    if not awaiting:
        return "- No `Met` recommendations are waiting for a reviewer decision at this time."

    lines = [
        f"- `Met` recommendations awaiting a reviewer decision: {len(awaiting)}.",
    ]
    doc_order = {doc: index for index, doc in enumerate(snapshot.document_ids)}
    sorted_awaiting = sorted(
        awaiting, key=lambda f: (doc_order.get(f.document_id, 10**9), f.requirement_id)
    )
    for finding in sorted_awaiting:
        page = f"p{finding.page}" if finding.page is not None else "no page"
        lines.append(
            f"  - `{finding.document_id}` · `{finding.requirement_id}`"
            f" · {finding.evaluation_method} · {page}"
        )
    return "\n".join(lines)


def _render_footer(snapshot: BriefSnapshot, stamp: str) -> str:
    return (
        f"_Last refreshed: {stamp} from {snapshot.total_findings} findings"
        f" and {snapshot.total_decisions} reviewer decisions._"
    )


# ---------------------------------------------------------------------------
# Update: splice rendered blocks into the brief file (idempotent)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateResult:
    """What ``update_brief`` did to the file."""

    replaced: tuple[str, ...]
    appended: tuple[str, ...]
    unchanged: tuple[str, ...]


def update_brief(brief_text: str, sections: Mapping[str, str]) -> tuple[str, UpdateResult]:
    """Replace each sentinel-marked block with its rendered content.

    If a block is missing from the file, append it at the end so a fresh file
    bootstraps itself on first run. The function is idempotent: running it
    twice with the same sections produces the same text.
    """

    replaced: list[str] = []
    unchanged: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        new_body = sections.get(name)
        if new_body is None:
            return match.group(0)
        existing = match.group("body").strip("\n")
        candidate = f"<!-- auto:begin:{name} -->\n{new_body}\n<!-- auto:end:{name} -->"
        if existing == new_body:
            unchanged.append(name)
        else:
            replaced.append(name)
        return candidate

    updated = _BLOCK_PATTERN.sub(_replace, brief_text)

    present = {match.group("name") for match in _BLOCK_PATTERN.finditer(brief_text)}
    appended: list[str] = []
    missing = [name for name in sections if name not in present]
    if missing:
        appended_parts = [updated.rstrip() + "\n"]
        for name in missing:
            appended_parts.append(
                f"\n<!-- auto:begin:{name} -->\n{sections[name]}\n<!-- auto:end:{name} -->\n"
            )
            appended.append(name)
        updated = "".join(appended_parts)

    return updated, UpdateResult(
        replaced=tuple(replaced),
        appended=tuple(appended),
        unchanged=tuple(unchanged),
    )
