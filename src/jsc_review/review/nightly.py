"""Nightly check: chain brief refresh, golden build, and regression into one run.

The operator schedules a nightly job (locally, by hand or by cron) that:

1. Refreshes ``brief.md`` from the current findings and decisions so the
   counts in the owner brief match reality.
2. Rebuilds the golden dataset from reviewer-confirmed decisions.
3. Re-scores the system against the golden dataset and reports flips.

Each step is wrapped so a failure in one does not cancel the others; the
operator still gets a status report that names which step broke and why. The
runner is deliberately stateless: it reads the local artifacts the pipeline
and reviewer already produce, writes new artifacts in place, and emits a
combined markdown status.

Verdicts
--------
* ``green``  — every step succeeded and no regressions.
* ``warn``   — a non-critical step failed (brief write, golden write) but
  the regression result was clean.
* ``fail``   — the regression runner saw accuracy drop or at least one
  reviewer-confirmed answer flipped from right to wrong.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from jsc_review.documents.cer_measurement import load_measurements
from jsc_review.review.brief_service import refresh_brief_from_disk
from jsc_review.review.findings import JudgeCallable, Requirement
from jsc_review.review.golden import (
    DEFAULT_SALT,
    build_golden_dataset,
    record_to_json,
    summarize_golden,
)
from jsc_review.review.golden import (
    render_report as render_golden_report,
)
from jsc_review.review.registry import load_selected_requirements
from jsc_review.review.regression import (
    render_regression_report,
    rescore_all,
    summarize_regression,
)
from jsc_review.review.regression import (
    result_to_json as rescore_result_to_json,
)
from jsc_review.reviewer.findings import load_findings
from jsc_review.reviewer.store import load_decisions

_LOGGER = logging.getLogger(__name__)

Verdict = Literal["green", "warn", "fail"]
StepName = Literal["brief", "golden", "regression"]


@dataclass(frozen=True)
class StepStatus:
    """Result of one nightly step."""

    name: StepName
    ok: bool
    message: str
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class NightlyPaths:
    """Local paths the nightly check reads and writes.

    Every path is a local file. No remote endpoints are touched.
    """

    findings: Path
    decisions: Path
    brief: Path
    gate_measurements: Path
    requirements: Path
    extracted_dir: Path
    ocr_dir: Path
    chunks_dir: Path
    cer_measurements: Path
    out_dir: Path  # directory for golden_dataset.json, regression_results.json, status.md


@dataclass(frozen=True)
class NightlyReport:
    """Combined result of all three steps."""

    verdict: Verdict
    started_at: datetime
    finished_at: datetime
    steps: tuple[StepStatus, ...]

    def step(self, name: StepName) -> StepStatus | None:
        for step in self.steps:
            if step.name == name:
                return step
        return None


# ---------------------------------------------------------------------------
# Step wrappers
# ---------------------------------------------------------------------------


def _refresh_brief_step(paths: NightlyPaths) -> StepStatus:
    try:
        summary = refresh_brief_from_disk(
            findings_path=paths.findings,
            decisions_path=paths.decisions,
            brief_path=paths.brief,
            gate_measurements_path=paths.gate_measurements,
        )
    except Exception as exc:  # noqa: BLE001  (nightly must not raise)
        _LOGGER.exception("nightly: brief refresh raised")
        return StepStatus(name="brief", ok=False, message=f"brief refresh raised: {exc}")

    if summary is None:
        return StepStatus(
            name="brief",
            ok=False,
            message="brief refresh skipped: findings or brief file missing",
        )
    return StepStatus(
        name="brief",
        ok=True,
        message=(
            f"brief refreshed from {summary.findings_count} findings"
            f" and {summary.decisions_count} decisions (wrote={summary.wrote})"
        ),
        details={
            "findings": summary.findings_count,
            "decisions": summary.decisions_count,
            "wrote": summary.wrote,
            "replaced": list(summary.replaced),
            "appended": list(summary.appended),
        },
    )


def _build_golden_step(
    paths: NightlyPaths, *, salt: str = DEFAULT_SALT, holdout_ratio: float = 0.3
) -> StepStatus:
    import json

    if not paths.findings.exists():
        return StepStatus(name="golden", ok=False, message="findings file missing")
    if not paths.decisions.exists():
        return StepStatus(
            name="golden",
            ok=True,
            message="no decisions yet; golden dataset not written",
            details={"records": 0},
        )

    try:
        findings = load_findings(paths.findings)
        decisions = load_decisions(paths.decisions)
        records = build_golden_dataset(findings, decisions, salt=salt, holdout_ratio=holdout_ratio)
        summary = summarize_golden(records)

        paths.out_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = paths.out_dir / "golden_dataset.json"
        report_path = paths.out_dir / "golden_report.md"
        dataset_path.write_text(
            json.dumps(
                {
                    "salt": salt,
                    "holdout_ratio": holdout_ratio,
                    "records": [record_to_json(r) for r in records],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        report_path.write_text(
            render_golden_report(summary, salt=salt, holdout_ratio=holdout_ratio),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("nightly: golden build raised")
        return StepStatus(name="golden", ok=False, message=f"golden build raised: {exc}")

    return StepStatus(
        name="golden",
        ok=True,
        message=(
            f"golden dataset: {summary.labeled} labeled rows"
            f" (dev={summary.dev_count}, val={summary.val_count},"
            f" excluded={summary.excluded}), accuracy={summary.accuracy:.1%}"
        ),
        details={
            "labeled": summary.labeled,
            "dev": summary.dev_count,
            "val": summary.val_count,
            "excluded": summary.excluded,
            "accuracy": summary.accuracy,
        },
    )


def _run_regression_step(
    paths: NightlyPaths,
    *,
    skip_llm: bool,
    judge: JudgeCallable | None = None,
    requirements_override: Mapping[str, Requirement] | None = None,
) -> StepStatus:
    import json

    dataset_path = paths.out_dir / "golden_dataset.json"
    if not dataset_path.exists():
        return StepStatus(
            name="regression",
            ok=True,
            message="no golden dataset to score against; skipping",
            details={"scored": 0},
        )
    if not paths.requirements.exists():
        return StepStatus(name="regression", ok=False, message="requirements register missing")

    try:
        from jsc_review.review.golden import GoldenRecord

        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        records: list[GoldenRecord] = []
        for raw in payload.get("records", []):
            records.append(
                GoldenRecord(
                    document_id=str(raw["document_id"]),
                    requirement_id=str(raw["requirement_id"]),
                    system_status=raw["system_status"],
                    final_status=raw["final_status"],
                    evaluation_method=str(raw["evaluation_method"]),
                    reviewer_action=str(raw["reviewer_action"]),
                    reviewer_name=str(raw["reviewer_name"]),
                    confidence=float(raw.get("confidence", 0.0)),
                    evidence_page=raw.get("evidence_page"),
                    evidence_verified=bool(raw.get("evidence_verified", False)),
                    split=raw["split"],
                    excluded_reason=raw.get("excluded_reason"),
                )
            )

        if requirements_override is not None:
            requirements_by_id = dict(requirements_override)
        else:
            requirements_by_id = {
                r.requirement_id: r for r in load_selected_requirements(paths.requirements)
            }

        cer_store = (
            load_measurements(paths.cer_measurements) if paths.cer_measurements.exists() else None
        )

        results = rescore_all(
            records,
            requirements_by_id,
            extracted_dir=paths.extracted_dir,
            ocr_dir=paths.ocr_dir,
            chunks_dir=paths.chunks_dir,
            cer_store=cer_store,
            judge=judge,
            skip_llm=skip_llm,
        )
        summary = summarize_regression(results)

        paths.out_dir.mkdir(parents=True, exist_ok=True)
        (paths.out_dir / "regression_results.json").write_text(
            json.dumps(
                {"results": [rescore_result_to_json(r) for r in results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (paths.out_dir / "regression_report.md").write_text(
            render_regression_report(summary, results),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("nightly: regression raised")
        return StepStatus(name="regression", ok=False, message=f"regression raised: {exc}")

    return StepStatus(
        name="regression",
        ok=True,
        message=(
            f"regression: {summary.total_scored} scored, {summary.skipped} skipped,"
            f" accuracy {summary.accuracy_before:.1%} -> {summary.accuracy_now:.1%}"
            f" ({summary.accuracy_delta:+.1%})"
        ),
        details={
            "scored": summary.total_scored,
            "skipped": summary.skipped,
            "accuracy_before": summary.accuracy_before,
            "accuracy_now": summary.accuracy_now,
            "accuracy_delta": summary.accuracy_delta,
            "regressed": summary.outcome_counts.get("regressed", 0),
            "fixed": summary.outcome_counts.get("fixed", 0),
            "skip_llm": skip_llm,
        },
    )


# ---------------------------------------------------------------------------
# Verdict + orchestrator
# ---------------------------------------------------------------------------


def compute_verdict(steps: tuple[StepStatus, ...]) -> Verdict:
    """Fold step results into a single verdict.

    Rules:
    * ``fail`` when the regression step saw a regressed outcome or an
      accuracy drop. A degraded answer on a reviewer-confirmed label is the
      strongest signal we have of a bad change.
    * ``warn`` when a step did not succeed but regression was clean (or was
      skipped). The operator should look, but it is not a known regression.
    * ``green`` otherwise.
    """

    regression = next((s for s in steps if s.name == "regression"), None)
    if regression is not None and regression.ok:
        regressed = _as_int(regression.details.get("regressed", 0))
        delta = _as_float(regression.details.get("accuracy_delta", 0.0))
        if regressed > 0 or delta < 0:
            return "fail"

    if any(not s.ok for s in steps):
        return "warn"
    return "green"


def run_nightly_check(
    paths: NightlyPaths,
    *,
    skip_llm: bool = False,
    salt: str = DEFAULT_SALT,
    holdout_ratio: float = 0.3,
    judge: JudgeCallable | None = None,
    requirements_override: Mapping[str, Requirement] | None = None,
) -> NightlyReport:
    """Run every step in order and return a :class:`NightlyReport`.

    The order matters: the golden dataset is rebuilt before the regression
    runs so any new reviewer decisions are picked up on this pass.
    """

    started = datetime.now(UTC)
    steps: list[StepStatus] = [
        _refresh_brief_step(paths),
        _build_golden_step(paths, salt=salt, holdout_ratio=holdout_ratio),
        _run_regression_step(
            paths,
            skip_llm=skip_llm,
            judge=judge,
            requirements_override=requirements_override,
        ),
    ]
    finished = datetime.now(UTC)
    return NightlyReport(
        verdict=compute_verdict(tuple(steps)),
        started_at=started,
        finished_at=finished,
        steps=tuple(steps),
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


_VERDICT_LABELS: dict[Verdict, str] = {
    "green": "GREEN — no regressions",
    "warn": "WARN — a non-critical step failed",
    "fail": "FAIL — regressions detected",
}


def render_nightly_report(report: NightlyReport) -> str:
    """Produce the combined markdown status report."""

    duration = (report.finished_at - report.started_at).total_seconds()
    header = "# Nightly check\n\n"
    overview = (
        f"- Verdict: **{_VERDICT_LABELS[report.verdict]}**.\n"
        f"- Started: {report.started_at.strftime('%Y-%m-%d %H:%M UTC')}.\n"
        f"- Finished: {report.finished_at.strftime('%Y-%m-%d %H:%M UTC')}"
        f" (duration {duration:.1f}s).\n"
    )

    step_lines = ["", "## Steps", ""]
    for step in report.steps:
        marker = "OK" if step.ok else "FAIL"
        step_lines.append(f"- `{step.name}` ({marker}): {step.message}")
    step_block = "\n".join(step_lines) + "\n"

    regression = report.step("regression")
    if regression is not None and regression.ok and _as_int(regression.details.get("scored", 0)):
        reg_lines = ["", "## Regression detail", ""]
        reg_lines.append(f"- Scored: {_as_int(regression.details.get('scored', 0))}.")
        reg_lines.append(f"- Skipped: {_as_int(regression.details.get('skipped', 0))}.")
        acc_before = _as_float(regression.details.get("accuracy_before", 0.0))
        acc_now = _as_float(regression.details.get("accuracy_now", 0.0))
        delta = _as_float(regression.details.get("accuracy_delta", 0.0))
        reg_lines.append(f"- Accuracy: {acc_before:.1%} -> {acc_now:.1%} ({delta:+.1%}).")
        reg_lines.append(f"- Regressed answers: {_as_int(regression.details.get('regressed', 0))}.")
        reg_lines.append(f"- Fixed answers: {_as_int(regression.details.get('fixed', 0))}.")
        regression_block = "\n".join(reg_lines) + "\n"
    else:
        regression_block = ""

    return header + overview + step_block + regression_block


def write_nightly_report(report: NightlyReport, out_dir: Path) -> Path:
    """Write the combined markdown to ``nightly_status.md`` under ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "nightly_status.md"
    target.write_text(render_nightly_report(report), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


EXIT_CODE_FOR_VERDICT: dict[Verdict, int] = {"green": 0, "warn": 2, "fail": 1}


# ---------------------------------------------------------------------------
# Small helpers: narrow ``object`` values from StepStatus.details so mypy and
# runtime both have a stable contract when reading them back.
# ---------------------------------------------------------------------------


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):  # bool is a subclass of int; keep intent explicit
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default
