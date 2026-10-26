"""Tests for the nightly check orchestrator.

The orchestrator has three jobs: run each step, aggregate the results into a
single verdict, and never let a per-step failure crash the whole run. Each
of these is tested in isolation (verdict math on synthetic steps) and
end-to-end (full orchestrator against a tmp_path layout with a mocked
judge).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from jsc_review.review.findings import Requirement
from jsc_review.review.nightly import (
    EXIT_CODE_FOR_VERDICT,
    NightlyPaths,
    StepStatus,
    compute_verdict,
    render_nightly_report,
    run_nightly_check,
)
from jsc_review.review.ollama import ModelJudgment

_PAGE_TEXT = "preamble verified passage marker text trailing"
_QUOTE = "verified passage marker text"


# ---------------------------------------------------------------------------
# compute_verdict
# ---------------------------------------------------------------------------


def _step(name: str, ok: bool, **details: object) -> StepStatus:
    return StepStatus(
        name=name,  # type: ignore[arg-type]
        ok=ok,
        message="test",
        details=details,
    )


def test_verdict_is_green_when_all_steps_ok_and_regression_clean() -> None:
    steps = (
        _step("brief", True),
        _step("golden", True),
        _step("regression", True, regressed=0, accuracy_delta=0.0),
    )

    assert compute_verdict(steps) == "green"


def test_verdict_is_fail_when_any_record_regressed() -> None:
    steps = (
        _step("brief", True),
        _step("golden", True),
        _step("regression", True, regressed=1, accuracy_delta=0.0),
    )

    assert compute_verdict(steps) == "fail"


def test_verdict_is_fail_when_accuracy_dropped() -> None:
    steps = (
        _step("brief", True),
        _step("golden", True),
        _step("regression", True, regressed=0, accuracy_delta=-0.05),
    )

    assert compute_verdict(steps) == "fail"


def test_verdict_is_warn_when_a_non_regression_step_failed() -> None:
    steps = (
        _step("brief", False),
        _step("golden", True),
        _step("regression", True, regressed=0, accuracy_delta=0.0),
    )

    assert compute_verdict(steps) == "warn"


def test_verdict_is_warn_when_regression_step_itself_failed() -> None:
    """A failing regression step is a warn, not a fail: we have no reading on it."""

    steps = (
        _step("brief", True),
        _step("golden", True),
        _step("regression", False),
    )

    assert compute_verdict(steps) == "warn"


# ---------------------------------------------------------------------------
# End-to-end orchestration with real step wrappers + tmp paths
# ---------------------------------------------------------------------------


def _write_findings(path: Path, document: str = "doc-a") -> None:
    findings = [
        {
            "document_id": document,
            "result": {
                "requirement_id": "GOV-001",
                "status": "Needs Review",
                "reason": "No supporting passage.",
                "evidence": None,
                "confidence": 0.4,
                "human_review": "Required",
                "evaluation_method": "llm",
            },
        }
    ]
    path.write_text(json.dumps(findings, ensure_ascii=False), encoding="utf-8")


def _write_decisions(path: Path, document: str = "doc-a") -> None:
    decisions = [
        {
            "document_id": document,
            "requirement_id": "GOV-001",
            "action": "changed",
            "final_status": "Met",
            "reviewer_name": "A. Reviewer",
            "reviewer_notes": "promoted to Met",
            "decided_at": "2026-10-25T10:00:00Z",
        }
    ]
    path.write_text(json.dumps(decisions, ensure_ascii=False), encoding="utf-8")


def _write_brief(path: Path) -> None:
    """A minimal sentinel-only brief; blocks will be appended on first refresh."""

    path.write_text("# Owner brief\n\nProse.\n", encoding="utf-8")


def _write_requirements(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "requirement_id",
                "section",
                "requirement_text_ar",
                "requirement_text_en",
                "requirement_type",
                "evaluation_method",
                "poc_selected",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "requirement_id": "GOV-001",
                "section": "s",
                "requirement_text_ar": "English text",
                "requirement_text_en": "English text",
                "requirement_type": "disclosure",
                "evaluation_method": "llm",
                "poc_selected": "yes",
            }
        )


def _write_extracted(dir_path: Path, document: str = "doc-a") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{document}.json").write_text(
        json.dumps({"pages": [{"page": 1, "text": _PAGE_TEXT}]}),
        encoding="utf-8",
    )


def _write_chunks(dir_path: Path, document: str = "doc-a") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{document}.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": f"{document}:p1:c01",
                        "document_id": document,
                        "page": 1,
                        "section": "s",
                        "text": _PAGE_TEXT,
                        "extraction_method": "pymupdf-text",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _build_paths(tmp_path: Path) -> NightlyPaths:
    out = tmp_path / "owner_review"
    out.mkdir(parents=True, exist_ok=True)
    brief_path = out / "brief.md"
    _write_brief(brief_path)
    return NightlyPaths(
        findings=tmp_path / "findings.json",
        decisions=tmp_path / "decisions.json",
        brief=brief_path,
        gate_measurements=tmp_path / "gate_missing.json",
        requirements=tmp_path / "requirements.csv",
        extracted_dir=tmp_path / "extracted",
        ocr_dir=tmp_path / "ocr",
        chunks_dir=tmp_path / "chunks",
        cer_measurements=tmp_path / "cer_missing.json",
        out_dir=out,
    )


def _judge(status: str) -> Callable[..., ModelJudgment]:
    def _call(_requirement: Requirement, _passage: str, *, passage_is_ocr: bool) -> ModelJudgment:
        _ = passage_is_ocr
        return ModelJudgment(status=status, reason="mocked", quote=_QUOTE, confidence=0.9)

    return _call


def test_orchestrator_end_to_end_green_when_judge_matches_truth(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    _write_findings(paths.findings)
    _write_decisions(paths.decisions)
    _write_requirements(paths.requirements)
    _write_extracted(paths.extracted_dir)
    _write_chunks(paths.chunks_dir)

    report = run_nightly_check(paths, judge=_judge("Met"))

    assert report.verdict == "green"
    for step in report.steps:
        assert step.ok, step.message
    # Golden dataset has 1 labeled row (the single reviewer decision).
    golden = report.step("golden")
    assert golden is not None
    assert golden.details["labeled"] == 1
    # Regression: baseline wrong (NR), judge now says Met => "fixed" => accuracy delta > 0.
    regression = report.step("regression")
    assert regression is not None
    assert regression.details["fixed"] == 1
    assert regression.details["regressed"] == 0


def test_orchestrator_marks_fail_when_judge_regresses_a_stable_correct_answer(
    tmp_path: Path,
) -> None:
    """A judge that flips a correct answer must drive the verdict to fail."""

    paths = _build_paths(tmp_path)
    # Set up a decision where the system was right (Met) but the judge will now say NR.
    findings = [
        {
            "document_id": "doc-a",
            "result": {
                "requirement_id": "GOV-001",
                "status": "Met",
                "reason": "ok",
                "evidence": {
                    "text": _QUOTE,
                    "source_document": "doc-a",
                    "page": 1,
                    "section": "s",
                    "chunk_id": "doc-a:p1:c01",
                    "verified": True,
                },
                "confidence": 0.9,
                "human_review": "Required",
                "evaluation_method": "llm",
            },
        }
    ]
    paths.findings.write_text(json.dumps(findings, ensure_ascii=False), encoding="utf-8")
    decisions = [
        {
            "document_id": "doc-a",
            "requirement_id": "GOV-001",
            "action": "accepted",
            "final_status": "Met",
            "reviewer_name": "A. Reviewer",
            "reviewer_notes": "",
            "decided_at": "2026-10-25T10:00:00Z",
        }
    ]
    paths.decisions.write_text(json.dumps(decisions, ensure_ascii=False), encoding="utf-8")
    _write_requirements(paths.requirements)
    _write_extracted(paths.extracted_dir)
    _write_chunks(paths.chunks_dir)

    report = run_nightly_check(paths, judge=_judge("Needs Review"))

    assert report.verdict == "fail"
    regression = report.step("regression")
    assert regression is not None
    assert regression.details["regressed"] == 1
    assert regression.details["fixed"] == 0


def test_orchestrator_continues_when_brief_file_missing(tmp_path: Path) -> None:
    """A missing brief must not stop the golden or regression steps."""

    paths = _build_paths(tmp_path)
    paths.brief.unlink()  # remove the brief file
    _write_findings(paths.findings)
    _write_decisions(paths.decisions)
    _write_requirements(paths.requirements)
    _write_extracted(paths.extracted_dir)
    _write_chunks(paths.chunks_dir)

    report = run_nightly_check(paths, judge=_judge("Met"))

    # Brief step warns, but golden and regression still ran and produced numbers.
    brief = report.step("brief")
    assert brief is not None and brief.ok is False
    golden = report.step("golden")
    assert golden is not None and golden.ok is True
    regression = report.step("regression")
    assert regression is not None and regression.ok is True
    # Verdict is warn because a non-critical step failed and regression was clean (fix).
    assert report.verdict == "warn"


def test_orchestrator_handles_no_decisions_gracefully(tmp_path: Path) -> None:
    """With zero decisions, golden has 0 rows and regression gets nothing to score."""

    paths = _build_paths(tmp_path)
    _write_findings(paths.findings)
    # Deliberately do not write decisions.json.
    _write_requirements(paths.requirements)
    _write_extracted(paths.extracted_dir)
    _write_chunks(paths.chunks_dir)

    report = run_nightly_check(paths, judge=_judge("Met"))

    golden = report.step("golden")
    assert golden is not None and golden.ok is True
    assert golden.details == {"records": 0}
    # No golden_dataset.json was written, so regression step short-circuits as ok with 0 scored.
    regression = report.step("regression")
    assert regression is not None and regression.ok is True
    assert regression.details == {"scored": 0}
    assert report.verdict == "green"


def test_render_nightly_report_includes_verdict_and_each_step(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    _write_findings(paths.findings)
    _write_decisions(paths.decisions)
    _write_requirements(paths.requirements)
    _write_extracted(paths.extracted_dir)
    _write_chunks(paths.chunks_dir)

    report = run_nightly_check(paths, judge=_judge("Met"))
    rendered = render_nightly_report(report)

    assert "# Nightly check" in rendered
    assert "Verdict:" in rendered
    assert "`brief`" in rendered
    assert "`golden`" in rendered
    assert "`regression`" in rendered


def test_exit_code_map_covers_every_verdict() -> None:
    assert set(EXIT_CODE_FOR_VERDICT.keys()) == {"green", "warn", "fail"}
    assert EXIT_CODE_FOR_VERDICT["green"] == 0
    assert EXIT_CODE_FOR_VERDICT["fail"] == 1
    assert EXIT_CODE_FOR_VERDICT["warn"] == 2


@pytest.mark.parametrize("verdict", ["green", "warn", "fail"])
def test_render_uses_distinct_label_per_verdict(verdict: str) -> None:
    from datetime import UTC, datetime

    from jsc_review.review.nightly import _VERDICT_LABELS, NightlyReport

    now = datetime.now(UTC)
    report = NightlyReport(verdict=verdict, started_at=now, finished_at=now, steps=())  # type: ignore[arg-type]
    rendered = render_nightly_report(report)

    assert _VERDICT_LABELS[verdict] in rendered  # type: ignore[index]
