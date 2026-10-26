"""Pipeline cancellation and stage-progress plumbing."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from jsc_review.documents.pipeline import (
    UploadCanceled,
    process_uploaded_pdf,
)
from jsc_review.review.findings import Requirement


def _write_tiny_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        for text in pages:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)  # type: ignore[no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]


def _requirements() -> list[Requirement]:
    return [
        Requirement(
            requirement_id="R-01",
            section="S1",
            text_ar="نص المتطلب",
            text_en="Requirement text",
            evaluation_method="قاعدة محددة",
        ),
    ]


def test_pipeline_raises_upload_canceled_when_flag_is_set(tmp_path: Path) -> None:
    """A cancel flag observed before extraction aborts the pipeline cleanly."""

    pdf_path = tmp_path / "report.pdf"
    _write_tiny_pdf(pdf_path, ["The board adopted a risk management policy."])

    with pytest.raises(UploadCanceled):
        process_uploaded_pdf(
            pdf_path,
            requirements=_requirements(),
            extracted_dir=tmp_path / "extracted",
            ocr_dir=tmp_path / "ocr",
            findings_path=tmp_path / "findings.json",
            is_canceled=lambda: True,
        )


def test_pipeline_emits_stage_progress_boundaries(tmp_path: Path) -> None:
    """Stage callbacks fire at least once per major stage boundary."""

    pdf_path = tmp_path / "report.pdf"
    _write_tiny_pdf(pdf_path, ["Deterministic rule evidence appears here."])

    stages_seen: list[str] = []

    def record(stage: str, _done: int | None, _total: int | None) -> None:
        if not stages_seen or stages_seen[-1] != stage:
            stages_seen.append(stage)

    process_uploaded_pdf(
        pdf_path,
        requirements=_requirements(),
        extracted_dir=tmp_path / "extracted",
        ocr_dir=tmp_path / "ocr",
        findings_path=tmp_path / "findings.json",
        stage_progress=record,
    )

    # Order matters: extraction, indexing, judging, done. OCR fires only when
    # a page needs it; this tiny PDF has embedded text so OCR may be empty.
    assert stages_seen[0] == "extracting"
    assert "indexing" in stages_seen
    assert "judging" in stages_seen
    assert stages_seen[-1] == "done"


def test_pipeline_cancel_mid_judging_stops_before_all_requirements(
    tmp_path: Path,
) -> None:
    """A cancel observed inside the judging loop aborts the next iteration."""

    pdf_path = tmp_path / "report.pdf"
    _write_tiny_pdf(pdf_path, ["Deterministic rule evidence appears here."])

    requirements = [
        Requirement(
            requirement_id=f"R-{i:02d}",
            section="S1",
            text_ar="نص",
            text_en="text",
            evaluation_method="قاعدة محددة",
        )
        for i in range(5)
    ]

    judged: list[str] = []

    def stage_progress(stage: str, done: int | None, _total: int | None) -> None:
        if stage == "judging" and isinstance(done, int) and done > 0:
            judged.append(f"R-{done - 1:02d}")

    cancel_after = {"n": 2}

    def is_canceled() -> bool:
        return len(judged) >= cancel_after["n"]

    with pytest.raises(UploadCanceled):
        process_uploaded_pdf(
            pdf_path,
            requirements=requirements,
            extracted_dir=tmp_path / "extracted",
            ocr_dir=tmp_path / "ocr",
            findings_path=tmp_path / "findings.json",
            stage_progress=stage_progress,
            is_canceled=is_canceled,
        )

    # At least some requirements ran, but strictly fewer than the full set.
    assert 1 <= len(judged) < len(requirements)
