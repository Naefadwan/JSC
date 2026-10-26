"""Tests for the OCR character error rate measurement helpers."""

from pathlib import Path

from jsc_review.documents.cer_measurement import (
    MeasurementStore,
    PageMeasurement,
    empty_store,
    load_measurements,
    measure_page,
    save_measurements,
    summarize,
)


def test_measure_page_computes_cer_and_marks_eligibility() -> None:
    reference = "قائمة المركز المالي الموحدة"
    hypothesis = "قائمة المركز المالي الموحدة"

    measurement = measure_page(
        reference,
        hypothesis,
        document_id="doc",
        page=10,
        engine="indra",
        threshold=0.05,
    )

    assert measurement is not None
    assert measurement.cer == 0.0
    assert measurement.verified_evidence_eligible is True


def test_measure_page_marks_noisy_output_ineligible() -> None:
    reference = "قائمة المركز المالي الموحدة للسنة المنتهية في ٣١ كانون الأول ٢٠٢٥"
    hypothesis = "قائمة المركزي المالي الموحده للسنة المنته 2025"

    measurement = measure_page(
        reference,
        hypothesis,
        document_id="doc",
        page=10,
        engine="tesseract",
        threshold=0.05,
    )

    assert measurement is not None
    assert measurement.cer > 0.05
    assert measurement.verified_evidence_eligible is False


def test_measure_page_returns_none_when_reference_is_empty() -> None:
    assert (
        measure_page(
            "",
            "whatever",
            document_id="doc",
            page=1,
            engine="indra",
        )
        is None
    )


def test_measurement_store_picks_the_lowest_cer_across_engines() -> None:
    store = MeasurementStore(
        threshold=0.05,
        measurements=(
            PageMeasurement(
                document_id="d",
                page=5,
                engine="tesseract",
                cer=0.42,
                wer=0.6,
                reference_chars=100,
                hypothesis_chars=100,
                verified_evidence_eligible=False,
                measured_at="2026-10-25T00:00:00+00:00",
            ),
            PageMeasurement(
                document_id="d",
                page=5,
                engine="indra",
                cer=0.03,
                wer=0.1,
                reference_chars=100,
                hypothesis_chars=100,
                verified_evidence_eligible=True,
                measured_at="2026-10-25T00:00:00+00:00",
            ),
        ),
    )

    assert store.cer_for("d", 5) == 0.03
    assert store.best_engine("d", 5) == "indra"
    assert store.allows_verified_evidence("d", 5) is True


def test_measurement_store_denies_pages_without_a_measurement() -> None:
    store = empty_store()

    assert store.cer_for("d", 1) is None
    assert store.best_engine("d", 1) is None
    assert store.allows_verified_evidence("d", 1) is False


def test_save_and_load_measurements_roundtrips(tmp_path: Path) -> None:
    measurement = PageMeasurement(
        document_id="d",
        page=3,
        engine="indra",
        cer=0.02,
        wer=0.05,
        reference_chars=120,
        hypothesis_chars=118,
        verified_evidence_eligible=True,
        measured_at="2026-10-25T00:00:00+00:00",
    )
    target = tmp_path / "cer.json"

    save_measurements([measurement], target, threshold=0.05)
    loaded = load_measurements(target)

    assert loaded.threshold == 0.05
    assert len(loaded.measurements) == 1
    assert loaded.measurements[0].document_id == "d"
    assert loaded.allows_verified_evidence("d", 3) is True


def test_load_measurements_returns_empty_store_when_file_is_missing(
    tmp_path: Path,
) -> None:
    loaded = load_measurements(tmp_path / "does-not-exist.json")

    assert loaded.measurements == ()
    assert loaded.allows_verified_evidence("d", 1) is False


def test_summarize_counts_eligible_entries() -> None:
    entries = [
        PageMeasurement(
            document_id="d",
            page=1,
            engine="indra",
            cer=0.02,
            wer=0.05,
            reference_chars=100,
            hypothesis_chars=99,
            verified_evidence_eligible=True,
            measured_at="2026-10-25T00:00:00+00:00",
        ),
        PageMeasurement(
            document_id="d",
            page=2,
            engine="tesseract",
            cer=0.4,
            wer=0.5,
            reference_chars=100,
            hypothesis_chars=100,
            verified_evidence_eligible=False,
            measured_at="2026-10-25T00:00:00+00:00",
        ),
    ]

    summary = summarize(entries)

    assert summary["count"] == 2
    assert summary["eligible"] == 1
    assert summary["cer_min"] == 0.02
    assert summary["cer_max"] == 0.4
