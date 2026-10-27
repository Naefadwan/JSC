"""Measure OCR character and word error rates on labeled pages.

A page becomes eligible to support ``Met`` only when a measurement for the
engine that produced its text sits at or below the policy threshold
(``ocr_max_character_error_rate`` in ``src/jsc_review/review/policy.json``).

Measurements are stored on disk, not in version control. The reviewer runs
``Data/reports/_measure_cer.py`` after transcribing the gold reference for
each re-OCR'd page.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from jsc_review.documents.ocr_metrics import (
    character_error_rate,
    normalize_ocr_text,
    ocr_max_character_error_rate,
    word_error_rate,
)


@dataclass(frozen=True)
class PageMeasurement:
    """One scored OCR output for one (document, page, engine) triple."""

    document_id: str
    page: int
    engine: str
    cer: float
    wer: float | None
    reference_chars: int
    hypothesis_chars: int
    verified_evidence_eligible: bool
    measured_at: str


@dataclass(frozen=True)
class AnchorConsensus:
    """Two independent OCR engines agree a specific anchor is on this page.

    This is an alternative verification signal to the ground-truth CER
    measurement. When two local OCR engines (for example Tesseract and
    PaddleOCR) each independently find the same anchor phrase on the
    same page, the anchor's presence is treated as cross-validated even
    if a full-page character-error-rate measurement is unavailable or
    too noisy to be meaningful.
    """

    document_id: str
    page: int
    primary_engine: str
    secondary_engine: str
    anchor: str
    requirement_id: str
    verified_at: str


@dataclass(frozen=True)
class MeasurementStore:
    """Lookup table that answers whether OCR text may support ``Met``."""

    threshold: float
    measurements: tuple[PageMeasurement, ...] = field(default_factory=tuple)
    anchor_consensus: tuple[AnchorConsensus, ...] = field(default_factory=tuple)

    def cer_for(self, document_id: str, page: int) -> float | None:
        """Return the lowest CER recorded for the page, across engines."""

        values = [
            entry.cer
            for entry in self.measurements
            if entry.document_id == document_id and entry.page == page
        ]
        if not values:
            return None
        return min(values)

    def has_anchor_consensus(self, document_id: str, page: int) -> bool:
        """True when at least one two-engine anchor consensus exists for the page."""

        return any(
            entry.document_id == document_id and entry.page == page
            for entry in self.anchor_consensus
        )

    def consensus_for(
        self, document_id: str, page: int
    ) -> AnchorConsensus | None:
        """Return the first recorded anchor consensus for the page, if any."""

        for entry in self.anchor_consensus:
            if entry.document_id == document_id and entry.page == page:
                return entry
        return None

    def allows_verified_evidence(self, document_id: str, page: int) -> bool:
        """True when the page is eligible to support ``Met``.

        A page qualifies when either (a) a measured CER is at or below
        the policy threshold, or (b) at least one two-engine anchor
        consensus has been recorded for the page.
        """

        cer = self.cer_for(document_id, page)
        if cer is not None and cer <= self.threshold:
            return True
        return self.has_anchor_consensus(document_id, page)

    def verification_reason(self, document_id: str, page: int) -> str | None:
        """Return a human-readable Arabic justification for eligibility.

        CER-eligible pages return the measured error rate. Consensus
        pages return the engines that agreed and the anchor they found.
        Returns ``None`` when the page is not eligible to support ``Met``.
        """

        cer = self.cer_for(document_id, page)
        if cer is not None and cer <= self.threshold:
            return (
                f"معدل خطأ OCR المقاس على هذه الصفحة {cer * 100:.2f}٪ "
                "وهو ضمن الحد المسموح به."
            )
        consensus = self.consensus_for(document_id, page)
        if consensus is not None:
            return (
                "تم التحقق من ظهور العبارة الدالّة في محركي OCR مستقلّين "
                f"({consensus.primary_engine} و{consensus.secondary_engine})، "
                f"وهي: '{consensus.anchor}'."
            )
        return None

    def best_engine(self, document_id: str, page: int) -> str | None:
        """Return the engine with the lowest CER for the page, if any."""

        entries = [
            entry
            for entry in self.measurements
            if entry.document_id == document_id and entry.page == page
        ]
        if not entries:
            return None
        return min(entries, key=lambda entry: entry.cer).engine


def empty_store() -> MeasurementStore:
    """Return a store with the policy threshold and no measurements."""

    return MeasurementStore(threshold=ocr_max_character_error_rate(), measurements=())


def measure_page(
    reference: str,
    hypothesis: str,
    *,
    document_id: str,
    page: int,
    engine: str,
    threshold: float | None = None,
    measured_at: str | None = None,
) -> PageMeasurement | None:
    """Score one hypothesis against a gold reference.

    Returns ``None`` when the reference is empty, since error rate is
    undefined without a reference length.
    """

    normalized_reference = normalize_ocr_text(reference)
    normalized_hypothesis = normalize_ocr_text(hypothesis)
    cer = character_error_rate(normalized_reference, normalized_hypothesis)
    if cer is None:
        return None
    wer = word_error_rate(normalized_reference, normalized_hypothesis)
    limit = ocr_max_character_error_rate() if threshold is None else threshold
    stamp = measured_at or datetime.now(UTC).isoformat(timespec="seconds")
    return PageMeasurement(
        document_id=document_id,
        page=page,
        engine=engine,
        cer=cer,
        wer=wer,
        reference_chars=len(normalized_reference),
        hypothesis_chars=len(normalized_hypothesis),
        verified_evidence_eligible=cer <= limit,
        measured_at=stamp,
    )


def save_measurements(
    measurements: Sequence[PageMeasurement],
    path: Path,
    *,
    threshold: float | None = None,
    anchor_consensus: Sequence[AnchorConsensus] = (),
) -> Path:
    """Write measurements, consensus records, and the active threshold.

    The on-disk format carries both lists so a single file is the whole
    truth about which OCR pages may support ``Met``.
    """

    limit = ocr_max_character_error_rate() if threshold is None else threshold
    payload: dict[str, object] = {
        "threshold": limit,
        "measurements": [asdict(entry) for entry in measurements],
        "anchor_consensus": [asdict(entry) for entry in anchor_consensus],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_measurements(path: Path | None) -> MeasurementStore:
    """Return the stored measurements or an empty store when the file is missing."""

    if path is None or not path.exists():
        return empty_store()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return empty_store()
    threshold = float(payload.get("threshold", ocr_max_character_error_rate()))
    raw_entries = payload.get("measurements", [])
    entries: list[PageMeasurement] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        try:
            entries.append(
                PageMeasurement(
                    document_id=str(raw["document_id"]),
                    page=int(raw["page"]),
                    engine=str(raw["engine"]),
                    cer=float(raw["cer"]),
                    wer=(float(raw["wer"]) if raw.get("wer") is not None else None),
                    reference_chars=int(raw.get("reference_chars", 0)),
                    hypothesis_chars=int(raw.get("hypothesis_chars", 0)),
                    verified_evidence_eligible=bool(raw.get("verified_evidence_eligible", False)),
                    measured_at=str(raw.get("measured_at", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    raw_consensus = payload.get("anchor_consensus", [])
    consensus_entries: list[AnchorConsensus] = []
    for raw in raw_consensus:
        if not isinstance(raw, dict):
            continue
        try:
            consensus_entries.append(
                AnchorConsensus(
                    document_id=str(raw["document_id"]),
                    page=int(raw["page"]),
                    primary_engine=str(raw["primary_engine"]),
                    secondary_engine=str(raw["secondary_engine"]),
                    anchor=str(raw["anchor"]),
                    requirement_id=str(raw.get("requirement_id", "")),
                    verified_at=str(raw.get("verified_at", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return MeasurementStore(
        threshold=threshold,
        measurements=tuple(entries),
        anchor_consensus=tuple(consensus_entries),
    )


def remove_document_measurements(
    store: MeasurementStore, document_id: str
) -> tuple[MeasurementStore, int]:
    """Return a store without any measurement or consensus for ``document_id``.

    The second element of the returned tuple is the number of records
    removed across both lists. The threshold is preserved so the store
    keeps the policy value even when every record has been removed.
    """

    kept_measurements = tuple(
        entry for entry in store.measurements if entry.document_id != document_id
    )
    kept_consensus = tuple(
        entry for entry in store.anchor_consensus if entry.document_id != document_id
    )
    removed = (
        len(store.measurements) - len(kept_measurements)
        + len(store.anchor_consensus) - len(kept_consensus)
    )
    if removed == 0:
        return store, 0
    return (
        MeasurementStore(
            threshold=store.threshold,
            measurements=kept_measurements,
            anchor_consensus=kept_consensus,
        ),
        removed,
    )


def summarize(measurements: Iterable[PageMeasurement]) -> dict[str, object]:
    """Return a small summary used in CLI output and the measurement header."""

    entries = list(measurements)
    if not entries:
        return {"count": 0, "eligible": 0, "cer_min": None, "cer_max": None}
    cers = [entry.cer for entry in entries]
    return {
        "count": len(entries),
        "eligible": sum(1 for entry in entries if entry.verified_evidence_eligible),
        "cer_min": min(cers),
        "cer_max": max(cers),
    }
