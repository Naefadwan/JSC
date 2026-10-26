"""Removal helpers and the document-deletion orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from jsc_review.documents.cer_measurement import (
    MeasurementStore,
    PageMeasurement,
    load_measurements,
    remove_document_measurements,
    save_measurements,
)
from jsc_review.documents.removal import (
    DeletionSummary,
    remove_document,
    remove_findings_for_document,
    remove_from_bm25_index,
    remove_from_dense_index,
    remove_pdf,
    remove_single_json,
)
from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval import dense as dense_module
from jsc_review.retrieval.bm25 import IndexedChunk, build_index
from jsc_review.reviewer.models import ReviewerDecision
from jsc_review.reviewer.store import remove_decisions_for_document, save_decision

# ---------------------------------------------------------------- helpers ----


def _chunk(chunk_id: str, document_id: str, page: int = 1) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page=page,
        section="Body",
        text=f"text of {chunk_id}",
        extraction_method="pymupdf-text",
    )


class _StubEncoder:
    """Returns a deterministic one-hot style vector per text so cosine is well-defined."""

    def encode(self, texts) -> NDArray[np.float32]:
        rows = []
        for text in texts:
            vector = np.zeros(8, dtype=np.float32)
            vector[hash(text) % 8] = 1.0
            rows.append(vector)
        return np.vstack(rows)


def _write_findings(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------------------------- bm25 + dense remove_document ---


def test_bm25_remove_document_rebuilds_without_the_target() -> None:
    index = build_index(
        [
            _chunk("a:p1:c0", "doc_a"),
            _chunk("a:p2:c0", "doc_a", page=2),
            _chunk("b:p1:c0", "doc_b"),
        ]
    )
    rebuilt, removed = bm25_module.remove_document(index, "doc_a")

    assert removed == 2
    assert len(rebuilt.chunks) == 1
    assert rebuilt.chunks[0].document_id == "doc_b"


def test_bm25_remove_document_is_a_noop_when_target_missing() -> None:
    index = build_index([_chunk("a:p1:c0", "doc_a")])
    rebuilt, removed = bm25_module.remove_document(index, "doc_absent")

    assert removed == 0
    assert rebuilt is index


def test_dense_remove_document_rebuilds_faiss_index() -> None:
    chunks = [
        _chunk("a:p1:c0", "doc_a"),
        _chunk("b:p1:c0", "doc_b"),
        _chunk("b:p2:c0", "doc_b", page=2),
    ]
    index = dense_module.build_index(chunks, _StubEncoder(), batch_size=4)

    rebuilt, removed = dense_module.remove_document(index, "doc_b")

    assert removed == 2
    assert tuple(c.document_id for c in rebuilt.chunks) == ("doc_a",)
    assert rebuilt.faiss_index.ntotal == 1


def test_dense_remove_document_handles_full_wipe() -> None:
    chunks = [_chunk("a:p1:c0", "doc_a"), _chunk("a:p2:c0", "doc_a", page=2)]
    index = dense_module.build_index(chunks, _StubEncoder())

    rebuilt, removed = dense_module.remove_document(index, "doc_a")

    assert removed == 2
    assert rebuilt.chunks == ()
    assert rebuilt.faiss_index.ntotal == 0


# ---------------------------------------------- cer and decisions removal ---


def test_remove_document_measurements_filters_entries() -> None:
    store = MeasurementStore(
        threshold=0.05,
        measurements=(
            PageMeasurement(
                document_id="doc_a",
                page=1,
                engine="indra",
                cer=0.03,
                wer=None,
                reference_chars=100,
                hypothesis_chars=100,
                verified_evidence_eligible=True,
                measured_at="2026-10-25T00:00:00+00:00",
            ),
            PageMeasurement(
                document_id="doc_b",
                page=1,
                engine="indra",
                cer=0.08,
                wer=None,
                reference_chars=100,
                hypothesis_chars=100,
                verified_evidence_eligible=False,
                measured_at="2026-10-25T00:00:00+00:00",
            ),
        ),
    )
    rebuilt, removed = remove_document_measurements(store, "doc_a")
    assert removed == 1
    assert len(rebuilt.measurements) == 1
    assert rebuilt.measurements[0].document_id == "doc_b"
    assert rebuilt.threshold == 0.05


def test_remove_document_measurements_is_a_noop_when_not_present() -> None:
    store = MeasurementStore(threshold=0.05, measurements=())
    rebuilt, removed = remove_document_measurements(store, "doc_a")
    assert removed == 0
    assert rebuilt is store


def test_remove_decisions_for_document(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    save_decision(
        path,
        ReviewerDecision(
            document_id="doc_a",
            requirement_id="GOV-001",
            action="accepted",
            final_status="Met",
            reviewer_name="R",
            reviewer_notes="",
        ),
    )
    save_decision(
        path,
        ReviewerDecision(
            document_id="doc_b",
            requirement_id="GOV-002",
            action="accepted",
            final_status="Met",
            reviewer_name="R",
            reviewer_notes="",
        ),
    )

    removed = remove_decisions_for_document(path, "doc_a")

    assert removed == 1
    remaining = json.loads(path.read_text(encoding="utf-8"))
    assert len(remaining) == 1
    assert remaining[0]["document_id"] == "doc_b"


def test_remove_decisions_for_document_missing_file_returns_zero(tmp_path: Path) -> None:
    assert remove_decisions_for_document(tmp_path / "missing.json", "doc_a") == 0


# ------------------------------------------------- small store removers ----


def test_remove_pdf_unlinks_and_returns_count(tmp_path: Path) -> None:
    pdf = tmp_path / "doc_a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")

    assert remove_pdf(tmp_path, "doc_a") == 1
    assert not pdf.exists()
    assert remove_pdf(tmp_path, "doc_a") == 0


def test_remove_pdf_handles_none_dir() -> None:
    assert remove_pdf(None, "doc_a") == 0


def test_remove_single_json_unlinks_matching_file(tmp_path: Path) -> None:
    (tmp_path / "doc_a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "doc_b.json").write_text("{}", encoding="utf-8")

    assert remove_single_json(tmp_path, "doc_a") == 1
    assert not (tmp_path / "doc_a.json").exists()
    assert (tmp_path / "doc_b.json").exists()


def test_remove_findings_for_document_rewrites_file(tmp_path: Path) -> None:
    path = tmp_path / "findings.json"
    _write_findings(
        path,
        [
            {"document_id": "doc_a", "result": {"requirement_id": "R1"}},
            {"document_id": "doc_b", "result": {"requirement_id": "R1"}},
            {"document_id": "doc_a", "result": {"requirement_id": "R2"}},
        ],
    )

    removed = remove_findings_for_document(path, "doc_a")

    assert removed == 2
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert len(kept) == 1
    assert kept[0]["document_id"] == "doc_b"


def test_remove_findings_for_document_tolerates_missing_file(tmp_path: Path) -> None:
    assert remove_findings_for_document(tmp_path / "missing.json", "doc_a") == 0


def test_remove_findings_for_document_tolerates_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "findings.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert remove_findings_for_document(path, "doc_a") == 0


# ------------------------------------------------ bm25/dense disk removers --


def test_remove_from_bm25_index_rewrites_disk(tmp_path: Path) -> None:
    index = build_index(
        [
            _chunk("a:p1:c0", "doc_a"),
            _chunk("b:p1:c0", "doc_b"),
        ]
    )
    index_path = tmp_path / "bm25.json"
    bm25_module.save_index(index, index_path)

    removed = remove_from_bm25_index(index_path, "doc_a")

    assert removed == 1
    reloaded = bm25_module.load_index(index_path)
    assert len(reloaded.chunks) == 1
    assert reloaded.chunks[0].document_id == "doc_b"


def test_remove_from_dense_index_rewrites_disk(tmp_path: Path) -> None:
    index = dense_module.build_index(
        [_chunk("a:p1:c0", "doc_a"), _chunk("b:p1:c0", "doc_b")],
        _StubEncoder(),
        batch_size=4,
    )
    index_path = tmp_path / "dense.faiss"
    dense_module.save_index(index, index_path)

    removed = remove_from_dense_index(index_path, "doc_a")

    assert removed == 1
    reloaded = dense_module.load_index(index_path)
    assert len(reloaded.chunks) == 1
    assert reloaded.chunks[0].document_id == "doc_b"
    assert reloaded.faiss_index.ntotal == 1


def test_remove_from_bm25_index_handles_missing_path(tmp_path: Path) -> None:
    assert remove_from_bm25_index(tmp_path / "missing.json", "doc_a") == 0


# --------------------------------------------------------- orchestrator ----


def _build_full_layout(tmp_path: Path, document_id: str = "doc_a") -> dict[str, Path]:
    reports_dir = tmp_path / "reports"
    extracted_dir = tmp_path / "extracted"
    ocr_dir = tmp_path / "ocr"
    chunks_dir = tmp_path / "chunks"
    findings_path = tmp_path / "findings.json"
    decisions_path = tmp_path / "decisions.json"
    cer_path = tmp_path / "cer.json"
    bm25_path = tmp_path / "bm25.json"
    dense_path = tmp_path / "dense.faiss"

    reports_dir.mkdir()
    extracted_dir.mkdir()
    ocr_dir.mkdir()
    chunks_dir.mkdir()

    (reports_dir / f"{document_id}.pdf").write_bytes(b"%PDF-1.4\n%EOF\n")
    (extracted_dir / f"{document_id}.json").write_text("{}", encoding="utf-8")
    (ocr_dir / f"{document_id}.json").write_text("{}", encoding="utf-8")
    (chunks_dir / f"{document_id}.json").write_text("{}", encoding="utf-8")

    _write_findings(
        findings_path,
        [
            {"document_id": document_id, "result": {"requirement_id": "R1"}},
            {"document_id": "doc_keep", "result": {"requirement_id": "R1"}},
        ],
    )
    save_decision(
        decisions_path,
        ReviewerDecision(
            document_id=document_id,
            requirement_id="R1",
            action="accepted",
            final_status="Met",
            reviewer_name="R",
            reviewer_notes="",
        ),
    )

    save_measurements(
        [
            PageMeasurement(
                document_id=document_id,
                page=1,
                engine="indra",
                cer=0.03,
                wer=None,
                reference_chars=100,
                hypothesis_chars=100,
                verified_evidence_eligible=True,
                measured_at="2026-10-25T00:00:00+00:00",
            )
        ],
        cer_path,
        threshold=0.05,
    )

    bm25 = build_index(
        [_chunk(f"{document_id}:p1:c0", document_id), _chunk("keep:p1:c0", "doc_keep")]
    )
    bm25_module.save_index(bm25, bm25_path)

    dense = dense_module.build_index(
        [_chunk(f"{document_id}:p1:c0", document_id), _chunk("keep:p1:c0", "doc_keep")],
        _StubEncoder(),
        batch_size=4,
    )
    dense_module.save_index(dense, dense_path)

    return {
        "reports_dir": reports_dir,
        "extracted_dir": extracted_dir,
        "ocr_dir": ocr_dir,
        "chunks_dir": chunks_dir,
        "findings_path": findings_path,
        "decisions_path": decisions_path,
        "cer_path": cer_path,
        "bm25_path": bm25_path,
        "dense_path": dense_path,
    }


def test_remove_document_hits_every_store(tmp_path: Path) -> None:
    layout = _build_full_layout(tmp_path, document_id="doc_a")

    summary = remove_document(
        "doc_a",
        reports_dir=layout["reports_dir"],
        extracted_dir=layout["extracted_dir"],
        ocr_dir=layout["ocr_dir"],
        chunks_with_ocr_dir=layout["chunks_dir"],
        findings_path=layout["findings_path"],
        decisions_path=layout["decisions_path"],
        cer_measurements_path=layout["cer_path"],
        bm25_index_path=layout["bm25_path"],
        dense_index_path=layout["dense_path"],
    )

    assert isinstance(summary, DeletionSummary)
    assert summary.document_id == "doc_a"
    assert summary.all_ok
    assert summary.removed_anything

    def removed(name: str) -> int:
        step = summary.outcome(name)
        assert step is not None
        return step.removed

    assert removed("pdf") == 1
    assert removed("extracted") == 1
    assert removed("ocr") == 1
    assert removed("chunks_with_ocr") == 1
    assert removed("findings") == 1
    assert removed("decisions") == 1
    assert removed("cer_measurements") == 1
    assert removed("bm25_index") == 1
    assert removed("dense_index") == 1
    # The company link store is not configured in this layout, so it stays
    # untouched but reports a successful no-op.
    link_step = summary.outcome("company_link")
    assert link_step is not None and link_step.ok and link_step.removed == 0

    # Side-by-side doc stays intact.
    kept_findings = json.loads(layout["findings_path"].read_text(encoding="utf-8"))
    assert len(kept_findings) == 1
    assert kept_findings[0]["document_id"] == "doc_keep"

    kept_bm25 = bm25_module.load_index(layout["bm25_path"])
    assert len(kept_bm25.chunks) == 1
    assert kept_bm25.chunks[0].document_id == "doc_keep"

    kept_dense = dense_module.load_index(layout["dense_path"])
    assert len(kept_dense.chunks) == 1
    assert kept_dense.chunks[0].document_id == "doc_keep"

    kept_cer = load_measurements(layout["cer_path"])
    assert kept_cer.measurements == ()


def test_remove_document_drops_company_link(tmp_path: Path) -> None:
    """When the mapping path is wired in, the delete pipeline clears the link."""

    from jsc_review.companies.mapping import load_store, set_link

    mapping_path = tmp_path / "doc_companies.json"
    set_link(mapping_path, document_id="doc_a", company_key="JOIB")
    set_link(mapping_path, document_id="doc_keep", company_key="BOJX")
    assert len(load_store(mapping_path).links) == 2

    summary = remove_document(
        "doc_a",
        document_companies_path=mapping_path,
    )
    step = summary.outcome("company_link")
    assert step is not None and step.ok and step.removed == 1
    remaining = load_store(mapping_path)
    assert [link.document_id for link in remaining.links] == ["doc_keep"]

    # A follow-up delete finds nothing to drop.
    repeat = remove_document("doc_a", document_companies_path=mapping_path)
    repeat_step = repeat.outcome("company_link")
    assert repeat_step is not None and repeat_step.ok and repeat_step.removed == 0


def test_remove_document_is_fully_idempotent(tmp_path: Path) -> None:
    """Calling delete twice does not raise; the second call reports zero removals."""

    layout = _build_full_layout(tmp_path, document_id="doc_a")
    first = remove_document(
        "doc_a",
        reports_dir=layout["reports_dir"],
        extracted_dir=layout["extracted_dir"],
        ocr_dir=layout["ocr_dir"],
        chunks_with_ocr_dir=layout["chunks_dir"],
        findings_path=layout["findings_path"],
        decisions_path=layout["decisions_path"],
        cer_measurements_path=layout["cer_path"],
        bm25_index_path=layout["bm25_path"],
        dense_index_path=layout["dense_path"],
    )
    assert first.all_ok

    second = remove_document(
        "doc_a",
        reports_dir=layout["reports_dir"],
        extracted_dir=layout["extracted_dir"],
        ocr_dir=layout["ocr_dir"],
        chunks_with_ocr_dir=layout["chunks_dir"],
        findings_path=layout["findings_path"],
        decisions_path=layout["decisions_path"],
        cer_measurements_path=layout["cer_path"],
        bm25_index_path=layout["bm25_path"],
        dense_index_path=layout["dense_path"],
    )
    assert second.all_ok
    assert not second.removed_anything


def test_remove_document_skips_stores_with_none_paths(tmp_path: Path) -> None:
    """None paths quietly produce ok outcomes with zero removals."""

    summary = remove_document("doc_absent")

    assert summary.all_ok
    assert not summary.removed_anything
    assert len(summary.outcomes) == 10
    assert {outcome.name for outcome in summary.outcomes} == {
        "pdf",
        "extracted",
        "ocr",
        "chunks_with_ocr",
        "findings",
        "decisions",
        "cer_measurements",
        "bm25_index",
        "dense_index",
        "company_link",
    }


def test_remove_document_isolates_a_bad_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken sub-step records a failure without blocking the others."""

    layout = _build_full_layout(tmp_path, document_id="doc_a")

    from jsc_review.documents import removal as removal_module

    def broken(path, document_id):  # noqa: ARG001
        raise RuntimeError("simulated disk error")

    monkeypatch.setattr(removal_module, "remove_from_bm25_index", broken)

    summary = removal_module.remove_document(
        "doc_a",
        reports_dir=layout["reports_dir"],
        extracted_dir=layout["extracted_dir"],
        ocr_dir=layout["ocr_dir"],
        chunks_with_ocr_dir=layout["chunks_dir"],
        findings_path=layout["findings_path"],
        decisions_path=layout["decisions_path"],
        cer_measurements_path=layout["cer_path"],
        bm25_index_path=layout["bm25_path"],
        dense_index_path=layout["dense_path"],
    )

    bm25_step = summary.outcome("bm25_index")
    assert bm25_step is not None
    assert bm25_step.ok is False
    assert "simulated disk error" in bm25_step.detail

    # Everything else still ran.
    assert summary.outcome("pdf").ok  # type: ignore[union-attr]
    assert summary.outcome("findings").ok  # type: ignore[union-attr]
    assert summary.outcome("dense_index").ok  # type: ignore[union-attr]
    assert summary.all_ok is False
