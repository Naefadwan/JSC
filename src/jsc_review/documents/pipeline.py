"""End-to-end processing for a single uploaded PDF.

Runs locally with no outside calls: extract embedded text, OCR pages that
need it, build a per-document BM25 index, judge every selected requirement
with the local model, and append each finding to the shared findings file.

The findings file is updated incrementally so a long-running job surfaces
partial progress to the reviewer screen.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jsc_review.documents.cer_measurement import MeasurementStore, load_measurements
from jsc_review.documents.chunk import chunk_pdf
from jsc_review.documents.extract import extract_pdf, save_extraction
from jsc_review.documents.ocr import OcrEngineError, ocr_page_record, recognize_pdf_page
from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval import dense as dense_module
from jsc_review.retrieval.bm25 import (
    Bm25Index,
    IndexedChunk,
    build_index,
    save_indexed_chunks,
)
from jsc_review.retrieval.dense import DenseIndex
from jsc_review.retrieval.reranker import CrossEncoder
from jsc_review.review.findings import Requirement, judge_requirement

_LOGGER = logging.getLogger(__name__)
_INVALID_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")

Progress = Callable[[str, str], None]
"""A callback that receives (stage, message) during processing."""

StageProgress = Callable[[str, int | None, int | None], None]
"""A callback that receives (stage, done, total) between stage boundaries.

``done`` and ``total`` may both be ``None`` to indicate the start of a stage
whose size is not yet known. A value of ``(stage, 0, N)`` marks the beginning
of a bounded stage; ``(stage, i, N)`` marks a tick inside it.
"""

CancelCheck = Callable[[], bool]
"""A callable that returns True when the pipeline should abort cleanly."""

AutolinkHook = Callable[[str, tuple[tuple[int, str], ...]], None]
"""Receives ``(document_id, pages)`` after extraction to decide a company link.

``pages`` is a tuple of ``(page_number, raw_text)`` pairs. The hook owns the
decision policy (threshold, confirmation, logging). It must not raise; the
pipeline swallows exceptions so autolink can never fail a review.
"""

_findings_lock = threading.Lock()
_global_index_lock = threading.Lock()


class UploadError(ValueError):
    """Raised when an uploaded file cannot be accepted."""


class UploadCanceled(RuntimeError):
    """Raised by the pipeline when a cancel request is observed."""


@dataclass(frozen=True)
class ProcessedDocument:
    """Summary of a one-document pipeline run."""

    document_id: str
    pdf_path: Path
    page_count: int
    ocr_page_count: int
    chunk_count: int
    requirement_count: int
    finding_count: int


def sanitize_document_id(filename: str) -> str:
    """Return a safe stem for a user-supplied PDF filename."""

    stem = Path(filename).stem.strip()
    if not stem:
        raise UploadError("The uploaded filename has no stem")
    safe = _INVALID_FILENAME.sub("_", stem).strip("_.")
    if not safe:
        raise UploadError("The filename only contains unsupported characters")
    return safe


def is_pdf_bytes(data: bytes) -> bool:
    """Return True when the leading bytes are a PDF magic header."""

    return data[:5] == b"%PDF-"


def save_uploaded_pdf(
    *,
    data: bytes,
    original_filename: str,
    reports_dir: Path,
) -> Path:
    """Validate and store an uploaded PDF locally. Raises on duplicate."""

    if not is_pdf_bytes(data):
        raise UploadError("The uploaded file is not a PDF")
    document_id = sanitize_document_id(original_filename)
    reports_dir.mkdir(parents=True, exist_ok=True)
    target = reports_dir / f"{document_id}.pdf"
    if target.exists():
        raise UploadError(f"A document with id {document_id} already exists")
    target.write_bytes(data)
    return target


def _notify(progress: Progress | None, stage: str, message: str) -> None:
    if progress is not None:
        progress(stage, message)


def _tick(
    stage_progress: StageProgress | None,
    stage: str,
    done: int | None,
    total: int | None,
) -> None:
    if stage_progress is not None:
        stage_progress(stage, done, total)


def _check_cancel(is_canceled: CancelCheck | None) -> None:
    if is_canceled is not None and is_canceled():
        raise UploadCanceled("Processing was canceled by the reviewer")


def process_uploaded_pdf(
    pdf_path: Path,
    *,
    requirements: Sequence[Requirement],
    extracted_dir: Path,
    ocr_dir: Path,
    findings_path: Path,
    progress: Progress | None = None,
    stage_progress: StageProgress | None = None,
    is_canceled: CancelCheck | None = None,
    bm25_index_path: Path | None = None,
    dense_index_path: Path | None = None,
    update_dense: bool = True,
    use_dense_at_review: bool = False,
    reranker: CrossEncoder | None = None,
    cer_measurements_path: Path | None = None,
    chunks_with_ocr_dir: Path | None = None,
    autolink_hook: AutolinkHook | None = None,
) -> ProcessedDocument:
    """Run the full local pipeline for one PDF and append findings in place.

    ``reranker`` is passed through to :func:`judge_requirement`. Production
    leaves it ``None`` by default because the measured retrieval gate shows
    the cross-encoder lowers top-k recall on the current label set; the
    caller flips it on by reading ``Settings.use_reranker`` and loading
    :func:`jsc_review.retrieval.reranker.shared_cross_encoder`.
    """

    document_id = pdf_path.stem
    cer_store = load_measurements(cer_measurements_path)

    _check_cancel(is_canceled)
    _notify(progress, "extracting", "قراءة النص المضمَّن من ملف PDF")
    _tick(stage_progress, "extracting", None, None)
    extracted = extract_pdf(pdf_path)
    save_extraction(extracted, extracted_dir)
    pages: dict[int, str] = {page.page: page.text for page in extracted.pages}
    ocr_candidates = [page for page in extracted.pages if page.needs_ocr]

    # Auto-link the document to a company based on the extracted text.
    # Hook is optional so the pipeline keeps working without a registry.
    if autolink_hook is not None:
        try:
            autolink_hook(document_id, tuple((p.page, p.text) for p in extracted.pages))
        except Exception:  # noqa: BLE001
            # Autolink is a side activity and must never fail the review run.
            _LOGGER.exception("Autolink hook failed for %s", document_id)

    _check_cancel(is_canceled)
    _notify(
        progress,
        "ocr",
        f"صفحات تحتاج إلى OCR: {len(ocr_candidates)} من {len(extracted.pages)}",
    )
    _tick(stage_progress, "ocr", 0, len(ocr_candidates))
    ocr_pages = _run_ocr(
        document_id,
        pdf_path,
        ocr_candidates,
        ocr_dir,
        progress,
        stage_progress=stage_progress,
        is_canceled=is_canceled,
    )

    _check_cancel(is_canceled)
    _notify(progress, "indexing", "بناء فهرس BM25 للمستند")
    _tick(stage_progress, "indexing", None, None)
    chunked = chunk_pdf(pdf_path)
    index_chunks = _chunks_for_index(chunked.chunks, ocr_pages)
    index = build_index(index_chunks) if index_chunks else None
    if chunks_with_ocr_dir is not None and index_chunks:
        save_indexed_chunks(index_chunks, chunks_with_ocr_dir, document_id)

    # Optional dense retrieval at review time. Load once per upload so the
    # encoder is not reinstantiated per requirement. Any failure degrades to
    # BM25-only review; the loop never raises on retrieval setup.
    dense_for_review = _maybe_load_dense_for_review(
        use_dense_at_review=use_dense_at_review,
        dense_index_path=dense_index_path,
        document_id=document_id,
        progress=progress,
    )

    _check_cancel(is_canceled)
    _notify(progress, "judging", f"حكم {len(requirements)} متطلب")
    _tick(stage_progress, "judging", 0, len(requirements))
    appended = 0
    for position, requirement in enumerate(requirements, start=1):
        _check_cancel(is_canceled)
        record = _review_one(
            document_id=document_id,
            requirement=requirement,
            pages=pages,
            ocr_pages=ocr_pages,
            index=index,
            dense=dense_for_review,
            reranker=reranker,
            cer_store=cer_store,
        )
        _append_finding(findings_path, record)
        appended += 1
        _notify(
            progress,
            "judging",
            f"{requirement.requirement_id} ({position}/{len(requirements)})",
        )
        _tick(stage_progress, "judging", position, len(requirements))

    if bm25_index_path is not None and index_chunks:
        _check_cancel(is_canceled)
        _tick(stage_progress, "indexing_global", None, None)
        _sync_global_indexes(
            new_chunks=index_chunks,
            bm25_index_path=bm25_index_path,
            dense_index_path=dense_index_path,
            update_dense=update_dense,
            progress=progress,
        )

    _notify(progress, "done", "اكتملت المعالجة")
    _tick(stage_progress, "done", None, None)
    return ProcessedDocument(
        document_id=document_id,
        pdf_path=pdf_path,
        page_count=len(extracted.pages),
        ocr_page_count=len(ocr_pages),
        chunk_count=len(index_chunks),
        requirement_count=len(requirements),
        finding_count=appended,
    )


def _sync_global_indexes(
    *,
    new_chunks: Sequence[IndexedChunk],
    bm25_index_path: Path,
    dense_index_path: Path | None,
    update_dense: bool,
    progress: Progress | None,
) -> None:
    """Append new chunks to the global BM25 and (optional) dense indexes.

    This is best-effort. A missing BM25 file is skipped. A missing or
    unloadable dense index is skipped. An encoder-load failure is skipped.
    Everything is reported through the progress callback.
    """

    with _global_index_lock:
        if not bm25_index_path.exists():
            _notify(
                progress,
                "indexing_global",
                "فهرس BM25 العام غير موجود. تخطي التحديث العام.",
            )
            return
        _notify(progress, "indexing_global", "تحديث فهرس BM25 العام")
        try:
            global_bm25 = bm25_module.load_index(bm25_index_path)
            updated = bm25_module.append_chunks(global_bm25, list(new_chunks))
            bm25_module.save_index(updated, bm25_index_path)
            added = len(updated.chunks) - len(global_bm25.chunks)
            _notify(
                progress,
                "indexing_global",
                f"أُضيف {added} مقطع إلى فهرس BM25 العام",
            )
        except (OSError, ValueError, KeyError) as exc:
            _notify(
                progress,
                "indexing_global",
                f"فشل تحديث فهرس BM25 العام: {exc}",
            )
            return

        if not update_dense or dense_index_path is None or not dense_index_path.exists():
            return
        _notify(progress, "indexing_global", "ترميز وتحديث فهرس BGE-M3 العام")
        try:
            global_dense = dense_module.load_index(dense_index_path)
            from jsc_review.retrieval.embedding import (
                SentenceTransformerEncoder,
                embedding_settings,
            )

            encoder = SentenceTransformerEncoder(embedding_settings())
            updated_dense = dense_module.append_chunks(
                global_dense, list(new_chunks), encoder, batch_size=8
            )
            dense_module.save_index(updated_dense, dense_index_path)
            added = len(updated_dense.chunks) - len(global_dense.chunks)
            _notify(
                progress,
                "indexing_global",
                f"أُضيف {added} مقطع إلى فهرس BGE-M3 العام",
            )
        except Exception as exc:  # noqa: BLE001
            _notify(
                progress,
                "indexing_global",
                f"فشل تحديث فهرس BGE-M3 العام (تجاوز): {exc}",
            )


def _run_ocr(
    document_id: str,
    pdf_path: Path,
    candidates: Sequence[object],
    ocr_dir: Path,
    progress: Progress | None,
    *,
    stage_progress: StageProgress | None = None,
    is_canceled: CancelCheck | None = None,
) -> dict[int, str]:
    """Run OCR for the pages that embedded extraction left short."""

    ocr_pages: dict[int, str] = {}
    page_records: list[dict[str, object]] = []
    for position, candidate in enumerate(candidates, start=1):
        _check_cancel(is_canceled)
        page_number = getattr(candidate, "page", None)
        if not isinstance(page_number, int):
            continue
        _notify(
            progress,
            "ocr",
            f"OCR صفحة {page_number} ({position}/{len(candidates)})",
        )
        try:
            text = recognize_pdf_page(pdf_path, page_number)
        except OcrEngineError as exc:
            _notify(progress, "ocr", f"فشل OCR صفحة {page_number}: {exc}")
            _tick(stage_progress, "ocr", position, len(candidates))
            continue
        ocr_pages[page_number] = text
        record = ocr_page_record(document_id=document_id, page_number=page_number, text=text)
        page_records.append(asdict(record))
        _tick(stage_progress, "ocr", position, len(candidates))

    if page_records:
        ocr_dir.mkdir(parents=True, exist_ok=True)
        payload = {"document_id": document_id, "pages": page_records}
        (ocr_dir / f"{document_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return ocr_pages


def _chunks_for_index(
    chunks: Sequence[object],
    ocr_pages: dict[int, str],
) -> list[IndexedChunk]:
    """Build searchable chunks: all embedded chunks plus one OCR chunk per page."""

    indexed: list[IndexedChunk] = []
    for chunk in chunks:
        document_id = str(getattr(chunk, "document_id", ""))
        page = int(getattr(chunk, "page", 0))
        text = str(getattr(chunk, "text", "")).strip()
        if not document_id or not page or not text:
            continue
        indexed.append(
            IndexedChunk(
                chunk_id=str(getattr(chunk, "chunk_id", f"{document_id}:p{page}:c?")),
                document_id=document_id,
                page=page,
                section=str(getattr(chunk, "section", "")),
                text=text,
                extraction_method=str(getattr(chunk, "extraction_method", "pymupdf-text")),
            )
        )
    document_id = indexed[0].document_id if indexed else ""
    for page, text in ocr_pages.items():
        if not text:
            continue
        indexed.append(
            IndexedChunk(
                chunk_id=f"{document_id}:p{page}:ocr",
                document_id=document_id,
                page=page,
                section="",
                text=text,
                extraction_method="tesseract-ara+eng",
            )
        )
    return indexed


def _review_one(
    *,
    document_id: str,
    requirement: Requirement,
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    index: Bm25Index | None,
    dense: DenseIndex | None = None,
    reranker: CrossEncoder | None = None,
    cer_store: MeasurementStore | None = None,
) -> dict[str, object]:
    """Review one requirement; call the local model when needed."""

    result = judge_requirement(
        document_id,
        requirement,
        pages,
        ocr_pages,
        index,
        dense=dense,
        reranker=reranker,
        cer_store=cer_store,
    )
    return {"document_id": document_id, "result": result.model_dump()}


def _maybe_load_dense_for_review(
    *,
    use_dense_at_review: bool,
    dense_index_path: Path | None,
    document_id: str,
    progress: Progress | None,
) -> DenseIndex | None:
    """Load the dense FAISS index + encoder for review-time hybrid search.

    Returns None when dense retrieval is disabled, when the index file is
    missing, when the encoder cannot load, or when the index has no chunks
    for this document (BM25-only is strictly cheaper in that case). Any
    exception degrades gracefully; the review loop keeps running on BM25.
    """

    if not use_dense_at_review:
        return None
    if dense_index_path is None or not dense_index_path.exists():
        _notify(
            progress,
            "indexing",
            "تخطي البحث الكثيف: ملف الفهرس غير موجود.",
        )
        return None
    try:
        global_dense = dense_module.load_index(dense_index_path)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        # FAISS raises RuntimeError on a corrupt file. Any load failure
        # degrades to BM25-only review; we never let it break the loop.
        _notify(progress, "indexing", f"تعذّر تحميل الفهرس الكثيف: {exc}")
        return None
    if not any(chunk.document_id == document_id for chunk in global_dense.chunks):
        # The current upload's chunks are not in the dense index yet, so
        # dense search for this document would return nothing useful.
        return None
    try:
        from jsc_review.retrieval.embedding import (
            SentenceTransformerEncoder,
            embedding_settings,
        )

        encoder = SentenceTransformerEncoder(embedding_settings())
    except Exception as exc:  # noqa: BLE001
        _notify(progress, "indexing", f"تعذّر تحميل مُرمّز BGE-M3: {exc}")
        return None
    return global_dense.with_encoder(encoder)


def _append_finding(findings_path: Path, record: dict[str, object]) -> None:
    """Append one finding to the shared file atomically within the process."""

    with _findings_lock:
        findings_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict[str, object]] = []
        if findings_path.exists():
            try:
                payload = json.loads(findings_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    existing = payload
            except json.JSONDecodeError:
                existing = []
        existing.append(record)
        temp_path = findings_path.with_suffix(findings_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(findings_path)
