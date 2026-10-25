"""End-to-end processing for a single uploaded PDF.

Runs locally with no outside calls: extract embedded text, OCR pages that
need it, build a per-document BM25 index, judge every selected requirement
with the local model, and append each finding to the shared findings file.

The findings file is updated incrementally so a long-running job surfaces
partial progress to the reviewer screen.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jsc_review.documents.chunk import chunk_pdf
from jsc_review.documents.extract import extract_pdf, save_extraction
from jsc_review.documents.ocr import OcrEngineError, ocr_page_record, recognize_pdf_page
from jsc_review.retrieval.bm25 import Bm25Index, IndexedChunk, build_index
from jsc_review.review.findings import (
    Requirement,
    apply_model_judgment,
    locate_passage,
    review_document,
)
from jsc_review.review.ollama import LocalModelError, judge_passage

_INVALID_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")

Progress = Callable[[str, str], None]
"""A callback that receives (stage, message) during processing."""

_findings_lock = threading.Lock()


class UploadError(ValueError):
    """Raised when an uploaded file cannot be accepted."""


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


def process_uploaded_pdf(
    pdf_path: Path,
    *,
    requirements: Sequence[Requirement],
    extracted_dir: Path,
    ocr_dir: Path,
    findings_path: Path,
    progress: Progress | None = None,
) -> ProcessedDocument:
    """Run the full local pipeline for one PDF and append findings in place."""

    document_id = pdf_path.stem
    _notify(progress, "extracting", "قراءة النص المضمَّن من ملف PDF")
    extracted = extract_pdf(pdf_path)
    save_extraction(extracted, extracted_dir)
    pages: dict[int, str] = {page.page: page.text for page in extracted.pages}
    ocr_candidates = [page for page in extracted.pages if page.needs_ocr]

    _notify(
        progress,
        "ocr",
        f"صفحات تحتاج إلى OCR: {len(ocr_candidates)} من {len(extracted.pages)}",
    )
    ocr_pages = _run_ocr(document_id, pdf_path, ocr_candidates, ocr_dir, progress)

    _notify(progress, "indexing", "بناء فهرس BM25 للمستند")
    chunked = chunk_pdf(pdf_path)
    index_chunks = _chunks_for_index(chunked.chunks, ocr_pages)
    index = build_index(index_chunks) if index_chunks else None

    _notify(progress, "judging", f"حكم {len(requirements)} متطلب")
    appended = 0
    for position, requirement in enumerate(requirements, start=1):
        record = _review_one(
            document_id=document_id,
            requirement=requirement,
            pages=pages,
            ocr_pages=ocr_pages,
            index=index,
        )
        _append_finding(findings_path, record)
        appended += 1
        _notify(
            progress,
            "judging",
            f"{requirement.requirement_id} ({position}/{len(requirements)})",
        )

    _notify(progress, "done", "اكتملت المعالجة")
    return ProcessedDocument(
        document_id=document_id,
        pdf_path=pdf_path,
        page_count=len(extracted.pages),
        ocr_page_count=len(ocr_pages),
        chunk_count=len(index_chunks),
        requirement_count=len(requirements),
        finding_count=appended,
    )


def _run_ocr(
    document_id: str,
    pdf_path: Path,
    candidates: Sequence[object],
    ocr_dir: Path,
    progress: Progress | None,
) -> dict[int, str]:
    """Run OCR for the pages that embedded extraction left short."""

    ocr_pages: dict[int, str] = {}
    page_records: list[dict[str, object]] = []
    for position, candidate in enumerate(candidates, start=1):
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
            continue
        ocr_pages[page_number] = text
        record = ocr_page_record(
            document_id=document_id, page_number=page_number, text=text
        )
        page_records.append(asdict(record))

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
                extraction_method=str(
                    getattr(chunk, "extraction_method", "pymupdf-text")
                ),
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
) -> dict[str, object]:
    """Review one requirement; call the local model when needed."""

    if requirement.evaluation_method != "llm":
        finding = review_document(
            document_id, requirement, pages, ocr_pages, index, dense=None
        )
        return {"document_id": document_id, "result": finding.result.model_dump()}

    passage = locate_passage(document_id, requirement, pages, index, dense=None)
    if passage is None:
        finding = review_document(
            document_id, requirement, pages, ocr_pages, index, dense=None
        )
        return {"document_id": document_id, "result": finding.result.model_dump()}
    try:
        judgment = judge_passage(
            requirement, passage.text, passage_is_ocr=passage.passage_is_ocr
        )
    except LocalModelError as exc:
        finding = review_document(
            document_id, requirement, pages, ocr_pages, index, dense=None
        )
        result = finding.result.model_copy(
            update={"reason": f"{finding.result.reason} خطأ النموذج المحلي: {exc}"}
        )
        return {"document_id": document_id, "result": result.model_dump()}
    accepted = apply_model_judgment(
        document_id,
        requirement,
        passage,
        status=judgment.status,
        reason=judgment.reason,
        quote=judgment.quote,
        confidence=judgment.confidence,
        pages=pages,
    )
    return {"document_id": document_id, "result": accepted.model_dump()}


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
        temp_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_path.replace(findings_path)
