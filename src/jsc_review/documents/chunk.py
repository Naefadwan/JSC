"""Split extracted pages into heading sections."""

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

HEADING_SIZE_MARGIN = 1.5
HEADING_MAX_CHARS = 120


@dataclass(frozen=True)
class TextChunk:
    """One heading section from a single page."""

    document_id: str
    page: int
    section: str
    chunk_id: str
    text: str
    extraction_method: str


@dataclass(frozen=True)
class ChunkedDocument:
    """Heading chunks for one local PDF."""

    document_id: str
    path: Path
    chunks: tuple[TextChunk, ...]


def chunk_pdf(path: Path) -> ChunkedDocument:
    """Chunk each page by font size. The full page record remains chunk c00."""

    document = pymupdf.open(path)  # type: ignore[no-untyped-call]
    try:
        document_id = path.stem
        chunks: list[TextChunk] = []
        for index in range(int(document.page_count)):
            page = document.load_page(index)  # type: ignore[no-untyped-call]
            chunks.extend(_chunk_page(document_id, index + 1, _line_sizes(page)))
    finally:
        document.close()  # type: ignore[no-untyped-call]
    return ChunkedDocument(document_id=path.stem, path=path, chunks=tuple(chunks))


def save_chunks(chunked: ChunkedDocument, directory: Path) -> Path:
    """Write heading chunks to a local JSON file."""

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{chunked.document_id}.json"
    payload = {
        "document_id": chunked.document_id,
        "source_path": chunked.path.as_posix(),
        "chunk_count": len(chunked.chunks),
        "chunks": [asdict(chunk) for chunk in chunked.chunks],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def chunks_from_lines(
    document_id: str,
    page_number: int,
    lines: list[tuple[str, float]],
) -> tuple[TextChunk, ...]:
    """Group lines into sections. A larger font starts a new section."""

    usable = [(text.strip(), size) for text, size in lines if text.strip()]
    if not usable:
        return ()
    body_size = statistics.median(size for _, size in usable)
    sections: list[tuple[str, list[str]]] = [("", [])]
    for text, size in usable:
        if _is_heading(text, size, body_size) and sections[-1][1]:
            sections.append((text, [text]))
            continue
        section, body = sections[-1]
        if _is_heading(text, size, body_size) and not body:
            sections[-1] = (text, [text])
            continue
        body.append(text)
        sections[-1] = (section, body)
    chunks: list[TextChunk] = []
    for index, (section, body) in enumerate(sections, start=1):
        passage = "\n".join(body).strip()
        if not passage:
            continue
        chunks.append(
            TextChunk(
                document_id=document_id,
                page=page_number,
                section=section,
                chunk_id=f"{document_id}:p{page_number}:c{index:02d}",
                text=passage,
                extraction_method="pymupdf-text",
            )
        )
    return tuple(chunks)


def _chunk_page(
    document_id: str,
    page_number: int,
    lines: list[tuple[str, float]],
) -> tuple[TextChunk, ...]:
    return chunks_from_lines(document_id, page_number, lines)


def _line_sizes(page: pymupdf.Page) -> list[tuple[str, float]]:
    raw = page.get_text("dict")  # type: ignore[no-untyped-call]
    if not isinstance(raw, dict):
        return []
    blocks = raw.get("blocks", [])
    if not isinstance(blocks, list):
        return []
    lines: list[tuple[str, float]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            if not isinstance(line, dict):
                continue
            spans = line.get("spans", [])
            if not isinstance(spans, list):
                continue
            parts: list[str] = []
            sizes: list[float] = []
            for span in spans:
                if not isinstance(span, dict):
                    continue
                parts.append(str(span.get("text", "")))
                size = span.get("size")
                if isinstance(size, int | float) and not isinstance(size, bool):
                    sizes.append(float(size))
            text = "".join(parts).strip()
            if text and sizes:
                lines.append((text, max(sizes)))
    return lines


def _is_heading(text: str, size: float, body_size: float) -> bool:
    return len(text) <= HEADING_MAX_CHARS and size >= body_size + HEADING_SIZE_MARGIN
