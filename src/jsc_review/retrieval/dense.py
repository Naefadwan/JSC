"""Local dense retrieval over the same chunks the BM25 index indexes.

The index is a FAISS inner-product search over L2-normalized vectors, which is
cosine similarity. Vectors come from a local multilingual embedding model;
nothing is sent to a remote service.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np
from numpy.typing import NDArray

from jsc_review.retrieval.bm25 import IndexedChunk, SearchHit


class TextEncoder(Protocol):
    """Encode a sequence of texts into a 2D float32 array of vectors."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


@dataclass
class DenseIndex:
    """In-memory FAISS index. The saved file stays out of version control."""

    chunks: tuple[IndexedChunk, ...]
    faiss_index: Any = field(repr=False)
    encoder: TextEncoder | None = field(default=None, repr=False)

    def with_encoder(self, encoder: TextEncoder) -> DenseIndex:
        """Attach the encoder used at search time. The index is still frozen."""

        return DenseIndex(
            chunks=self.chunks,
            faiss_index=self.faiss_index,
            encoder=encoder,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        document_id: str | None = None,
    ) -> tuple[SearchHit, ...]:
        """Return the top passages for a query. An empty query returns nothing."""

        if k < 1 or not query.strip():
            return ()
        if self.encoder is None:
            raise RuntimeError("DenseIndex has no encoder. Call with_encoder first.")
        if not self.chunks:
            return ()
        vector = _normalize(self.encoder.encode([query]).astype(np.float32))
        over = min(max(k * 4, k), len(self.chunks))
        scores, indices = self.faiss_index.search(vector, over)
        hits: list[SearchHit] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0 or index >= len(self.chunks):
                continue
            chunk = self.chunks[index]
            if document_id is not None and chunk.document_id != document_id:
                continue
            hits.append(_hit(chunk, float(score)))
            if len(hits) >= k:
                break
        return tuple(hits)


def build_index(
    chunks: Sequence[IndexedChunk],
    encoder: TextEncoder,
    *,
    batch_size: int = 16,
) -> DenseIndex:
    """Encode every chunk and build a cosine-similarity FAISS index."""

    texts = [chunk.text for chunk in chunks]
    if not texts:
        empty = faiss.IndexFlatIP(1)
        return DenseIndex(chunks=(), faiss_index=empty, encoder=encoder)
    batches = [
        encoder.encode(texts[start : start + batch_size]).astype(np.float32)
        for start in range(0, len(texts), batch_size)
    ]
    vectors = _normalize(np.vstack(batches))
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return DenseIndex(
        chunks=tuple(chunks),
        faiss_index=index,
        encoder=encoder,
    )


def save_index(index: DenseIndex, index_path: Path) -> Path:
    """Write the FAISS index and the chunk metadata beside it."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index.faiss_index, str(index_path))
    meta_path = index_path.with_suffix(index_path.suffix + ".meta.json")
    payload = {
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "page": chunk.page,
                "section": chunk.section,
                "text": chunk.text,
                "extraction_method": chunk.extraction_method,
            }
            for chunk in index.chunks
        ]
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return index_path


def load_index(index_path: Path) -> DenseIndex:
    """Read a dense index written by ``save_index``. No encoder is attached."""

    faiss_index = faiss.read_index(str(index_path))
    meta_path = index_path.with_suffix(index_path.suffix + ".meta.json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = tuple(IndexedChunk(**entry) for entry in payload["chunks"])
    return DenseIndex(chunks=chunks, faiss_index=faiss_index)


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


def _hit(chunk: IndexedChunk, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        page=chunk.page,
        section=chunk.section,
        text=chunk.text,
        extraction_method=chunk.extraction_method,
        score=score,
    )
