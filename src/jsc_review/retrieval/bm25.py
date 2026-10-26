"""Local multilingual BM25 index.

Neural embedding runtimes have no Python 3.15 wheels in this environment.
BM25 over Arabic and English tokens is the local index until that runtime exists.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class IndexedChunk:
    """One searchable passage with its provenance."""

    chunk_id: str
    document_id: str
    page: int
    section: str
    text: str
    extraction_method: str


@dataclass(frozen=True)
class SearchHit:
    """A ranked passage. The score is a BM25 relevance score, not a probability."""

    chunk_id: str
    document_id: str
    page: int
    section: str
    text: str
    extraction_method: str
    score: float


@dataclass(frozen=True)
class Bm25Index:
    """In-memory BM25 index. The saved file stays out of version control."""

    k1: float
    b: float
    chunks: tuple[IndexedChunk, ...]
    doc_lengths: tuple[int, ...]
    average_length: float
    document_frequency: dict[str, int]
    postings: dict[str, tuple[tuple[int, int], ...]]

    def search(
        self,
        query: str,
        k: int = 5,
        document_id: str | None = None,
    ) -> tuple[SearchHit, ...]:
        """Return the top passages for a query. An empty query returns nothing."""

        if k < 1:
            return ()
        terms = tokenize(query)
        if not terms or not self.chunks:
            return ()
        scores = [0.0] * len(self.chunks)
        total = len(self.chunks)
        for term in set(terms):
            matches = self.postings.get(term)
            if not matches:
                continue
            doc_frequency = self.document_frequency[term]
            idf = math.log(1.0 + (total - doc_frequency + 0.5) / (doc_frequency + 0.5))
            for doc_index, frequency in matches:
                length_norm = (
                    1.0 - self.b + self.b * self.doc_lengths[doc_index] / self.average_length
                )
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * length_norm
                scores[doc_index] += idf * numerator / denominator
        ranked = sorted(
            (
                (score, index)
                for index, score in enumerate(scores)
                if score > 0.0
                and (document_id is None or self.chunks[index].document_id == document_id)
            ),
            key=lambda item: (-item[0], self.chunks[item[1]].chunk_id),
        )
        return tuple(_hit(self.chunks[index], score) for score, index in ranked[:k])


def tokenize(text: str) -> tuple[str, ...]:
    """Split Arabic and English text into searchable tokens."""

    return tuple(token.casefold() for token in _TOKEN.findall(text))


def build_index(chunks: Sequence[IndexedChunk]) -> Bm25Index:
    """Build an index from passages that already carry provenance."""

    k1, b = _parameters()
    postings: dict[str, list[tuple[int, int]]] = {}
    document_frequency: Counter[str] = Counter()
    lengths: list[int] = []
    for doc_index, chunk in enumerate(chunks):
        counts = Counter(tokenize(chunk.text))
        lengths.append(sum(counts.values()))
        for term, frequency in counts.items():
            postings.setdefault(term, []).append((doc_index, frequency))
            document_frequency[term] += 1
    average = (sum(lengths) / len(lengths)) if lengths else 0.0
    return Bm25Index(
        k1=k1,
        b=b,
        chunks=tuple(chunks),
        doc_lengths=tuple(lengths),
        average_length=average,
        document_frequency=dict(document_frequency),
        postings={term: tuple(matches) for term, matches in postings.items()},
    )


def append_chunks(index: Bm25Index, new_chunks: Sequence[IndexedChunk]) -> Bm25Index:
    """Return a new index that contains every existing chunk plus the new ones.

    Chunks whose ``chunk_id`` already appears in ``index`` are skipped so a
    repeated upload does not inflate the posting lists.
    """

    if not new_chunks:
        return index
    seen_ids = {chunk.chunk_id for chunk in index.chunks}
    additions = [chunk for chunk in new_chunks if chunk.chunk_id not in seen_ids]
    if not additions:
        return index
    combined: tuple[IndexedChunk, ...] = index.chunks + tuple(additions)
    return build_index(combined)


def remove_document(index: Bm25Index, document_id: str) -> tuple[Bm25Index, int]:
    """Return a new index with every chunk for ``document_id`` dropped.

    The second element of the returned tuple is the number of chunks removed,
    which the caller can surface in a deletion summary. When the index does
    not reference ``document_id``, the original index is returned unchanged
    and the count is ``0``.
    """

    if not index.chunks:
        return index, 0
    kept = tuple(chunk for chunk in index.chunks if chunk.document_id != document_id)
    removed = len(index.chunks) - len(kept)
    if removed == 0:
        return index, 0
    rebuilt = build_index(kept)
    return rebuilt, removed


def load_chunk_directory(directory: Path) -> tuple[IndexedChunk, ...]:
    """Read merged chunk JSON files from a local directory."""

    chunks: list[IndexedChunk] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload["chunks"]:
            chunks.append(
                IndexedChunk(
                    chunk_id=str(raw["chunk_id"]),
                    document_id=str(raw["document_id"]),
                    page=int(raw["page"]),
                    section=str(raw["section"]),
                    text=str(raw["text"]),
                    extraction_method=str(raw["extraction_method"]),
                )
            )
    return tuple(chunks)


def save_indexed_chunks(chunks: Sequence[IndexedChunk], directory: Path, document_id: str) -> Path:
    """Write per-document indexed chunks in the ``load_chunk_directory`` format."""

    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    target = directory / f"{document_id}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def save_index(index: Bm25Index, path: Path) -> Path:
    """Write the index beside other generated local artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "bm25-multilingual",
        "k1": index.k1,
        "b": index.b,
        "average_length": index.average_length,
        "document_frequency": index.document_frequency,
        "postings": {
            term: [list(pair) for pair in matches] for term, matches in index.postings.items()
        },
        "chunks": [asdict(chunk) for chunk in index.chunks],
        "doc_lengths": list(index.doc_lengths),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_index(path: Path) -> Bm25Index:
    """Load an index written by ``save_index``."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = tuple(IndexedChunk(**chunk) for chunk in payload["chunks"])
    postings = {
        term: tuple((int(doc_index), int(frequency)) for doc_index, frequency in matches)
        for term, matches in payload["postings"].items()
    }
    return Bm25Index(
        k1=float(payload["k1"]),
        b=float(payload["b"]),
        chunks=chunks,
        doc_lengths=tuple(int(length) for length in payload["doc_lengths"]),
        average_length=float(payload["average_length"]),
        document_frequency={
            term: int(count) for term, count in payload["document_frequency"].items()
        },
        postings=postings,
    )


@lru_cache
def _parameters() -> tuple[float, float]:
    config_path = Path(__file__).with_name("bm25_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return float(config["k1"]), float(config["b"])


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
