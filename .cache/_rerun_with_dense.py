"""Re-review every requirement (deterministic + retrieval-fallback) with
BM25 + dense retrieval fused via RRF.

This refreshes findings.json so every row either carries the latest
anchor-based judgment (deterministic requirements) or the strongest
hybrid-retrieval passage (requirements without a deterministic rule).

Nothing in this script contacts the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsc_review.companies.autolink import (
    iter_pages_from_extracted,
    iter_pages_from_ocr,
    merge_pages_with_ocr,
)
from jsc_review.core.settings import get_settings
from jsc_review.documents.cer_measurement import load_measurements
from jsc_review.retrieval import bm25 as bm25_module
from jsc_review.retrieval import dense as dense_module
from jsc_review.retrieval.embedding import (
    SentenceTransformerEncoder,
    embedding_settings,
)
from jsc_review.review.findings import judge_requirement
from jsc_review.review.registry import load_selected_requirements


def _load_pages(extracted_dir: Path, ocr_dir: Path, document_id: str) -> dict[int, str]:
    native_file = extracted_dir / f"{document_id}.json"
    ocr_file = ocr_dir / f"{document_id}.json"
    native: list[tuple[int, str]] = []
    ocr: list[tuple[int, str]] = []
    if native_file.exists():
        native = list(iter_pages_from_extracted(json.loads(native_file.read_text(encoding="utf-8"))))
    if ocr_file.exists():
        ocr = list(iter_pages_from_ocr(json.loads(ocr_file.read_text(encoding="utf-8"))))
    return dict(merge_pages_with_ocr(native, ocr))


def _load_ocr_pages(ocr_dir: Path, document_id: str) -> dict[int, str]:
    path = ocr_dir / f"{document_id}.json"
    if not path.exists():
        return {}
    return dict(iter_pages_from_ocr(json.loads(path.read_text(encoding="utf-8"))))


def main() -> int:
    settings = get_settings()
    print(f"USE_DENSE_AT_REVIEW setting: {settings.use_dense_at_review}")
    requirements = list(load_selected_requirements(settings.requirements_register_path))
    print(f"Total requirements: {len(requirements)}")

    bm25 = bm25_module.load_index(settings.bm25_index_path)
    print(f"BM25 chunks: {len(bm25.chunks)}")

    dense = None
    if settings.use_dense_at_review and settings.dense_index_path.exists():
        print("Loading dense FAISS index + BGE-M3 encoder (first run downloads the model) ...")
        dense_bare = dense_module.load_index(settings.dense_index_path)
        encoder = SentenceTransformerEncoder(embedding_settings())
        dense = dense_bare.with_encoder(encoder)
        print(f"Dense chunks: {len(dense.chunks)}")

    cer_store = load_measurements(settings.cer_measurements_path)
    print(
        f"CER store: {len(cer_store.measurements)} measurements, "
        f"{len(cer_store.anchor_consensus)} consensus records"
    )

    documents = sorted(path.stem for path in settings.extracted_dir.glob("*.json"))
    print(f"Documents: {len(documents)}")

    rows: list[dict[str, object]] = []
    if settings.findings_path.exists():
        rows = json.loads(settings.findings_path.read_text(encoding="utf-8"))

    index: dict[tuple[str, str], int] = {}
    for position, row in enumerate(rows):
        doc = str(row.get("document_id", ""))
        result = row.get("result", {})
        req_id = str(result.get("requirement_id", "")) if isinstance(result, dict) else ""
        if doc and req_id:
            index[(doc, req_id)] = position

    before_review: dict[str, int] = {}
    after_review: dict[str, int] = {}
    before_status: dict[str, int] = {}
    after_status: dict[str, int] = {}

    for doc in documents:
        pages = _load_pages(settings.extracted_dir, settings.ocr_dir, doc)
        ocr_pages = _load_ocr_pages(settings.ocr_dir, doc)
        if not pages:
            print(f"  {doc}: no pages, skipping")
            continue
        for req in requirements:
            key = (doc, req.requirement_id)
            new_result = judge_requirement(
                doc, req, pages, ocr_pages, bm25,
                dense=dense,
                cer_store=cer_store,
            )
            after_review[new_result.human_review] = after_review.get(new_result.human_review, 0) + 1
            after_status[new_result.status] = after_status.get(new_result.status, 0) + 1
            if key in index:
                old = rows[index[key]].get("result", {})
                if isinstance(old, dict):
                    before_review[str(old.get("human_review", ""))] = (
                        before_review.get(str(old.get("human_review", "")), 0) + 1
                    )
                    before_status[str(old.get("status", ""))] = (
                        before_status.get(str(old.get("status", "")), 0) + 1
                    )
                rows[index[key]] = {"document_id": doc, "result": new_result.model_dump()}
            else:
                index[key] = len(rows)
                rows.append({"document_id": doc, "result": new_result.model_dump()})

    tmp = settings.findings_path.with_suffix(settings.findings_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(settings.findings_path)

    print(f"Before statuses: {before_status}")
    print(f"After  statuses: {after_status}")
    print(f"Before human_review: {before_review}")
    print(f"After  human_review: {after_review}")
    required_after = after_review.get("Required", 0)
    total_after = sum(after_review.values())
    if total_after:
        print(f"Referral rate: {required_after}/{total_after} = {required_after / total_after:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
