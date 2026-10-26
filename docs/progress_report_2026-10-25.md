# Progress report

Prepared 2026-10-25. Covers work since the owner brief on 2026-10-22.

## Short version

The reviewer app now accepts PDF uploads from the browser and runs the full
local pipeline in the background. All model-written reasons are in Modern
Standard Arabic. Targeted OCR work moved one more finding into `Met`
(`DIS-001` on `217_10216`, page 7) with zero regressions on the twelve
earlier `Met` results. The project passes 145 tests, ruff, and strict mypy
on Python 3.13.15. All processing stays on this machine.

## Where the recommendations stand

- Five reports in `Data/reports/`, 18 selected requirements, 90 recommendations.
- `Met`: 16. Human review is `Not Required` on 13 of them; the remaining 3
  come from the local model (`DIS-058`, `GOV-001`, `DIS-001`) and still
  require human review per the rule.
- `Needs Review`: 74.
- `Not Met`: 0.

Current `Met` recommendations:

| Document | Requirement | Method | Page |
| --- | --- | --- | --- |
| 131_10130 | DIS-017 | deterministic | 53 |
| 131_10130 | DIS-058 | llm | 113 |
| 131_10130 | DIS-062 | deterministic | 36 |
| 131_10130 | DIS-064 | deterministic | 37 |
| 164_10163 | GOV-001 | llm | 60 |
| 164_10163 | DIS-010 | deterministic | 44 |
| 164_10163 | DIS-011 | deterministic | 44 |
| 164_10163 | DIS-017 | deterministic | 99 |
| 1_10000 | DIS-010 | deterministic | 75 |
| 1_10000 | DIS-011 | deterministic | 75 |
| 1_10000 | DIS-062 | deterministic | 102 |
| 1_10000 | DIS-064 | deterministic | 102 |
| 217_10216 | DIS-001 | llm | 7 |
| 217_10216 | DIS-010 | deterministic | 13 |
| 217_10216 | DIS-011 | deterministic | 13 |
| 217_10216 | DIS-064 | deterministic | 3 |

## What shipped since the owner brief

1. **PDF upload from the UI.** New routes under `/reviewer/documents`
   accept a local PDF, validate the magic bytes, save it to
   `Data/reports/{document_id}.pdf`, and launch a daemon-thread worker.
   The worker runs extract → OCR → per-document BM25 → local-model judging
   and appends each finding to the shared findings file. The browser shows
   live stage and per-requirement progress and auto-reloads when the job
   completes.
2. **Arabic-only system reasons.** The local-model prompt now requires
   Modern Standard Arabic for every `reason`. All deterministic reasons
   (headings, threshold checks, OCR notices) were translated and kept
   test-covered.
3. **Arabic text normalization at the local-model boundary.** A new
   `jsc_review.documents.normalization` module folds alef variants, strips
   tashkeel and tatweel, normalizes Arabic-Indic digits, and removes
   invisibles. The passage sent to the model and the quote-on-page check
   both use the normalized surface form. This moved `DIS-001` on
   `217_10216` into `Met`.
4. **Re-OCR of the five pages previously cited only through OCR.** Three
   documents, five pages, rendered at 600 DPI with grayscale, denoise,
   deskew, and Otsu binarize. Indra OCR extracted clean Arabic markdown
   where Tesseract garbled dates and headings. The clean text is overlaid
   on the pipeline so model reasons on those pages are coherent. OCR
   evidence still cannot promote to `Met` until the character error rate
   is measured at 5 percent or lower on a labeled sample.
5. **Reviewer screen.** Already shipped last cycle; the upload bar and
   jobs panel were added in this cycle. The screen reads
   `Data/reports/reviews/findings.json` and writes decisions to
   `data/reviewer/decisions.json`. Both paths stay local and both are
   excluded from version control.

## Enforced rules

- `Met` and `Not Met` require a quote that is present in the embedded page.
- OCR text cannot support `Met` until its character error rate is at or
  below 5 percent on a labeled sample.
- `human_review` is `Not Required` only for deterministic results with
  verified evidence at confidence of at least 0.85. Local-model results
  are always `Required`.
- The local model (Ollama, Qwen3 4B, temperature 0) runs on this machine
  and the host must stay on `127.0.0.1`. Report text is never sent to any
  outside service.
- `Partially Met` is rejected by the schema.

## Quality posture

- Python 3.13.15 across the venv, pyproject, ruff, and mypy settings.
- 145 tests pass. Added this cycle: `test_normalization` (15),
  `test_jobs` (6), `test_pipeline_upload` (12), `test_upload_routes` (7).
- `ruff check` clean on 64 source files; `ruff format --check` clean.
- `mypy` strict clean on `src` and `tests`.
- The runner for the full 90-finding batch completed in ~45 minutes with
  zero errors in the log. Zero regressions compared with the previous
  15 `Met` set.

## Try it locally

1. Start the server:

   ```powershell
   python -m uvicorn jsc_review.api.app:app --host 127.0.0.1 --port 8000
   ```

2. Open `http://127.0.0.1:8000/reviewer/`. The upload bar sits in the
   header. Pick a PDF and press **Upload PDF**. The jobs panel shows live
   Arabic stage messages (`قراءة النص المضمَّن…`, `OCR صفحة …`,
   `حكم DIS-xxx (n/18)`). Expect ~5–10 minutes per report. When the job
   flips to `done`, the findings list reloads automatically.
3. Walk through any `Met` on `217_10216` to see the new Arabic reasons.
4. Compare `131_10130 DIS-058` and `164_10163 GOV-001` with the local
   model's quoted evidence to assess whether the three LLM `Met` results
   are acceptable without further review.

## Open items

- Measure the character error rate for the five re-OCR'd pages against a
  small labeled sample. The policy at `src/jsc_review/review/policy.json`
  sets the OCR threshold at 5 percent. Until a measurement is recorded,
  `supports_verified_evidence()` returns false and OCR evidence stays
  ineligible for `Met`.
- Add newly uploaded documents to the global BM25 and BGE-M3 dense
  indexes before the next batch run. Each upload already builds a
  per-document index that is sufficient for judging that one document.
  The batch runner `Data/reports/_judge_with_model.py` loads the global
  indexes, so an uploaded document that is not in them will return no
  passages during a full re-run.
- Confirm the proposed `requirement_type` and `evaluation_method` on the
  18 selected requirements with Role A.
- Three LLM-written `Met` results still need owner acceptance in the
  reviewer screen.

## Files changed this cycle

Modified: `.gitignore`, `README.md`, `docs/implementation_report.md`,
`docs/work_plan.md`, `pyproject.toml`, `src/jsc_review/api/app.py`,
`src/jsc_review/core/settings.py`, `src/jsc_review/documents/ocr_metrics.py`,
`src/jsc_review/review/findings.py`, `src/jsc_review/review/ollama.py`,
`src/jsc_review/review/prompt_requirement.txt`,
`src/jsc_review/review/thresholds.py`, `tests/test_thresholds.py`.

Added: `src/jsc_review/api/routes/reviewer.py`,
`src/jsc_review/api/routes/uploads.py`, `src/jsc_review/api/static/`
(reviewer HTML), `src/jsc_review/documents/normalization.py`,
`src/jsc_review/documents/pipeline.py`, `src/jsc_review/retrieval/dense.py`,
`src/jsc_review/retrieval/dense_config.json`,
`src/jsc_review/retrieval/embedding.py`, `src/jsc_review/retrieval/hybrid.py`,
`src/jsc_review/retrieval/rerank_config.json`,
`src/jsc_review/retrieval/reranker.py`,
`src/jsc_review/review/registry.py`, `src/jsc_review/reviewer/` (jobs,
models, store, findings), plus eleven test files.

Nothing has been committed. See `Data/owner_review/brief.md` for the
owner-facing summary and `Data/reports/reviews/findings.json` for the full
set of recommendations.
