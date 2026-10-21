# Implementation report

Jordan Securities Commission AI Review proof of concept.
Status date: 2026-10-19. Package version: 0.1.0.

The system recommends only. A Commission employee makes every final
regulatory decision. Processing stays on this workstation. No report,
checklist, prompt, or result is sent to an external service.

## What was used

Runtime, installed from `pyproject.toml`:

| Tool | Role |
| --- | --- |
| Python 3.11 or newer | Application language. Verified on 2026-10-15 with Python 3.15. Python 3.12 was not installed on this machine. |
| FastAPI | Local HTTP application. |
| Uvicorn | Local server. Default address `127.0.0.1:8000`. |
| Pydantic 2 | Typed review-result models. |
| pydantic-settings | Settings from `JSC_` environment variables and an optional `.env`. |
| jsonschema | Validation of the versioned review-result schema. |
| PyMuPDF 1.28 | Local PDF text inspection. It does not perform OCR. |

Development checks:

| Tool | Role |
| --- | --- |
| pytest and httpx | Tests, including the `/health` client. |
| Ruff | Lint and format. |
| mypy, strict | Type checking for `src` and `tests`. |
| setuptools | Editable install of the `jsc-review` package. |

Local source material, stored under `Data/` and excluded from git:

- Governance checklist: `نموذج تقرير حوكمة2024 (3).docx`.
- Annual-report checklist: `نموذج تدقيق بنود التقرير السنوي لعام 2025.pdf`.
- Company list: `اسماء الشركات التي نريد تدقيقها.xlsx`.
- Five reports in `Data/reports/`.

These files were read to build the requirements register. The application
now keeps extracted page text and tables under `Data/reports/extracted/`.

Not used: Docker, a database, OCR, embeddings, a vector index, a local or
hosted language model, XBRL, or a reviewer interface.

## What was implemented

### Application scaffold

- Installable package `jsc-review` under `src/jsc_review`.
- `GET /health` returns `{"status":"ok"}`.
- Settings default to a local environment, host `127.0.0.1`, port `8000`,
  and log level `INFO`. See `.env.example`.
- Logging is configured from those settings.
- Module packages exist for `api`, `core`, `documents`, `retrieval`,
  `review`, `evaluation`, `storage`, and `ui`.

Business logic is not in the API route. The health route only reports that
the process is up.

### Review-result contract

`schemas/review_result.schema.json` and `src/jsc_review/core/models.py`
define one result per requirement:

- `requirement_id`
- `status`
- `reason`
- `evidence` with verbatim text, document, page, section, chunk ID, and
  `verified`
- `confidence` from 0 to 1
- `human_review` of `Required` or `Not Required`

`Met` and `Not Met` are rejected unless evidence is present and `verified`
is true. `Partially Met` is rejected. `human_review` may be `Not Required`
only for a deterministic result with verified evidence and confidence of at
least 0.85. That threshold is in `src/jsc_review/review/policy.json`.
Each result records `evaluation_method` as `deterministic` or `llm`.
The same rules are enforced in `src/jsc_review/review/evidence.py`.

### PDF inspection and page extraction

`src/jsc_review/documents/inspection.py` opens a local PDF, counts pages,
and classifies each page as text when extracted text has at least 30
characters. A file is `text`, `scan`, or `mixed`. Language is `ar`, `en`,
`ar-en`, or blank.

`src/jsc_review/documents/extract.py` saves each page's text and any ruled
tables with document ID, 1-based page, chunk ID, and extraction method.
A page below 30 characters is marked `needs_ocr`. Section is still empty
until heading chunking. The five reports were written to
`Data/reports/extracted/` on 2026-10-19: 992 pages, 295 marked for OCR,
and 1,318 detected tables. Ruled financial pages can be split into many
small tables, so those table boundaries still need review.

On 2026-10-19 Tesseract with Arabic and English read the 295 pages marked
`needs_ocr`. Output is in `Data/reports/ocr/`. 257 of those pages produced
at least 30 characters. A 10-page digital re-render probe, saved in
`Data/reports/ocr/render_probe.json`, scored every page above the 5%
character-error limit. `supports_verified_evidence` stays false on every
OCR page. The 5% limit is versioned in `src/jsc_review/review/policy.json`.

On 2026-10-21 Python 3.13.15 was installed beside 3.15, and Indra OCR 1.1.2
ran locally in `.venv313`. The same 10-page digital re-render is in
`Data/reports/ocr/indra_render_probe.json`. Character error rates are about
18% to 94%. Every page stays above the 5% limit, so Indra text cannot
support `Met` either. The model weights are cached on this machine from
Hugging Face. The five reports were not sent to a hosted OCR service.

On 2026-10-21 the ownership and customer thresholds were applied to the
same findings file. `DIS-010` requires a major-shareholder disclosure and
a holding at or above 5%. `DIS-011` also requires a share count.
`DIS-017` requires a supplier or customer figure at or above 10%, kept
separate from a shareholder table on the same page. The file now has
14 `Met` findings and 76 `Needs Review` findings. Eight Met findings come
from these thresholds. `163_10162_161_07-04-2026_12-47-10` has the
major-shareholder wording on page 74, and the percentage cells on that
page were not fully read, so `DIS-010` and `DIS-011` stay Needs Review.
`131_10130_889_31-03-2026_12-52-11` still has no recognized
major-shareholder wording.

The same day, language findings whose evidence came from OCR were judged
again from the closest embedded page in the top 100 search results.
The model left those judgments Needs Review. Six evidence quotes still
cite OCR text, and each of those is a deterministic heading or the
supplier disclosure found only by OCR.

Heading chunks from embedded text are in `Data/reports/chunks/`: 3,246
chunks, 2,675 with a detected section. The merged corpus in
`Data/reports/chunks_with_ocr/` adds the 257 recovered OCR pages and drops
the short embedded chunks on those same pages. OCR chunks have no section
heading.

The searchable index is `indexes/chunk_bm25.json`: 3,415 chunks and 41,798
terms. It is BM25 over Arabic and English words, saved locally and excluded
from git. A risk-policy query returns a risk-management passage, and
financial-statement queries return financial-statement pages. A neural
embedding index is still blocked because this Python 3.15 environment has
no ONNX Runtime, PyTorch, or FAISS wheel.

The five reports are also recorded in `Data/reports/report_inventory.csv`.
All five are mixed Arabic and English. File metadata did not include a
title or a report year.

On 2026-10-20 local Ollama served `qwen3:4b` at temperature 0 with a
4,096-token context. The model host is `http://127.0.0.1:11434`. It judged
the 65 language requirements. The other 25 stayed on the heading and
threshold rules. `Data/reports/reviews/findings.md` now has 90
recommendations: 6 `Met` and 84 `Needs Review`. Five `Met` results are
embedded statement headings and do not require human review. The sixth is
`DIS-058` on `131_10130_889_31-03-2026_12-52-11`, page 113, from the model,
and it requires human review. No model quote was rejected for failing the
embedded-page check. OCR passages stayed ineligible for `Met`.

### Tests and checks

On 2026-10-15: 10 tests, Ruff, strict mypy, and a live `/health` check passed.
On 2026-10-19, after the local BM25 index: 52 tests passed, Ruff lint and
format passed, and strict mypy passed.

Role A has not yet recorded peer approval of that delivery.

### Requirements work

`Data/checklists/requirements_register.csv` holds 85 requirements:
17 governance (`GOV-001` onward) and 68 disclosure (`DIS-001` onward).
The column structure was approved on 2026-10-19.

Eighteen rows are marked `poc_selected=yes` and copied to
`Data/checklists/selected_requirements.md`:

`GOV-001`, `GOV-006`, `GOV-007`, `GOV-011`, `GOV-015`, `DIS-001`,
`DIS-004`, `DIS-010`, `DIS-011`, `DIS-017`, `DIS-026`, `DIS-034`,
`DIS-039`, `DIS-056`, `DIS-058`, `DIS-062`, `DIS-064`, `DIS-067`.

`requirement_type` and `evaluation_method` on those rows are proposals.
Role A has not reviewed them. The empty templates remain in
`Data/templates/`.

### Decisions recorded

`docs/adr/0001-local-first-design.md` records the local-only design.
`docs/open_questions.md` records the confirmed proof-of-concept rules,
including identifiers, the three statuses, the 0.85 human-review rule, and
the 5% Arabic character-error-rate limit for OCR evidence.
`docs/work_plan.md` is the remaining phase plan.

## What the code does not do yet

- The OCR error sample is a digital re-render, not a scanned-page transcription.
- It does not run a neural embedding model. Python 3.15 has no local wheel
  for ONNX Runtime, PyTorch, or FAISS.
- It does not yet let an employee accept or edit a finding.
- It does not store results or reviewer decisions.
- It does not provide an upload or review screen.
- It does not process XBRL or compare reports over time.

`retrieval`, `evaluation`, `storage`, and `ui` are still boundaries only.

## Still open before Phase 3

Role A has not reviewed the type and evaluation method of the 18 selected
requirements. Questions 7 and 10, retention and any hosted-model vendor,
remain unanswered.
