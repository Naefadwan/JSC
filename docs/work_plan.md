# JSC AI Review PoC — Work Plan

Local-first proof of concept for Jordan Securities Commission employees.
The system recommends. A Commission employee always makes the final
regulatory decision.

Status date: 2026-10-19.

## Current position

Phase 1 scaffolding is implemented and verified (2026-10-15, Python 3.15):
editable install, 10 tests, Ruff, strict mypy, and a live `/health` check.

Also done:

- Architecture decision `docs/adr/0001-local-first-design.md`.
- Open questions 1–6, 8, and 9 confirmed in `docs/open_questions.md`.
- Requirements register populated locally: 17 governance items (`GOV-001`
  onward) and 68 disclosure items (`DIS-001` onward), 85 rows in
  `Data/checklists/requirements_register.csv`. That file stays out of
  version control.
- Five source reports are on disk under `Data/reports/`.

The project owner approved the register columns and the 18-row
proof-of-concept set on 2026-10-19. On the same day the review-result schema
was aligned with the confirmed status, evidence, and human-review rules.
Role A has not reviewed the 18 classifications. The five reports are being
recorded from the local PDFs.

## How work is accepted

Every task has one owner, one reviewer from another role, a written
acceptance criterion, and verification evidence.

- Role A — AI and LLM Lead: architecture, classification, prompts, rules,
  findings, confidence, and model evaluation.
- Role B — Data and Document Engineer: PDF extraction, OCR, chunking,
  metadata, indexing, retrieval, and evidence validation.
- Role C — Evaluation, Backend, and Interface Engineer: API, persistence,
  golden dataset, KPIs, testing, interface, packaging, and operations.

Do not send reports, checklists, prompts, results, or reviewer decisions to
an external service. Do not start XBRL, anomaly detection, or historical
comparison before the Phase 4 gate passes.

---

## Close Phase 1 — remaining gate

Owner: project owner for selection. Role A reviews classification.
Role C implements the schema alignment.

### 1. Proof-of-concept set

Selected on 2026-10-19 in `Data/checklists/requirements_register.csv`.
These 18 rows are the only requirements Phases 2 and 3 will run:

`GOV-001`, `GOV-006`, `GOV-007`, `GOV-011`, `GOV-015`, `DIS-001`,
`DIS-004`, `DIS-010`, `DIS-011`, `DIS-017`, `DIS-026`, `DIS-034`,
`DIS-039`, `DIS-056`, `DIS-058`, `DIS-062`, `DIS-064`, `DIS-067`.

`requirement_type` and `evaluation_method` on those rows are proposals.
Role A reviews each selected row before Phase 3 uses them.

### 2. Report inventory

Recorded on 2026-10-19 in `Data/reports/report_inventory.csv`, which stays
out of version control with the PDFs. All five files are mixed: they contain
both extractable text and pages below the 30-character text threshold.
Languages detected from extracted text are Arabic and English. PDF metadata
did not include a title or a report year. Non-text pages range from 14 to
159, so Phase 2 still needs OCR on those pages.

### 3. Review-result schema

Aligned on 2026-10-19. Statuses are `Met`, `Not Met`, and `Needs Review`.
`Met` and `Not Met` require verified evidence. `Needs Review` may omit
evidence. `human_review` may be `Not Required` only for a deterministic
result with verified evidence and confidence of at least 0.85. Each result
also records `evaluation_method` as `deterministic` or `llm`.

Phase 1 gate: clean local install starts `/health`; checks pass; schema
policy matches the confirmed decisions; the project owner has approved the
register structure and the 15–20 selected requirements.

---

## Phase 2 — Document processing and knowledge base

Weeks 3 and 4. Owner: Role B. Reviewer: Role A.

1. Detect whether each of the five PDFs is text-based, scanned, or mixed.
   Done on 2026-10-19: all five are mixed.
2. Extract text and tables with page-level provenance (document ID, page,
   section, chunk ID, extraction method). Text and ruled tables were
   extracted on 2026-10-19 into `Data/reports/extracted/`. Section is still
   blank; heading assignment is step 4. Pages with fewer than 30 characters
   are marked `needs_ocr`.
3. Run OCR only where needed. On 2026-10-19 Tesseract `ara+eng` read the
   295 pages marked `needs_ocr`. Results are in `Data/reports/ocr/`. 257
   pages produced at least 30 characters. A 10-page digital re-render
   probe is in `Data/reports/ocr/render_probe.json`. Every page scored
   above the 5% character-error limit (about 32% to 100%). That probe
   compares a rendered text page with its embedded text, so reading order
   can inflate the rate. It is not a transcription of a scanned page.
   On 2026-10-21 the same 10 pages were scored with local Indra OCR 1.1.2
   on Python 3.13.15. Results are in `Data/reports/ocr/indra_render_probe.json`.
   Every page is still above 5% (about 18% to 94%). OCR text still cannot
   support `Met`.
4. Chunk by section and heading and keep metadata on every chunk. Embedded
   heading chunks are in `Data/reports/chunks/`. The merged set, with the
   257 recovered OCR pages included, is in `Data/reports/chunks_with_ocr/`.
   Those OCR chunks have no section heading.
5. Generate multilingual embeddings and build a local index. On 2026-10-19
   a local BM25 index of the 3,415 merged chunks was written to
   `indexes/chunk_bm25.json`. It searches Arabic and English tokens.
   ONNX Runtime, PyTorch, and FAISS have no Python 3.15 wheels here, so
   BGE-M3 and multilingual-e5 are not installed yet. The 80% top-five
   retrieval gate is not measured.
6. Compare candidate local models for later review use. Do not call a
   hosted model. On 2026-10-20 `qwen3:4b` was pulled through local Ollama
   and used at temperature 0. The context window is 4,096 tokens. A smaller
   Qwen3 remains the fallback if this machine runs out of memory.
7. Build golden dataset version 0: the first 30 double-reviewed judgments.

Tools to evaluate, all local: PyMuPDF and pdfplumber; Tesseract, PaddleOCR,
or EasyOCR; BGE-M3 or multilingual-e5; Qdrant, Chroma, or FAISS;
bge-reranker-v2-m3; Ollama or llama.cpp.

Gate: for the selected requirements, the correct passage is in the top five
retrieval results at least 80% of the time.

---

## Phase 3 — Review engine

Weeks 5 to 7. Owner: Role A for rules and prompts. Role C for API and
persistence. Reviewer from the other role on each deliverable.

1. Implement deterministic checks for the selected requirements Role A
   marked deterministic. On 2026-10-21 the ownership and customer
   thresholds were added. `DIS-010` is Met when the embedded page names
   major shareholders and a holding is at or above 5%. `DIS-011` also
   requires a share count. `DIS-017` is Met when a supplier or customer
   disclosure has a figure at or above 10%, and that figure is closer to
   the customer wording than to a shareholder table. OCR text stays
   ineligible for Met. Unfamiliar wording stays Needs Review.
2. Version prompt templates for presence, content-understanding, and
   completeness requirements. Run the local model at temperature 0 and
   require schema-valid JSON.
3. Build retrieval, then analysis, then evidence validation.
4. Confirm every quoted passage exists on the cited page. A schema failure
   is an execution failure, not a partial result.
5. Set confidence and the human-review flag using the 0.85 rule above.
6. Expose the workflow through the API and persist results locally.
7. Run every selected requirement against all five reports. On 2026-10-20
   the local run wrote 90 schema-checked recommendations to
   `Data/reports/reviews/findings.md`: 6 `Met` and 84 `Needs Review`.
   Five `Met` results are recognized statement headings. One `Met` result
   is a local-model judgment, `DIS-058` on
   `131_10130_889_31-03-2026_12-52-11`, page 113, and it still requires
   human review. The API does not persist these results yet.

Gate: the workflow completes for all five reports; every output is
schema-valid; every `Met` and every `Not Met` has verified evidence.

---

## Phase 4 — Evaluation and improvement

Weeks 8 and 9. Owner: Role C for metrics and the dataset split. Role A for
prompt and retrieval changes. Reviewer: the other role.

1. Expand the golden dataset to at least 100 double-reviewed judgments.
2. Split 70% development and 30% untouched final validation. Do not tune
   on the validation split.
3. Measure retrieval, understanding, status, evidence, referral, and
   consistency separately.
4. Classify each error as retrieval, understanding, wording, data, or
   validation.
5. Improve retrieval and prompts using only the development split.

Gate targets:

| Measure | Target |
| --- | --- |
| Status accuracy | 80% or higher |
| Recall for unmet requirements | 90% or higher |
| False-negative rate | 10% or lower |
| Evidence accuracy | 95% or higher |
| Retrieval Hit at 5 | 90% or higher |
| Result consistency | 95% or higher |
| Human-review referral rate | 30% or lower, without dropping accuracy |

---

## Phase 5 — Reviewer interface and packaging

Weeks 10 and 11. Owner: Role C. Reviewer: Role A.

1. Let a reviewer upload a report and inspect each requirement result.
2. Show the cited evidence beside the source text.
3. Let the reviewer accept or edit the status and the finding.
4. Save reviewer decisions and feedback locally for later evaluation.
5. Package the application for repeatable local deployment.
6. Measure full-report processing time and resource use.

Optional XBRL or historical comparison starts only after the Phase 4 gate
passes, and only with a new approved scope.

Gate: a Commission employee can review a complete report in the interface
without developer assistance.

---

## Phase 6 — Final evaluation and handover

Week 12. Owner: Role C for the results report and operations docs. Role A
for model and prompt handover. Role B for document-pipeline handover.

1. Run the untouched 30% validation set once.
2. Write the PoC results report: KPIs, examples, limitations, and
   recommendations.
3. Finish technical, AI, user, and operations documentation.
4. Demonstrate the system to the Commission.
5. Record feedback and write a scale-up plan.

Gate: demonstration completed, and handover includes code, configuration,
tests, schemas, prompts, documentation, and reproducible evaluation results.

---

## Still blocking later work

These stay open. They do not block closing Phase 1 or starting document
extraction. They block retention design and any hosted-model fallback.

7. How long may reports, generated results, and reviewer decisions be
   retained, and who may access them?
10. Which third-party model vendor, if any, is approved, and what written
    security, privacy, retention, and no-training terms apply?

A hosted model remains disabled. It may be considered only for content that
is already officially public, and only after written Commission approval and
a new architecture decision. Internal reports, checklist text, reviewer
data, identifiers, results, and confidential prompts must not be sent.

## Immediate next step

A Commission employee reviews `Data/reports/reviews/findings.md`.
On 2026-10-21 the file has 14 `Met` findings and 76 `Needs Review`
findings. Language findings that had been judged from OCR text were
judged again from the closest embedded page. Those judgments stayed
`Needs Review`. Six evidence quotes still come from OCR, all of them
deterministic statement headings or the supplier disclosure on
`163_10162_161_07-04-2026_12-47-10`. That report's major-shareholder
wording is on page 74, and the percentage cells were not fully read.
`131_10130_889_31-03-2026_12-52-11` still has no recognized
major-shareholder wording. Confirm the model `Met` on `DIS-058`, page
113 of that report, and the two statement headings that share page 102
of `1_10000_599_29-03-2026_14-49-00`. The 80% top-five retrieval gate
and the first 30 double-reviewed judgments are still open.
