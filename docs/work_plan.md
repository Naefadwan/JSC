# JSC AI Review PoC — Work Plan

Local-first proof of concept for Jordan Securities Commission employees.
The system recommends. A Commission employee always makes the final
regulatory decision.

Status date: 2026-10-18.

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

The Phase 1 gate is not closed. `poc_selected` is still empty, Role A has
not reviewed classifications, and the review-result schema still accepts
`Partially Met` and requires verified evidence for `Met` only.

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

### 1. Approve the register and select the PoC set

Confirm these columns in `Data/checklists/requirements_register.csv`:

`requirement_id`, `checklist`, `section`, `requirement_text_ar`,
`requirement_text_en`, `requirement_type`, `evaluation_method`,
`poc_selected`, `owner`, `notes`.

Mark `poc_selected` as `yes` on 15 to 20 rows. Those rows are the only
requirements Phases 2 and 3 will run. Leave the other rows blank.

The sample should include both checklists, at least one deterministic
check, and several checks that need the local model. Rows currently
proposed as `deterministic`: `DIS-011` to `DIS-013`, `DIS-018`, and
`DIS-062` to `DIS-066`.

`requirement_type` and `evaluation_method` are proposals. Role A reviews
each selected row before Phase 3 uses them.

### 2. Record the five reports

Fill `Data/templates/report_inventory.csv` (or a local copy kept out of
version control if it contains internal paths or notes) with document ID,
title, language, year, page count, local path, and whether the PDF is
text, scan, or mixed. Store the PDFs outside git. They already live under
`Data/reports/`.

### 3. Align the review-result schema

Decisions 3 and 8 are confirmed and not yet in code:

- Statuses are `Met`, `Not Met`, and `Needs Review`. Remove `Partially Met`.
- `Met` and `Not Met` both require verified evidence.
- `Needs Review` may omit evidence.
- `human_review` may be `Not Required` only for a deterministic check with
  verified evidence and confidence of at least 0.85.

Acceptance: schema and typed models reject the old status and reject
`Met` or `Not Met` without verified evidence. Tests, lint, and types pass.

Phase 1 gate: clean local install starts `/health`; checks pass; schema
policy matches the confirmed decisions; the project owner has approved the
register structure and the 15–20 selected requirements.

---

## Phase 2 — Document processing and knowledge base

Weeks 3 and 4. Owner: Role B. Reviewer: Role A.

1. Detect whether each of the five PDFs is text-based, scanned, or mixed.
2. Extract text and tables with page-level provenance (document ID, page,
   section, chunk ID, extraction method).
3. Run OCR only where needed. Measure Arabic character error rate and word
   error rate. OCR text may support verified evidence only when character
   error rate is 5% or lower. Pages above that target cannot support `Met`.
4. Chunk by section and heading and keep metadata on every chunk.
5. Generate multilingual embeddings and build a local index.
6. Compare candidate local models for later review use. Do not call a
   hosted model.
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
   marked deterministic.
2. Version prompt templates for presence, content-understanding, and
   completeness requirements. Run the local model at temperature 0 and
   require schema-valid JSON.
3. Build retrieval, then analysis, then evidence validation.
4. Confirm every quoted passage exists on the cited page. A schema failure
   is an execution failure, not a partial result.
5. Set confidence and the human-review flag using the 0.85 rule above.
6. Expose the workflow through the API and persist results locally.
7. Run every selected requirement against all five reports.

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

Approve the register columns and mark 15–20 rows `poc_selected=yes`.
Role C can align the schema with decisions 3 and 8 in parallel. Phase 2
starts after that approval.
