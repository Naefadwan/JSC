# JSC AI Review PoC

A local-first proof of concept that will help Jordan Securities Commission
employees review disclosure and governance reports against approved audit
checklists.

The system provides recommendations only. A Commission employee always makes
the final regulatory decision.

## Phase 1 scope

This scaffold provides:

- a local FastAPI application with `GET /health`;
- environment-backed settings with localhost defaults;
- a versioned review-result JSON Schema and matching Pydantic models;
- enforcement that `Met` and `Not Met` require verified evidence, that
  `Partially Met` is rejected, and that `human_review` may be `Not Required`
  only for a deterministic result at confidence 0.85 or higher;
- local PDF inspection that records page count and whether a file is text,
  scanned, or mixed;
- page-level text and table extraction with document, page, chunk, and
  extraction-method provenance;
- local Tesseract OCR for pages without embedded text, kept ineligible for
  `Met` until the character error rate is measured;
- heading chunks that keep document, page, section, and chunk ID;
- a local Arabic and English BM25 index over those chunks;
- a local Qwen3 4B judgment through Ollama on this machine, at temperature 0;
- module boundaries for later document, retrieval, review, evaluation,
  storage, and interface work;
- empty requirements-register and report-inventory templates.

It does not yet build XBRL, anomaly detection, or a database. The runtime
is Python 3.13, so local wheels for ONNX Runtime, PyTorch, FAISS, and
sentence-transformers are available through the `retrieval` optional extra.
A model result still requires human review. A local reviewer screen is at
`http://127.0.0.1:8000/reviewer/` for accepting or changing recommendations.
OCR text cannot support `Met` until its character error rate is within the
confirmed limit.

## Prerequisites

- Python 3.13
- PowerShell

No Docker installation is needed or supported in this phase.

## Set up

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the local dense retrieval work, add the optional `retrieval` extra:

```powershell
python -m pip install -e ".[dev,retrieval]"
```

Optionally copy `.env.example` to `.env` and adjust only approved local
settings. The default API address is `127.0.0.1:8000`.

## Run locally

```powershell
python -m uvicorn jsc_review.api.app:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`; it returns `{"status":"ok"}`.
The reviewer screen is at `http://127.0.0.1:8000/reviewer/`. It reads
`Data/reports/reviews/findings.json` and writes reviewer decisions to
`data/reviewer/decisions.json`. Both paths stay on this machine and both
are excluded from version control.

The reviewer screen also accepts PDF uploads: the file is saved to
`Data/reports/{document_id}.pdf`, a background worker extracts embedded
text, OCRs the pages that need it, builds a per-document BM25 index,
judges every selected requirement locally, and appends the new findings
to the shared file. The UI polls `/reviewer/documents` and reloads
findings automatically when the job completes. Uploads stay on this
machine; nothing is sent to any outside service.

## Verify

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
```

## Data handling

Do not commit or send reports, internal checklists, prompts, results, reviewer
decisions, model files, indexes, or secrets to an external service. Sensitive
inputs and generated artifacts remain local or on explicitly approved
Commission infrastructure.

## Phase 1 delivery record

- Owner: Role C — Evaluation, Backend, and Interface Engineer
- Reviewer: Role A — AI and LLM Lead
- Acceptance criterion: a clean local install starts `/health`; tests, linting,
  and type checking pass; the schema and typed model reject `Partially Met`
  and reject `Met` or `Not Met` without verified evidence.
- Verification evidence (2026-10-15, Python 3.15; re-verified 2026-10-21 on
  Python 3.13.15): editable installation succeeded; 70 tests passed; Ruff
  lint and format checks passed; strict mypy passed; a live `/health` smoke
  test returned `{"status":"ok"}`.
- Peer approval: pending review by Role A.

See `docs/open_questions.md` for decisions that require Commission
confirmation. The working requirements register is
`Data/checklists/requirements_register.csv`. It is excluded from version
control because it contains internal checklist text. On 2026-10-19 the
project owner approved 18 proof-of-concept rows, marked `poc_selected=yes`.
Role A still reviews each proposed classification.
