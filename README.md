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

It does not implement a neural embedding model, a reviewer interface, XBRL,
anomaly detection, or a database. A model result still requires human review.
OCR text cannot support `Met` until its character error rate is within the
confirmed limit.

## Prerequisites

- Python 3.11 or newer
- PowerShell

No Docker installation is needed or supported in this phase.

## Set up

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optionally copy `.env.example` to `.env` and adjust only approved local
settings. The default API address is `127.0.0.1:8000`.

## Run locally

```powershell
python -m uvicorn jsc_review.api.app:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`; it returns `{"status":"ok"}`.

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
- Verification evidence (2026-10-15, Python 3.15): editable installation
  succeeded; 10 tests passed; Ruff lint and format checks passed; strict mypy
  passed; a live `/health` smoke test returned `{"status":"ok"}`.
- Peer approval: pending review by Role A.

See `docs/open_questions.md` for decisions that require Commission
confirmation. The working requirements register is
`Data/checklists/requirements_register.csv`. It is excluded from version
control because it contains internal checklist text. On 2026-10-19 the
project owner approved 18 proof-of-concept rows, marked `poc_selected=yes`.
Role A still reviews each proposed classification.
