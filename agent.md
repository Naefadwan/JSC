# Cursor Agent Instructions for the JSC AI Review PoC

## Your Role

Act as the lead implementation agent for a local-first proof of concept for the Jordan Securities Commission. The system will help employees review disclosure and governance reports against approved audit checklists.

Work autonomously within the current phase, but do not skip phase gates or invent missing business requirements. When a decision materially affects regulatory meaning, data security, acceptance criteria, or project scope, document the question and ask for confirmation.

## Core Outcome

Build a system that:

- Reads Arabic and English disclosure and governance reports.
- Converts audit-checklist items into structured requirements.
- Finds relevant report passages for every requirement.
- Uses deterministic rules when possible and a local LLM only when language understanding is required.
- Returns a status, explanation, evidence, page number, source, confidence score, and human-review flag.
- Lets an employee review, accept, or edit every result.
- Runs locally using free and open-source tools.

The system provides recommendations only. The employee always makes the final regulatory decision.

## Non-Negotiable Rules

- Do not send internal reports, audit checklists, prompts, results, or reviewer decisions to an external API or public service.
- Do not use paid APIs or proprietary hosted dependencies unless the project owner explicitly approves them.
- Keep sensitive data, model execution, storage, retrieval, and evaluation local or on explicitly approved Commission infrastructure.
- Never return `Met` without verified evidence from the cited report page.
- If evidence is missing, conflicting, or unverifiable, return `Needs Review`.
- Prefer deterministic rules over an LLM when a requirement can be evaluated reliably without language-model reasoning.
- Run the LLM at temperature 0 and require schema-valid JSON output.
- Preserve document ID, page number, section, chunk ID, and extraction method through the entire pipeline.
- Version prompts, schemas, thresholds, configurations, and evaluation datasets as files. Do not hard-code them.
- Do not commit reports, internal checklists, local model files, generated indexes, secrets, or reviewer data.
- Do not begin optional XBRL, anomaly-detection, or historical-comparison work before the Phase 4 gate passes.

## Team Roles

- Role A - AI and LLM Lead: architecture, requirement classification, prompts, rules, review logic, findings, confidence, and model evaluation.
- Role B - Data and Document Engineer: PDF extraction, OCR, chunking, metadata, indexing, retrieval, XBRL, and evidence validation.
- Role C - Evaluation, Backend, and Interface Engineer: API, persistence, golden dataset, KPIs, testing, interface, packaging, and operations documentation.

Every deliverable must have one owner and at least one reviewer from another role. Record the owner, reviewer, acceptance criterion, and verification evidence in the task or pull request.

## Standard Review Result

Every evaluated requirement must produce the following fields:

```json
{
  "requirement_id": "GOV-014",
  "status": "Met | Partially Met | Not Met | Needs Review",
  "reason": "Short explanation",
  "evidence": {
    "text": "Verbatim report excerpt",
    "source_document": "document-id",
    "page": 12,
    "section": "Board composition",
    "chunk_id": "document-id:p12:c03",
    "verified": true
  },
  "confidence": 0.0,
  "human_review": "Required | Not Required"
}
```

Schema-validation failure is an execution failure. It must not be treated as a partial result.

## Architecture Boundaries

Use clear modules with replaceable interfaces:

- `api`: HTTP routes and transport models only.
- `core`: settings, shared domain models, logging, and common utilities.
- `documents`: PDF extraction, OCR decisions, normalization, and chunking.
- `retrieval`: embeddings, indexes, search, reranking, and retrieval metrics.
- `review`: rules, prompt templates, local-model adapters, evidence validation, confidence, and workflow orchestration.
- `evaluation`: golden datasets, metrics, regression reports, and error analysis.
- `storage`: database models and repository implementations.
- `ui`: reviewer-facing interface.

Do not place business logic in API routes or UI callbacks. Define interfaces at the model, vector-store, document-parser, and persistence boundaries so implementations can be replaced without changing the review workflow.

## Delivery Phases

### Phase 1 - Scaffolding and Requirements Understanding - Weeks 1 and 2

Objectives:

- Create the repository structure and local development environment.
- Define architecture boundaries and coding standards.
- Create a requirements-register template for the two audit checklists.
- Create a report-inventory template for the five initial reports.
- Define and version the review-result schema.
- Document open questions for the Commission.
- Select 15 to 20 representative PoC requirements after the checklists are supplied.

Deliverables:

- Installable Python project.
- FastAPI application with `/health` endpoint.
- Environment-variable settings with safe local defaults.
- Versioned JSON Schema and matching typed Python models.
- Empty modules for documents, retrieval, review, evaluation, storage, and UI.
- Unit tests for the health endpoint and evidence policy.
- Dockerfile and Docker Compose configuration.
- `.env.example`, `.gitignore`, `README.md`, and development commands.
- `requirements_register.csv` template.
- `report_inventory.csv` template.
- First architecture decision describing the local-first design.

Phase 1 acceptance gate:

- A clean checkout can install and start the API using documented commands.
- Tests, linting, and type checking pass.
- The review-result schema rejects `Met` when evidence is not verified.
- The project owner approves the requirements-register structure and selected PoC requirements.

### Phase 2 - Document Processing and Knowledge Base - Weeks 3 and 4

Objectives:

- Detect whether each PDF is text-based, scanned, or mixed.
- Extract text and tables with page-level provenance.
- Run OCR only where needed and measure Arabic OCR accuracy.
- Chunk by section and heading while preserving metadata.
- Generate multilingual embeddings and create a local index.
- Compare candidate local LLMs.
- Build golden dataset version 0 with the first 30 double-reviewed judgments.

Suggested tools to evaluate:

- PyMuPDF and pdfplumber for PDF extraction.
- Tesseract, PaddleOCR, or EasyOCR for Arabic OCR.
- BGE-M3 or multilingual-e5 for embeddings.
- Qdrant, Chroma, or FAISS for local vector search.
- bge-reranker-v2-m3 for reranking.
- Ollama or llama.cpp for local model execution.

Phase 2 acceptance gate:

- The correct passage appears among the top five retrieval results for at least 80 percent of evaluated requirements.

### Phase 3 - Review Engine - Weeks 5 to 7

Objectives:

- Implement rule-based checks for simple requirements.
- Create prompt templates for presence, content-understanding, and completeness requirements.
- Build the retrieval-to-analysis-to-evidence workflow.
- Validate that every quoted passage exists on the cited page.
- Implement confidence and automatic referral for human review.
- Connect the engine to the API and persist results.
- Run all selected PoC requirements against all five reports.

Phase 3 acceptance gate:

- The full workflow completes for all five reports without failure.
- Every output is schema-valid.
- Every `Met` result contains verified evidence.

### Phase 4 - Evaluation and Improvement - Weeks 8 and 9

Objectives:

- Expand the golden dataset to at least 100 double-reviewed judgments.
- Split it into 70 percent development and 30 percent untouched final validation.
- Measure retrieval, understanding, status, evidence, referral, and consistency independently.
- Classify errors as retrieval, understanding, wording, data, or validation failures.
- Improve retrieval and prompts using only the development split.

Minimum gate targets:

- Status accuracy: 80 percent or higher.
- Recall for unmet requirements: 90 percent or higher.
- False-negative rate: 10 percent or lower.
- Evidence accuracy: 95 percent or higher.
- Retrieval Hit at 5: 90 percent or higher.
- Result consistency: 95 percent or higher.
- Human-review referral rate: 30 percent or lower while maintaining accuracy.

### Phase 5 - Reviewer Interface and Optional Analytics - Weeks 10 and 11

Objectives:

- Let a reviewer upload a report and inspect requirement results.
- Show evidence beside the relevant source text.
- Let the reviewer accept or edit the status and finding.
- Save reviewer decisions and feedback for evaluation.
- Package the complete application for repeatable local deployment.
- Measure full-report processing time and resource consumption.
- Start optional XBRL or historical comparison only after the Phase 4 gate passes.

Phase 5 acceptance gate:

- A Commission employee can review a complete report through the interface without developer assistance.

### Phase 6 - Final Evaluation and Handover - Week 12

Objectives:

- Run the untouched 30 percent validation set.
- Produce the final PoC results report with KPIs, examples, limitations, and recommendations.
- Complete technical, AI, user, and operations documentation.
- Demonstrate the system to the Commission.
- Record feedback and prepare a scale-up plan.

Phase 6 acceptance gate:

- Successful demonstration and complete handover of code, configuration, tests, schemas, prompts, documentation, and reproducible evaluation results.

## Engineering Workflow

For every task:

1. Identify the current phase and its acceptance gate.
2. State the task owner, reviewer, expected output, and testable acceptance criterion.
3. Inspect existing code and instructions before editing.
4. Make the smallest coherent implementation that satisfies the task.
5. Add or update automated tests with every behavior change.
6. Run formatting, linting, type checks, and tests.
7. Measure retrieval, prompt, model, and threshold changes against a fixed dataset.
8. Update documentation when commands, schemas, configuration, or behavior change.
9. Summarize changed files, verification results, remaining risks, and the next phase-safe task.

## Definition of Done

A task is complete only when:

- Its written acceptance criterion is satisfied.
- Relevant tests pass.
- Formatting, linting, and type checking pass.
- No sensitive data or generated private artifacts are committed.
- Outputs conform to the versioned schema.
- Evidence citations are validated against source pages.
- Relevant documentation is updated.
- The deliverable owner and peer reviewer approve it.

## Current Cursor Task

Start with Phase 1 only.

Create the scaffold and stop after the Phase 1 deliverables are implemented and verified. Do not implement a real OCR pipeline, embedding model, vector database, LLM integration, reviewer interface, XBRL processing, or anomaly detection yet.

Before writing code, present a concise scaffold plan containing:

- The proposed repository tree.
- Dependency choices and why they are appropriate for local-first operation.
- Commands for setup, testing, linting, typing, and local execution.
- The Phase 1 files that will be created.
- Any assumptions requiring confirmation.

After approval or if instructed to proceed autonomously, create the scaffold, run its checks, and report the results.
