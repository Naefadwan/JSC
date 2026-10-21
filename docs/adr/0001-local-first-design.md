# ADR 0001: Local-first design

- Status: Accepted for Phase 1
- Date: 2026-10-15
- Owner: Role C — Evaluation, Backend, and Interface Engineer
- Reviewer: Role A — AI and LLM Lead

## Context

The proof of concept processes Commission reports, audit checklists, prompts,
results, and reviewer decisions. These may contain sensitive or internal data.
The application must provide recommendations while preserving the employee's
authority over every regulatory decision.

## Decision

All processing and storage will run on the user's workstation or explicitly
approved Commission infrastructure. The application will not send inputs or
outputs to hosted APIs or public services. Phase 1 binds the API to
`127.0.0.1` by default and uses no database, model, index, or external service.

The code is separated into `api`, `core`, `documents`, `retrieval`, `review`,
`evaluation`, `storage`, and `ui` packages. Business rules belong in `review`
or shared domain models, not API routes or UI callbacks. Later phases will
introduce explicit parser, local-model, vector-store, and persistence
interfaces before selecting implementations, allowing replacement without
changing the review workflow.

Report provenance—document ID, page, section, chunk ID, and extraction
method—must be retained through future processing. A result cannot be `Met`
or `Not Met` without complete evidence verified against its cited report page.

## Consequences

- Local installation and operation are required.
- Paid or proprietary hosted services are excluded unless explicitly approved.
- Sensitive source files, indexes, models, secrets, and reviewer data are
  ignored by version control.
- Local compute and storage capacity may limit later model choices.
- OCR, retrieval, model, persistence, and UI implementations remain deferred
  to their approved phases.

## Acceptance and verification

The Phase 1 scaffold must install in a Python virtual environment, start the
local `/health` endpoint, pass tests, linting, and type checking, and reject
`Met` unless its evidence is verified. Peer approval is recorded separately;
the named reviewer remains responsible for that approval.
