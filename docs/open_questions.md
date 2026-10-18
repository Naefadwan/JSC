# Open questions for the Commission

Confirmed on 2026-10-15 by the project owner. These decisions apply to the
proof of concept. The employee still makes every final regulatory decision.

## Confirmed

1. **Source delivery.** The two approved audit checklists and five initial
   reports are available now and will be supplied through a local handoff.
2. **Requirement identifiers.** Governance requirements use `GOV-001`.
   Disclosure requirements use `DIS-001`. Each identifier is assigned once
   and is never reused.
3. **Review statuses.** The proof of concept uses `Met`, `Not Met`, and
   `Needs Review`. It does not use `Partially Met`. `Needs Review` remains
   the status when evidence is missing, conflicting, or cannot be verified.
4. **Human review.** `human_review` may be `Not Required` only for a
   deterministic check with verified evidence and confidence of at least
   0.85. `Needs Review` results, local-model results, and results below 0.85
   remain `Required`. The flag does not make the regulatory decision.
5. **Arabic OCR.** Phase 2 measures character error rate and also reports
   word error rate. OCR text may support verified evidence only when the
   character error rate is 5% or lower. Pages above that target cannot
   support `Met`.
6. **Execution environment.** Local execution remains preferred. If this
   workstation cannot run the required model, a third-party hosted model may
   be considered only for content from documents that are already officially
   public. Internal reports, internal checklist content, reviewer data,
   identifiers, results, and confidential prompts must not be sent. The
   fallback remains disabled until the Commission approves the vendor and its
   security, privacy, retention, and model-training terms in writing.
8. **Verified evidence.** `Met` and `Not Met` both require verified evidence.
   `Needs Review` may omit evidence.
9. **Register approval.** The project owner approves the requirements-register
   structure and the 15–20 selected requirements. Role A reviews each
   requirement’s type and whether its evaluation is deterministic or requires
   the local model.

## Still open

7. How long may reports, generated results, and reviewer decisions be retained,
   and who may access them?
10. Which third-party model vendor is approved, and what written security,
    privacy, retention, and no-training controls apply?

## Follow-up

The current review-result schema still accepts `Partially Met`, and it
requires verified evidence for `Met` only. Aligning the schema with decisions
3 and 8 is a separate change.

ADR 0001 continues to prohibit external services until the conditional model
fallback in decision 6 receives the required written approval. Any approved
exception requires a new architecture decision before implementation.
