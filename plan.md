# Referral-rate improvement plan

> Target: drop referral rate from **85.6 %** to **≤ 30 %** (Phase 4 target).
> Scope: no policy change. Everything respects the confirmed rules — Met + Not Met require verified evidence, `human_review = Not Required` only for deterministic + verified + confidence ≥ 0.85, OCR text supports Met only when CER ≤ 5 %.

---

## Baseline (today's numbers)

Measured against `Data/reports/reviews/findings.json`.

| Bucket | Count | Human review |
|---|---|---|
| Met · deterministic · verified · ≥ 0.85 confidence | 13 | **Not Required** |
| Met · LLM | 3 | Required |
| Needs Review · deterministic | 12 | Required |
| Needs Review · LLM | 62 | Required |
| **Total** | **90** | **77 Required (85.6 %)** |

Flips needed to reach 30 %: **77 → 27 Required**, i.e. **50 findings** must become Not Required.

### Needs-Review hit-rate per requirement (today, out of 5 reports)

Already deterministic (keyword list sometimes misses):
- DIS-010 — 2 NR, 3 Met
- DIS-011 — 2 NR, 3 Met
- DIS-017 — 3 NR, 2 Met
- DIS-062 — 3 NR, 2 Met
- DIS-064 — 2 NR, 3 Met

LLM-judged, always Needs Review (5 of 5):
- GOV-006, GOV-007, GOV-011, GOV-015
- DIS-004, DIS-026, DIS-034, DIS-039, DIS-056, DIS-067

LLM-judged, mostly Needs Review (4 of 5):
- GOV-001, DIS-001, DIS-058

---

## Phase A — Fix the 12 deterministic Needs-Review failures

Expected flips: **6 to 10**. Expected new referral rate: **≈ 76 %**.

### Steps

1. **Diagnostic command.** New `python -m jsc_review.review.diagnose_deterministic` that for every current deterministic Needs-Review case prints:
   - the requirement id,
   - the document id,
   - the top-5 hybrid-search hits (page + section + first 120 chars),
   - the current keyword / variant list that was tried.
2. **Expand `HEADING_VARIANTS`** in `src/jsc_review/review/findings.py` with the Arabic variants actually used in the failing reports. Common gaps worth seeding first pass:
   - DIS-062: `الميزانية العمومية`, `الميزانية العامة`, `قائمة المركز المالي الموحدة`
   - DIS-064: `قائمة التدفقات النقدية`, `بيان التدفق النقدي الموحد`
   - DIS-010/011: `كبار المساهمين`, `كبار المالكين`, `قائمة المساهمين`
   - DIS-017: `كبار الموردين`, `كبار العملاء`, `تركز المبيعات`, `تركز المشتريات`
3. **Arabic affix tolerance.** Allow the prefix set `[لبكفوسا]{0,2}` before each anchor (same technique the company autolink module already uses). This catches `للميزانية`, `بالميزانية`, `والميزانية`.
4. **OCR-only page handling.** When the only matched evidence sits on an OCR page, keep the result at `Needs Review` with a clear "evidence found on OCR-only page" reason until Phase D unlocks it. Record the (document, page) pair in a sidecar so Phase D knows which pages matter.
5. Re-run the pipeline on the five reports, re-measure the scorecard, record the new referral rate in `docs/progress_report_<date>.md`.

### Done when
- `python -m jsc_review.review.diagnose_deterministic` lists each failure with its retrieved context.
- Re-running the pipeline flips at least 6 findings to Met + Not Required.
- The scorecard's referral_rate drops to ≤ 80 %.

---

## Phase B — Convert LLM requirements to deterministic

Expected flips (strong + presence): **30 to 50**. Expected new referral rate: **30 % to 53 %**.

### B.1 Strong conversions (6 requirements)

Each has a crisp textual anchor and a structural rule for Met.

| Req | Arabic anchor | English anchor | Rule for Met |
|---|---|---|---|
| GOV-007 | `أعضاء مجلس الإدارة`, `تشكيلة المجلس` | `board of directors`, `board composition` | section present + a members list that distinguishes executive/non-executive and independent/non-independent |
| GOV-011 | `ضابط ارتباط الحوكمة` | `governance liaison officer` | a name follows the heading on the same page |
| GOV-015 | `عدد اجتماعات`, `اجتماعات اللجان` | `number of meetings`, `committee meetings` | heading + numeric column of meeting counts |
| DIS-039 | `أسهم أعضاء مجلس الإدارة` | `shares owned by board members`, `securities owned by directors` | numeric table listing each board member and a share count |
| DIS-058 | `أمور جوهرية قد تؤثر على استمرارية الشركة`, `استمرارية الشركة` | `going concern`, `continuity declaration` | declaration sentence present |
| DIS-067 | `تقرير مدققي الحسابات` + `معايير التدقيق المعتمدة` + `تعليمات الإفصاح` | `independent auditor's report` + `auditing standards` | all three phrases present in the same section |

### B.2 Presence-only conversions (5 requirements)

Weaker but still deterministic. Met when the dedicated section is present; Not Met when a search of the whole report finds nothing.

| Req | Anchors |
|---|---|
| GOV-001 | `سياسة إدارة المخاطر`, `risk management policy` |
| DIS-001 | `أنشطة الشركة الرئيسية`, `principal activities`, `main activities` |
| DIS-026 | `المخاطر`, `risks` section header + at least one risk subheading |
| DIS-034 | `تحليل المركز المالي`, `operating results`, `MD&A` |
| DIS-056 | `معاملات الأطراف ذات العلاقة`, `related party transactions` |

### Steps

1. For each requirement above, add a deterministic entry in the existing deterministic dispatch (same place that currently handles DIS-010, DIS-011, DIS-017, DIS-062, DIS-064). Keep the Arabic anchors in `HEADING_VARIANTS`-style constants.
2. Update the register: set `evaluation_method = deterministic` for all 11 rows.
3. Re-run the pipeline on the five reports.
4. For each requirement, record the hit rate (Met count out of 5). Any requirement below 60 % goes back to Phase A-style keyword expansion before being left as deterministic.

### Done when
- The 11 requirements show as deterministic in the register.
- Re-running the pipeline flips at least 30 findings to Not Required.
- The scorecard's referral_rate drops to ≤ 55 %.

---

## Phase C — Hardest two LLM requirements

Expected flips: **4 to 10**. Expected new referral rate: **26 % to 41 %**.

| Req | Challenge | Approach |
|---|---|---|
| GOV-006 | Governance is mentioned everywhere in a disclosure report | Require a dedicated chapter header `قواعد حوكمة الشركات` or `حوكمة الشركات` within the first 15 pages, plus a sub-list that mentions this instruction's name |
| DIS-004 | Needs an employees-by-activity table, not just a count | Require `عدد الموظفين` or `number of employees` plus a tabular structure (≥ 2 columns, ≥ 3 rows) |

### Done when
- Both requirements have deterministic checks shipped.
- Scorecard shows the realised flip count for each.

---

## Phase D — OCR CER (unlocks pages that currently block Met)

Expected flips: **5 to 15**. Runs in parallel with B and C.

### Steps

1. Human transcription of **10 real scanned pages** chosen from the sidecar built in Phase A step 4 (so the pages we transcribe are the ones that would unlock a Met flip).
2. Store transcriptions as `Data/reports/ocr_gold/<document_id>_p<page>.txt`.
3. Extend `Data/reports/ocr_rerun/cer_measurements.json` to hold per-page CER against the gold transcription.
4. If CER > 5 % on any page, add preprocessing to `src/jsc_review/documents/ocr.py`:
   - binarise (Otsu threshold),
   - deskew,
   - DPI boost to 300 for sub-200 DPI pages,
   - Arabic-tuned PSM mode.
5. Re-measure CER; iterate until every selected page is ≤ 5 %.
6. Flip the per-page "evidence found on OCR-only page" guard in Phase A step 4 for pages that now pass.

### Done when
- At least 10 gold transcriptions exist.
- Every one of those 10 pages shows CER ≤ 5 % in `cer_measurements.json`.
- Re-running the pipeline on affected reports flips the gated findings to Met + Not Required.

---

## End-to-end expected arithmetic

| After phase | Required / 90 | Referral rate |
|---|---|---|
| Today | 77 | 85.6 % |
| Phase A | 68 | 76 % |
| Phase A + B strong (70 % hit) | 47 | 52 % |
| Phase A + B + C | 32 | 36 % |
| Phase A + B + C + D | **24** | **27 %** — under target |

---

## Guardrails

- Policy stays as confirmed by the Commission on 2026-10-15. No change to the 0.85 confidence threshold, no change to the "LLM results stay Required" rule, no change to the OCR CER limit.
- Every new deterministic check must produce verified evidence (quote found on the cited page) before it is allowed to report Met.
- Any requirement where the realised hit rate is below 60 % after Phase B is reverted to LLM so the system does not emit Not Met on noise.
- Changes land requirement-by-requirement, each with a scorecard before/after measurement. The scorecard is the single source of truth.
- Nothing in this plan sends data to a remote service.

---

## Suggested order to run tomorrow

1. Phase A diagnostic command (short, measurable).
2. Phase A keyword expansion + Arabic affix tolerance.
3. Re-run pipeline, record new scorecard.
4. Phase B.1 (strong 6 reqs), one requirement at a time, re-run after each.
5. Phase B.2 (presence 5 reqs), same cadence.
6. Decide whether Phase C is still needed based on the measured rate.
7. Phase D is a parallel human-task track; start the 10-page transcription at the same time as Phase A.
