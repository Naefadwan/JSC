"""Aggregate every known document into one list for the Documents drawer.

The inventory is derived purely from existing local stores:

* ``findings_path``        -- every document the pipeline has judged
* ``extracted_dir``        -- every PDF whose text has been extracted
* ``reports_dir``          -- every PDF the reviewer has uploaded
* ``decisions_path``       -- reviewer decisions per (document, requirement)
* ``document_companies``   -- the current (document -> company) link

The union of document ids keeps a doc visible even when some stores lag
(e.g. a PDF uploaded but not yet judged). Nothing in this module writes to
disk or calls any network -- it reads JSON and filesystem metadata only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from jsc_review.companies.compliance import parse_submission_timestamp
from jsc_review.companies.mapping import DocumentCompanyLink, latest_links
from jsc_review.companies.registry import Company
from jsc_review.reviewer.findings import load_findings
from jsc_review.reviewer.models import ReviewerDecision
from jsc_review.reviewer.store import load_decisions

# Doc-id's that the HTTP layer rejects (traversal, odd characters). Mirror
# the same guard here so a bad entry in findings.json cannot leak into the
# inventory and crash downstream URL building.
_SAFE_DOC_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class FindingCounts(BaseModel):
    """Per-document finding + decision tallies for the Documents drawer."""

    model_config = ConfigDict(extra="forbid")

    total: int = 0
    met: int = 0
    not_met: int = 0
    needs_review: int = 0
    reviewed: int = 0
    unreviewed: int = 0


class DocumentInventoryEntry(BaseModel):
    """One row in the inventory list."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    pdf_present: bool
    extracted: bool
    judged: bool
    submitted_at: datetime | None = None
    counts: FindingCounts = FindingCounts()
    link: DocumentCompanyLink | None = None
    company: Company | None = None


def _tally_finding(counts: FindingCounts, status: str, has_decision: bool) -> None:
    """Update ``counts`` in place for one finding row."""

    counts.total += 1
    if status == "Met":
        counts.met += 1
    elif status == "Not Met":
        counts.not_met += 1
    else:
        # "Needs Review" and any legacy status fall into the review bucket.
        counts.needs_review += 1
    if has_decision:
        counts.reviewed += 1
    else:
        counts.unreviewed += 1


def _collect_doc_ids(
    findings_rows: Sequence[tuple[str, object]],
    extracted_dir: Path | None,
    reports_dir: Path | None,
    links: Mapping[str, DocumentCompanyLink],
) -> list[str]:
    """Return the union of doc ids seen in any store, filtered to safe ids."""

    seen: dict[str, None] = {}  # dict preserves insertion order
    for document_id, _ in findings_rows:
        if _SAFE_DOC_ID.fullmatch(document_id):
            seen.setdefault(document_id, None)
    if extracted_dir is not None and extracted_dir.exists():
        for path in sorted(extracted_dir.glob("*.json")):
            document_id = path.stem
            if _SAFE_DOC_ID.fullmatch(document_id):
                seen.setdefault(document_id, None)
    if reports_dir is not None and reports_dir.exists():
        for path in sorted(reports_dir.glob("*.pdf")):
            document_id = path.stem
            if _SAFE_DOC_ID.fullmatch(document_id):
                seen.setdefault(document_id, None)
    for document_id in links:
        if _SAFE_DOC_ID.fullmatch(document_id):
            seen.setdefault(document_id, None)
    return list(seen.keys())


def _tally_counts_for_docs(
    doc_ids: Sequence[str],
    findings_rows: Sequence[tuple[str, object]],
    decisions_by_pair: Mapping[tuple[str, str], ReviewerDecision],
) -> dict[str, FindingCounts]:
    """Return per-doc counts tallied from findings + decisions."""

    counts: dict[str, FindingCounts] = {doc_id: FindingCounts() for doc_id in doc_ids}
    for document_id, result in findings_rows:
        if document_id not in counts:
            continue
        requirement_id = getattr(result, "requirement_id", "")
        status = getattr(result, "status", "Needs Review")
        has_decision = (document_id, requirement_id) in decisions_by_pair
        _tally_finding(counts[document_id], status, has_decision)
    return counts


def _resolve_company_for_link(
    link: DocumentCompanyLink | None, companies: Sequence[Company]
) -> Company | None:
    """Return the Company whose key, ticker, or numeric_code matches."""

    if link is None:
        return None
    for company in companies:
        if company.key == link.company_key:
            return company
    for company in companies:
        if company.ticker.strip().upper() == link.company_key.strip().upper():
            return company
    for company in companies:
        if company.numeric_code.strip() == link.company_key.strip():
            return company
    return None


def build_inventory(
    *,
    findings_path: Path | None,
    extracted_dir: Path | None,
    reports_dir: Path | None,
    decisions_path: Path | None,
    document_companies_path: Path | None,
    companies: Sequence[Company] = (),
) -> list[DocumentInventoryEntry]:
    """Assemble the per-document inventory for the Documents drawer.

    Any store whose path is ``None`` or missing is treated as empty. The
    result is sorted by submitted_at (newest first), falling back to the
    document id so the order stays stable for docs without a parsed date.
    """

    findings_rows: list[tuple[str, object]] = []
    if findings_path is not None and findings_path.exists():
        findings_rows = list(load_findings(findings_path))

    decisions_list: list[ReviewerDecision] = []
    if decisions_path is not None:
        decisions_list = load_decisions(decisions_path)
    decisions_by_pair: dict[tuple[str, str], ReviewerDecision] = {}
    for decision in decisions_list:
        decisions_by_pair.setdefault(
            (decision.document_id, decision.requirement_id), decision
        )

    links: Mapping[str, DocumentCompanyLink] = (
        latest_links(document_companies_path)
        if document_companies_path is not None
        else {}
    )

    doc_ids = _collect_doc_ids(
        findings_rows, extracted_dir, reports_dir, links
    )
    counts_map = _tally_counts_for_docs(doc_ids, findings_rows, decisions_by_pair)

    entries: list[DocumentInventoryEntry] = []
    for document_id in doc_ids:
        pdf_path = reports_dir / f"{document_id}.pdf" if reports_dir else None
        extracted_path = extracted_dir / f"{document_id}.json" if extracted_dir else None
        link = links.get(document_id)
        company = _resolve_company_for_link(link, companies)
        submitted = parse_submission_timestamp(
            document_id, pdf_path=pdf_path if pdf_path and pdf_path.exists() else None
        )
        entries.append(
            DocumentInventoryEntry(
                document_id=document_id,
                pdf_present=bool(pdf_path and pdf_path.exists()),
                extracted=bool(extracted_path and extracted_path.exists()),
                judged=counts_map[document_id].total > 0,
                submitted_at=submitted,
                counts=counts_map[document_id],
                link=link,
                company=company,
            )
        )

    def sort_key(entry: DocumentInventoryEntry) -> tuple[int, datetime, str]:
        # Newest first; entries with no submission date go to the end.
        if entry.submitted_at is None:
            return (1, datetime.min.replace(tzinfo=UTC), entry.document_id)
        return (0, entry.submitted_at, entry.document_id)

    entries.sort(key=lambda entry: sort_key(entry), reverse=True)
    # Keep "no submission date" entries at the end even after reverse().
    entries.sort(key=lambda entry: 0 if entry.submitted_at is not None else 1)
    return entries
