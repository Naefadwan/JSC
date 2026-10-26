"""Disclosure-period deadlines and per-company lateness assessment.

A reviewer configures one or more disclosure periods in a small local JSON
file (``Settings.disclosure_periods_path``). Each period has an id, a short
description, and a ``due_date``. Companies are then classified against the
latest submission linked to them for that period:

* ``submitted_on_time`` — the submission date is on or before the deadline
* ``submitted_late`` — a submission exists but after the deadline
* ``not_submitted`` — no document has been linked to that company for this
  period

The submission date is parsed from the document id, which follows the
upstream filing convention ``..._DD-MM-YYYY_HH-MM-SS``. When no date can be
parsed, the file's modification time on disk is used as a fallback so the
reviewer still sees a status rather than a silent miss.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jsc_review.companies.mapping import DocumentCompanyLink, load_store
from jsc_review.companies.registry import Company, CompanyRegistry

ComplianceStatus = Literal["submitted_on_time", "submitted_late", "not_submitted"]

# Matches the "..._DD-MM-YYYY_HH-MM-SS" suffix on upload filenames.
_SUBMISSION_PATTERN = re.compile(r"_(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2}-\d{2})(?:$|\.|_)")
_DATE_ONLY_PATTERN = re.compile(r"_(\d{2}-\d{2}-\d{4})(?:$|\.|_)")


class DisclosurePeriod(BaseModel):
    """One disclosure period plus its submission deadline."""

    model_config = ConfigDict(extra="forbid")

    period_id: str
    description: str = ""
    due_date: date


class DisclosurePeriodFile(BaseModel):
    """The persistent envelope for every configured period."""

    model_config = ConfigDict(extra="forbid")

    periods: list[DisclosurePeriod] = Field(default_factory=list)


class CompanyComplianceEntry(BaseModel):
    """One company's status for one disclosure period."""

    model_config = ConfigDict(extra="forbid")

    company_key: str
    company_name_ar: str
    short_name_ar: str = ""
    ticker: str = ""
    numeric_code: str = ""
    sector_ar: str = ""
    status: ComplianceStatus
    submitted_at: datetime | None = None
    days_from_due: int | None = None
    document_id: str | None = None


class ComplianceReport(BaseModel):
    """Everything the UI needs to render the compliance dashboard."""

    model_config = ConfigDict(extra="forbid")

    period: DisclosurePeriod
    entries: list[CompanyComplianceEntry]
    submitted_on_time_count: int
    submitted_late_count: int
    not_submitted_count: int

    @property
    def total(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class _SubmissionCandidate:
    document_id: str
    submitted_at: datetime


def load_periods(path: Path) -> list[DisclosurePeriod]:
    """Return every configured disclosure period, or ``[]`` when missing."""

    if not path.exists():
        return []
    try:
        payload = DisclosurePeriodFile.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError):
        return []
    return payload.periods


def save_periods(path: Path, periods: Sequence[DisclosurePeriod]) -> None:
    """Write the disclosure period list atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        DisclosurePeriodFile(periods=list(periods)).model_dump_json(indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


_PERIOD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class PeriodValidationError(ValueError):
    """Raised when a disclosure period payload fails the shared contract."""


def _validated_period(
    *, period_id: str, description: str, due_date: date
) -> DisclosurePeriod:
    pid = period_id.strip()
    if not _PERIOD_ID_PATTERN.match(pid):
        raise PeriodValidationError(
            "period_id must be 1-64 chars from A-Z, a-z, 0-9, '.', '_', or '-'."
        )
    desc = description.strip()
    if len(desc) > 240:
        raise PeriodValidationError("description must be 240 characters or fewer.")
    return DisclosurePeriod(period_id=pid, description=desc, due_date=due_date)


def upsert_period(
    path: Path,
    *,
    period_id: str,
    description: str,
    due_date: date,
    allow_replace: bool = False,
) -> DisclosurePeriod:
    """Create or replace one period and persist the full file atomically.

    When ``allow_replace`` is False and a period with the same id already
    exists, raises :class:`PeriodValidationError`. Callers that intend an
    edit must pass ``allow_replace=True`` explicitly so a POST that collides
    with an existing id cannot silently overwrite.
    """

    validated = _validated_period(
        period_id=period_id, description=description, due_date=due_date
    )
    existing = load_periods(path)
    found = next(
        (i for i, period in enumerate(existing) if period.period_id == validated.period_id),
        None,
    )
    if found is not None and not allow_replace:
        raise PeriodValidationError(
            f"A period with id {validated.period_id!r} already exists."
        )
    if found is None:
        existing.append(validated)
    else:
        existing[found] = validated
    save_periods(path, existing)
    return validated


def delete_period(path: Path, period_id: str) -> bool:
    """Remove one period by id and persist the result; returns True on hit."""

    existing = load_periods(path)
    remaining = [period for period in existing if period.period_id != period_id]
    if len(remaining) == len(existing):
        return False
    save_periods(path, remaining)
    return True


def parse_submission_timestamp(
    document_id: str, *, pdf_path: Path | None = None
) -> datetime | None:
    """Return the submission timestamp parsed from the document id.

    First tries the full ``DD-MM-YYYY_HH-MM-SS`` suffix. Falls back to a
    date-only suffix. If neither parses, uses the PDF's filesystem
    modification time when ``pdf_path`` is given and exists.
    """

    match = _SUBMISSION_PATTERN.search(document_id)
    if match:
        try:
            return datetime.strptime(
                f"{match.group(1)} {match.group(2)}", "%d-%m-%Y %H-%M-%S"
            ).replace(tzinfo=UTC)
        except ValueError:
            pass
    date_match = _DATE_ONLY_PATTERN.search(document_id)
    if date_match:
        try:
            return datetime.strptime(date_match.group(1), "%d-%m-%Y").replace(tzinfo=UTC)
        except ValueError:
            pass
    if pdf_path is not None and pdf_path.exists():
        return datetime.fromtimestamp(pdf_path.stat().st_mtime, tz=UTC)
    return None


def classify_submission(
    submission: datetime | None, due_date: date
) -> ComplianceStatus:
    """Return the per-period classification for a company's latest submission."""

    if submission is None:
        return "not_submitted"
    deadline = datetime(due_date.year, due_date.month, due_date.day, 23, 59, 59, tzinfo=UTC)
    return "submitted_on_time" if submission <= deadline else "submitted_late"


def _resolve_company(
    link: DocumentCompanyLink, registry: CompanyRegistry
) -> Company | None:
    """Return the Company for a link's key, tolerating ticker/numeric-code shapes."""

    direct = registry.by_key(link.company_key)
    if direct is not None:
        return direct
    ticker = registry.by_ticker(link.company_key)
    if ticker is not None:
        return ticker
    numeric = registry.by_numeric_code(link.company_key)
    return numeric


def _best_submission(
    candidates: Iterable[_SubmissionCandidate],
) -> _SubmissionCandidate | None:
    """Return the latest submission timestamp among candidates, or ``None``."""

    best: _SubmissionCandidate | None = None
    for candidate in candidates:
        if best is None or candidate.submitted_at > best.submitted_at:
            best = candidate
    return best


def _entry_for_company(
    company: Company,
    submission: _SubmissionCandidate | None,
    period: DisclosurePeriod,
) -> CompanyComplianceEntry:
    status = classify_submission(
        submission.submitted_at if submission else None, period.due_date
    )
    days_delta: int | None = None
    if submission is not None:
        delta = submission.submitted_at.date() - period.due_date
        days_delta = delta.days
    return CompanyComplianceEntry(
        company_key=company.key,
        company_name_ar=company.name_ar,
        short_name_ar=company.short_name_ar,
        ticker=company.ticker,
        numeric_code=company.numeric_code,
        sector_ar=company.sector_ar,
        status=status,
        submitted_at=submission.submitted_at if submission else None,
        days_from_due=days_delta,
        document_id=submission.document_id if submission else None,
    )


def assess_period(
    period: DisclosurePeriod,
    *,
    registry: CompanyRegistry,
    links: Mapping[str, DocumentCompanyLink],
    reports_dir: Path | None = None,
) -> ComplianceReport:
    """Return the compliance report for one period across every company.

    ``links`` is the latest ``{document_id: link}`` map returned by
    :func:`jsc_review.companies.mapping.latest_links`. ``reports_dir`` is
    optional and only used as a fallback source of submission timestamps
    when a document id does not embed a parseable date.
    """

    by_company: dict[str, list[_SubmissionCandidate]] = {}
    for link in links.values():
        company = _resolve_company(link, registry)
        if company is None:
            continue
        pdf_path = (reports_dir / f"{link.document_id}.pdf") if reports_dir else None
        timestamp = parse_submission_timestamp(link.document_id, pdf_path=pdf_path)
        if timestamp is None:
            continue
        by_company.setdefault(company.key, []).append(
            _SubmissionCandidate(document_id=link.document_id, submitted_at=timestamp)
        )

    entries: list[CompanyComplianceEntry] = []
    on_time = late = missing = 0
    for company in registry.companies:
        best = _best_submission(by_company.get(company.key, ()))
        entry = _entry_for_company(company, best, period)
        entries.append(entry)
        if entry.status == "submitted_on_time":
            on_time += 1
        elif entry.status == "submitted_late":
            late += 1
        else:
            missing += 1

    return ComplianceReport(
        period=period,
        entries=entries,
        submitted_on_time_count=on_time,
        submitted_late_count=late,
        not_submitted_count=missing,
    )


def assess_period_from_disk(
    period: DisclosurePeriod,
    *,
    registry_path: Path,
    mapping_path: Path,
    reports_dir: Path | None = None,
) -> ComplianceReport:
    """Convenience: load registry and mapping from disk and assess the period."""

    from jsc_review.companies.registry import load_registry

    registry = load_registry(registry_path)
    links = load_store(mapping_path).as_mapping()
    return assess_period(period, registry=registry, links=links, reports_dir=reports_dir)
