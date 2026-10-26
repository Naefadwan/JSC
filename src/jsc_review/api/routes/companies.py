"""Company registry, document-company linkage, and compliance endpoints."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from jsc_review.companies.autolink import (
    CompanyMatch,
    decide_autolink,
    iter_pages_from_extracted,
    iter_pages_from_ocr,
    merge_pages_with_ocr,
    rank_candidates_with_fallback,
)
from jsc_review.companies.compliance import (
    ComplianceReport,
    DisclosurePeriod,
    PeriodValidationError,
    assess_period,
    delete_period,
    load_periods,
    upsert_period,
)
from jsc_review.companies.mapping import (
    DocumentCompanyLink,
    clear_link,
    get_link,
    latest_links,
    set_link,
)
from jsc_review.companies.registry import (
    Company,
    CompanyRegistry,
    load_registry,
)
from jsc_review.core.settings import get_settings
from jsc_review.documents.inventory import (
    DocumentInventoryEntry,
    build_inventory,
)

_LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/reviewer", tags=["reviewer"])

_SAFE_DOC_ID = re.compile(r"^[A-Za-z0-9._-]+$")

# Serializes read-modify-write of the disclosure-periods JSON file so two
# concurrent edits from the UI cannot race on the same store.
_PERIODS_WRITE_LOCK = threading.Lock()


class CompanyListResponse(BaseModel):
    """Full company registry returned to the UI picker."""

    model_config = ConfigDict(extra="forbid")

    companies: list[Company]
    total: int
    source_excel: str = ""


class LinkRequest(BaseModel):
    """Reviewer-provided payload when assigning a company to a document."""

    model_config = ConfigDict(extra="forbid")

    company_key: str
    linked_by: str = ""


class LinkResponse(BaseModel):
    """The link plus the resolved company (for one-call UI rendering)."""

    model_config = ConfigDict(extra="forbid")

    link: DocumentCompanyLink | None = None
    company: Company | None = None


class LinksListResponse(BaseModel):
    """Every current link, keyed by document id."""

    model_config = ConfigDict(extra="forbid")

    links: dict[str, DocumentCompanyLink] = Field(default_factory=dict)


class PeriodsResponse(BaseModel):
    """Every configured disclosure period."""

    model_config = ConfigDict(extra="forbid")

    periods: list[DisclosurePeriod]


def _ensure_safe(document_id: str) -> None:
    if not _SAFE_DOC_ID.fullmatch(document_id):
        raise HTTPException(status_code=422, detail="Invalid document id")


# ---------------------------------------------------------------- registry ---


@router.get("/companies", response_model=CompanyListResponse)
def list_companies() -> CompanyListResponse:
    """Return the full registry parsed from the local Excel cache."""

    settings = get_settings()
    from jsc_review.companies.registry import load_registry_payload

    payload = load_registry_payload(settings.companies_registry_path)
    return CompanyListResponse(
        companies=payload.companies,
        total=len(payload.companies),
        source_excel=payload.source_excel,
    )


# ------------------------------------------------- document-company linkage ---


@router.get("/documents/by-id/{document_id}/company", response_model=LinkResponse)
def get_document_company(document_id: str) -> LinkResponse:
    """Return the current (document -> company) link, if any."""

    _ensure_safe(document_id)
    settings = get_settings()
    link = get_link(settings.document_companies_path, document_id)
    if link is None:
        return LinkResponse()
    registry = load_registry(settings.companies_registry_path)
    company = (
        registry.by_key(link.company_key)
        or registry.by_ticker(link.company_key)
        or registry.by_numeric_code(link.company_key)
    )
    return LinkResponse(link=link, company=company)


@router.put("/documents/by-id/{document_id}/company", response_model=LinkResponse)
def set_document_company(document_id: str, request: LinkRequest) -> LinkResponse:
    """Attach ``document_id`` to the given company key (ticker or numeric code)."""

    _ensure_safe(document_id)
    settings = get_settings()
    registry = load_registry(settings.companies_registry_path)
    company = (
        registry.by_key(request.company_key)
        or registry.by_ticker(request.company_key)
        or registry.by_numeric_code(request.company_key)
    )
    if company is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown company key: {request.company_key}. "
                "Rebuild the registry if the Excel changed."
            ),
        )
    link = set_link(
        settings.document_companies_path,
        document_id=document_id,
        company_key=company.key,
        linked_by=request.linked_by,
    )
    return LinkResponse(link=link, company=company)


@router.delete("/documents/by-id/{document_id}/company", response_model=LinkResponse)
def clear_document_company(document_id: str) -> LinkResponse:
    """Remove the (document -> company) link if one exists."""

    _ensure_safe(document_id)
    settings = get_settings()
    clear_link(settings.document_companies_path, document_id=document_id)
    return LinkResponse()


@router.get("/document-companies", response_model=LinksListResponse)
def list_document_companies() -> LinksListResponse:
    """Return every (document -> company) link, keyed by document id."""

    settings = get_settings()
    return LinksListResponse(links=dict(latest_links(settings.document_companies_path)))


# ----------------------------------------------------------------- periods ---


@router.get("/disclosure-periods", response_model=PeriodsResponse)
def list_periods() -> PeriodsResponse:
    """Return every configured disclosure period in Settings-driven order."""

    settings = get_settings()
    return PeriodsResponse(periods=load_periods(settings.disclosure_periods_path))


class PeriodCreateRequest(BaseModel):
    """Payload for creating or replacing one disclosure period."""

    model_config = ConfigDict(extra="forbid")

    period_id: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=240)
    due_date: date
    replace: bool = Field(
        default=False,
        description="Set true to overwrite an existing period with the same id.",
    )


@router.post("/disclosure-periods", response_model=DisclosurePeriod, status_code=201)
def create_period(request: PeriodCreateRequest) -> DisclosurePeriod:
    """Add or replace a disclosure period and persist atomically."""

    settings = get_settings()
    with _PERIODS_WRITE_LOCK:
        try:
            return upsert_period(
                settings.disclosure_periods_path,
                period_id=request.period_id,
                description=request.description,
                due_date=request.due_date,
                allow_replace=request.replace,
            )
        except PeriodValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/disclosure-periods/{period_id}", status_code=204)
def remove_period(period_id: str) -> None:
    """Delete one disclosure period; 404 when no such id exists."""

    settings = get_settings()
    with _PERIODS_WRITE_LOCK:
        removed = delete_period(settings.disclosure_periods_path, period_id)
    if not removed:
        raise HTTPException(
            status_code=404, detail=f"Unknown period id: {period_id}."
        )


# -------------------------------------------------------------- compliance ---


@router.get("/compliance", response_model=ComplianceReport)
def compliance_report(
    period_id: str = Query(..., description="Disclosure period id (e.g. Q4-2025)."),
) -> ComplianceReport:
    """Classify every company as on-time, late, or missing for ``period_id``."""

    settings = get_settings()
    periods = load_periods(settings.disclosure_periods_path)
    match = next((period for period in periods if period.period_id == period_id), None)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown period id: {period_id}. "
                f"Configure it in {settings.disclosure_periods_path}."
            ),
        )
    registry = load_registry(settings.companies_registry_path)
    links = latest_links(settings.document_companies_path)
    return assess_period(
        match,
        registry=registry,
        links=links,
        reports_dir=settings.reports_dir,
    )


# ----------------------------------------------------- autolink / detect ---


class CandidatePayload(BaseModel):
    """One ranked suggestion for the UI."""

    model_config = ConfigDict(extra="forbid")

    company: Company
    score: float
    hits: int
    matched_terms: list[str] = Field(default_factory=list)


class DetectResponse(BaseModel):
    """What :func:`detect_company_for_document` returns to the UI."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    candidates: list[CandidatePayload] = Field(default_factory=list)
    reason: str  # "accepted", "below_threshold", "ambiguous", "no_match", "no_text"
    threshold: float
    accepted_key: str | None = None  # key the policy would auto-link to
    linked_at: DocumentCompanyLink | None = None  # present when ?apply=true linked it
    already_linked: DocumentCompanyLink | None = None  # when a manual link exists


class BulkDetectEntry(BaseModel):
    """One document's detection result in the bulk response."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    reason: str
    accepted_key: str | None = None
    candidates: list[CandidatePayload] = Field(default_factory=list)
    linked_at: DocumentCompanyLink | None = None
    already_linked: DocumentCompanyLink | None = None


class BulkDetectResponse(BaseModel):
    """Aggregate result of a bulk detect run."""

    model_config = ConfigDict(extra="forbid")

    threshold: float
    scanned: int
    linked: int
    skipped_existing: int
    ambiguous: int
    below_threshold: int
    no_match: int
    entries: list[BulkDetectEntry] = Field(default_factory=list)


def _match_to_payload(match: CompanyMatch) -> CandidatePayload:
    return CandidatePayload(
        company=match.company,
        score=round(match.score, 4),
        hits=match.hits,
        matched_terms=list(match.matched_terms),
    )


def _load_json_payload(path: Path) -> dict[str, object] | None:
    """Return a parsed JSON dict or ``None`` when missing or malformed."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_pages_for_doc(
    extracted_dir: Path, ocr_dir: Path, document_id: str
) -> tuple[tuple[int, str], ...]:
    """Return merged (native + OCR) pages for ``document_id``.

    Native extraction wins where it has usable text. OCR fills in for pages
    that were scanned, so the detector still sees the issuer name on an
    image-only cover page. Returns an empty tuple when both sources are
    missing or empty so callers can emit a clean ``no_text`` reason.
    """

    native_payload = _load_json_payload(extracted_dir / f"{document_id}.json")
    ocr_payload = _load_json_payload(ocr_dir / f"{document_id}.json")
    native_pages = (
        list(iter_pages_from_extracted(native_payload))
        if native_payload is not None
        else []
    )
    ocr_pages = (
        list(iter_pages_from_ocr(ocr_payload))
        if ocr_payload is not None
        else []
    )
    merged = merge_pages_with_ocr(native_pages, ocr_pages)
    return tuple(merged)


# How far the "deep" fallback scan goes when the quick scan returns no match.
# This is intentionally generous because it only runs on docs the quick path
# cannot resolve, and only during on-demand detection (not during upload).
_DEEP_SCAN_PAGES = 40


def _detect_for_document(
    document_id: str,
    *,
    registry: CompanyRegistry,
    extracted_dir: Path,
    ocr_dir: Path,
    scan_pages: int,
    threshold: float,
    min_lead_margin: float,
) -> tuple[tuple[CompanyMatch, ...], str, CompanyMatch | None]:
    """Return (candidates, reason, accepted) for one document."""

    pages = _load_pages_for_doc(extracted_dir, ocr_dir, document_id)
    if not pages:
        return (), "no_text", None
    result = rank_candidates_with_fallback(
        pages, registry, scan_pages=scan_pages, deep_scan_pages=_DEEP_SCAN_PAGES
    )
    decision = decide_autolink(
        result.candidates,
        threshold=threshold,
        min_lead_margin=min_lead_margin,
        from_deep_scan=result.deep,
    )
    return decision.candidates, decision.reason, decision.accepted


@router.get(
    "/documents/by-id/{document_id}/company/detect",
    response_model=DetectResponse,
)
def detect_company_for_document(
    document_id: str,
    apply: bool = Query(False, description="When true, write the link if accepted."),
) -> DetectResponse:
    """Return ranked candidates for one document. Optionally persist the top one.

    Reads from the already-extracted JSON, so no PDF re-parse. When
    ``apply=true`` and the policy accepts, the top candidate is written to
    the mapping store with ``linked_by='auto'``.
    """

    _ensure_safe(document_id)
    settings = get_settings()
    registry = load_registry(settings.companies_registry_path)
    existing = get_link(settings.document_companies_path, document_id)
    candidates, reason, accepted = _detect_for_document(
        document_id,
        registry=registry,
        extracted_dir=settings.extracted_dir,
        ocr_dir=settings.ocr_dir,
        scan_pages=settings.auto_link_scan_pages,
        threshold=settings.auto_link_confidence_threshold,
        min_lead_margin=settings.auto_link_min_lead_margin,
    )
    linked_at: DocumentCompanyLink | None = None
    if apply and accepted is not None and existing is None:
        linked_at = set_link(
            settings.document_companies_path,
            document_id=document_id,
            company_key=accepted.company.key,
            linked_by="auto",
        )
    return DetectResponse(
        document_id=document_id,
        candidates=[_match_to_payload(match) for match in candidates],
        reason=reason,
        threshold=settings.auto_link_confidence_threshold,
        accepted_key=accepted.company.key if accepted is not None else None,
        linked_at=linked_at,
        already_linked=existing,
    )


@router.post(
    "/documents/companies/detect-all",
    response_model=BulkDetectResponse,
)
def detect_companies_for_all(
    apply: bool = Query(False, description="When true, auto-link every accepted result."),
    include_linked: bool = Query(
        False,
        description="When true, also re-detect documents that already have a manual link.",
    ),
) -> BulkDetectResponse:
    """Scan every extracted document and return per-doc suggestions.

    Reviewer uses this to backfill the existing docs. ``apply=true`` is the
    one-click bulk mode: the strict-with-suggestions policy writes a link
    for every unambiguous match and leaves the ambiguous ones for a human.
    """

    settings = get_settings()
    registry = load_registry(settings.companies_registry_path)
    extracted_dir = settings.extracted_dir
    entries: list[BulkDetectEntry] = []
    linked = skipped_existing = ambiguous = below_threshold = no_match = 0

    if not extracted_dir.exists():
        return BulkDetectResponse(
            threshold=settings.auto_link_confidence_threshold,
            scanned=0,
            linked=0,
            skipped_existing=0,
            ambiguous=0,
            below_threshold=0,
            no_match=0,
            entries=[],
        )

    for path in sorted(extracted_dir.glob("*.json")):
        document_id = path.stem
        existing = get_link(settings.document_companies_path, document_id)
        if existing is not None and not include_linked:
            skipped_existing += 1
            entries.append(
                BulkDetectEntry(
                    document_id=document_id,
                    reason="already_linked",
                    accepted_key=existing.company_key,
                    already_linked=existing,
                )
            )
            continue

        candidates, reason, accepted = _detect_for_document(
            document_id,
            registry=registry,
            extracted_dir=extracted_dir,
            ocr_dir=settings.ocr_dir,
            scan_pages=settings.auto_link_scan_pages,
            threshold=settings.auto_link_confidence_threshold,
            min_lead_margin=settings.auto_link_min_lead_margin,
        )
        linked_at: DocumentCompanyLink | None = None
        if apply and accepted is not None and existing is None:
            linked_at = set_link(
                settings.document_companies_path,
                document_id=document_id,
                company_key=accepted.company.key,
                linked_by="auto-bulk",
            )
            linked += 1
        else:
            if reason == "ambiguous":
                ambiguous += 1
            elif reason == "below_threshold":
                below_threshold += 1
            elif reason == "no_match" or reason == "no_text":
                no_match += 1

        entries.append(
            BulkDetectEntry(
                document_id=document_id,
                reason=reason,
                accepted_key=accepted.company.key if accepted is not None else None,
                candidates=[_match_to_payload(match) for match in candidates],
                linked_at=linked_at,
                already_linked=existing,
            )
        )

    return BulkDetectResponse(
        threshold=settings.auto_link_confidence_threshold,
        scanned=len(entries),
        linked=linked,
        skipped_existing=skipped_existing,
        ambiguous=ambiguous,
        below_threshold=below_threshold,
        no_match=no_match,
        entries=entries,
    )


# -------------------------------------------- documents inventory drawer ---


class InventoryResponse(BaseModel):
    """What the Documents drawer renders."""

    model_config = ConfigDict(extra="forbid")

    entries: list[DocumentInventoryEntry] = Field(default_factory=list)
    total: int
    linked: int
    unlinked: int


@router.get("/document-inventory", response_model=InventoryResponse)
def document_inventory() -> InventoryResponse:
    """Return every known document plus its link, counts, and submission date."""

    settings = get_settings()
    registry = load_registry(settings.companies_registry_path)
    entries = build_inventory(
        findings_path=settings.findings_path,
        extracted_dir=settings.extracted_dir,
        reports_dir=settings.reports_dir,
        decisions_path=settings.reviewer_decisions_path,
        document_companies_path=settings.document_companies_path,
        companies=registry.companies,
    )
    linked = sum(1 for entry in entries if entry.link is not None)
    return InventoryResponse(
        entries=entries,
        total=len(entries),
        linked=linked,
        unlinked=len(entries) - linked,
    )
