"""Side-effectful shim around the pure ``brief`` module.

The pure functions in ``jsc_review.review.brief`` take data in and return
strings out. This module adds the one-shot "read findings and decisions,
splice the brief, write it back" workflow that the upload worker and the
reviewer decision endpoint call after they make a change.

The service is deliberately graceful: a missing brief or a missing findings
file logs a warning and returns ``None`` rather than raising. The caller is
always a background task or an endpoint that already committed its own work,
and a stale brief must never cause a user-facing failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from jsc_review.review.brief import (
    compute_snapshot,
    render_sections,
    update_brief,
)
from jsc_review.reviewer.findings import load_findings
from jsc_review.reviewer.store import load_decisions

_LOGGER = logging.getLogger(__name__)


def _load_gate_payload(path: Path | None) -> dict[str, object] | None:
    """Read the retrieval gate measurements JSON when present.

    Returns ``None`` when ``path`` is ``None`` or missing so the brief
    renders an "unmeasured" block rather than failing. A malformed file is
    logged and treated the same as missing.
    """

    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("brief: retrieval gate file unreadable at %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    return payload


@dataclass(frozen=True)
class RefreshSummary:
    """What ``refresh_brief_from_disk`` did on one call."""

    findings_count: int
    decisions_count: int
    replaced: tuple[str, ...]
    appended: tuple[str, ...]
    unchanged: tuple[str, ...]
    wrote: bool


def refresh_brief_from_disk(
    *,
    findings_path: Path,
    decisions_path: Path,
    brief_path: Path,
    gate_measurements_path: Path | None = None,
) -> RefreshSummary | None:
    """Recompute the brief from local findings and decisions, write it back.

    Returns ``None`` when either the findings file or the brief file is
    missing, so the caller can treat a first-time setup gracefully. A missing
    decisions file is treated as "no decisions yet" rather than an error,
    which matches how the reviewer store itself behaves. The retrieval gate
    measurements file is optional: when absent, the brief block says so and
    tells the operator which script to run.
    """

    if not findings_path.exists():
        _LOGGER.warning("brief refresh skipped: findings file missing at %s", findings_path)
        return None
    if not brief_path.exists():
        _LOGGER.warning("brief refresh skipped: brief file missing at %s", brief_path)
        return None

    findings = load_findings(findings_path)
    decisions = load_decisions(decisions_path) if decisions_path.exists() else []
    gate_payload = _load_gate_payload(gate_measurements_path)

    snapshot = compute_snapshot(findings, decisions, gate_measurements=gate_payload)
    sections = render_sections(snapshot)

    original = brief_path.read_text(encoding="utf-8")
    updated, result = update_brief(original, sections)

    wrote = updated != original
    if wrote:
        brief_path.write_text(updated, encoding="utf-8")

    return RefreshSummary(
        findings_count=snapshot.total_findings,
        decisions_count=snapshot.total_decisions,
        replaced=result.replaced,
        appended=result.appended,
        unchanged=result.unchanged,
        wrote=wrote,
    )


def refresh_brief_safely(
    *,
    findings_path: Path,
    decisions_path: Path,
    brief_path: Path,
    context: str,
    gate_measurements_path: Path | None = None,
) -> RefreshSummary | None:
    """Wrap ``refresh_brief_from_disk`` so a failure is logged, not raised.

    Both callers (the upload worker and the decision endpoint) commit their
    own primary effect before invoking the brief refresh. Letting an exception
    from the refresh bubble up would wrongly surface a stale-brief problem as
    a failed upload or a failed decision.
    """

    try:
        summary = refresh_brief_from_disk(
            findings_path=findings_path,
            decisions_path=decisions_path,
            brief_path=brief_path,
            gate_measurements_path=gate_measurements_path,
        )
    except Exception:  # noqa: BLE001  (intentional: a brief refresh must never fail the caller)
        _LOGGER.exception("brief refresh failed after %s", context)
        return None
    if summary is None:
        return None
    _LOGGER.info(
        "brief refreshed after %s: findings=%d decisions=%d wrote=%s",
        context,
        summary.findings_count,
        summary.decisions_count,
        summary.wrote,
    )
    return summary
