"""Requirement-driven query augmentation for retrieval.

Each selected requirement carries universal cue vocabulary: heading variants
for statement-heading rules, cue phrases for ownership and customer
thresholds. These cues are keyed by the requirement, not by the company, so
augmentation remains reproducible and fair across every new upload.

Nothing in this module reads the document being searched, so the behavior
is identical regardless of which company's report is loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jsc_review.review.thresholds import CUSTOMER_CUES, OWNER_CUES

if TYPE_CHECKING:
    from jsc_review.review.findings import Requirement

_OWNER_RULES = frozenset({"DIS-010", "DIS-011"})
_CUSTOMER_RULES = frozenset({"DIS-017"})


def cue_terms_for(requirement: Requirement) -> tuple[str, ...]:
    """Return the universal cue vocabulary associated with a requirement."""

    # Local import defers the dependency on ``findings`` so the retrieval
    # layer stays free of circular imports: findings imports search, search
    # imports this module, and this module needs HEADING_VARIANTS from findings.
    from jsc_review.review.findings import HEADING_VARIANTS

    requirement_id = requirement.requirement_id
    if requirement_id in HEADING_VARIANTS:
        return tuple(HEADING_VARIANTS[requirement_id])
    if requirement_id in _OWNER_RULES:
        return tuple(OWNER_CUES)
    if requirement_id in _CUSTOMER_RULES:
        return tuple(CUSTOMER_CUES)
    return ()


def _unique_preserving_order(terms: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    kept: list[str] = []
    for term in terms:
        folded = term.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        kept.append(term)
    return tuple(kept)


def build_query(requirement: Requirement) -> str:
    """Return a query that pairs the requirement text with its universal cues.

    The base query is ``text_ar text_en`` so retrieval sees both languages.
    Cue phrases are appended once so near-duplicate wording cannot inflate
    one page's BM25 score with repeated synonyms.
    """

    parts: list[str] = []
    if requirement.text_ar:
        parts.append(requirement.text_ar)
    if requirement.text_en:
        parts.append(requirement.text_en)
    cues = _unique_preserving_order(cue_terms_for(requirement))
    parts.extend(cues)
    if not parts:
        return requirement.requirement_id
    return " ".join(parts)
