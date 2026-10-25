"""Load selected checklist requirements from the local register CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from jsc_review.review.findings import Requirement


def load_selected_requirements(register_path: Path) -> list[Requirement]:
    """Return the requirements that are flagged for the proof of concept."""

    selected: list[Requirement] = []
    with register_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("poc_selected", "").strip() != "yes":
                continue
            selected.append(
                Requirement(
                    requirement_id=str(row["requirement_id"]),
                    section=str(row.get("section", "")),
                    text_ar=str(row.get("requirement_text_ar", "")),
                    text_en=str(row.get("requirement_text_en", "")),
                    evaluation_method=str(row.get("evaluation_method", "")),
                )
            )
    return selected
