"""Register load/save round-trip and selection update helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from jsc_review.review.register_edit import (
    REGISTER_COLUMNS,
    RegisterRow,
    apply_selection_updates,
    load_register,
    save_register,
    update_register_selections,
)


def _write_register(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REGISTER_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REGISTER_COLUMNS})


def _sample_rows() -> list[dict[str, str]]:
    return [
        {
            "requirement_id": "GOV-001",
            "checklist": "governance",
            "section": "التزامات المجلس",
            "requirement_text_ar": "اعتماد سياسة إدارة المخاطر.",
            "requirement_text_en": "Adopt a risk management policy.",
            "requirement_type": "presence",
            "evaluation_method": "llm",
            "poc_selected": "yes",
            "owner": "",
            "notes": "proof of concept",
        },
        {
            "requirement_id": "GOV-002",
            "checklist": "governance",
            "section": "التزامات المجلس",
            "requirement_text_ar": "اعتماد مدونة سلوك مهني.",
            "requirement_text_en": "Adopt a code of conduct.",
            "requirement_type": "completeness",
            "evaluation_method": "llm",
            "poc_selected": "",
            "owner": "",
            "notes": "",
        },
        {
            "requirement_id": "DIS-001",
            "checklist": "disclosure",
            "section": "القوائم المالية",
            "requirement_text_ar": "قائمة الدخل.",
            "requirement_text_en": "Income statement.",
            "requirement_type": "presence",
            "evaluation_method": "deterministic",
            "poc_selected": "yes",
            "owner": "",
            "notes": "",
        },
    ]


# --------------------------------------------------------- load_register ----


def test_load_register_parses_yes_as_true(tmp_path: Path) -> None:
    path = tmp_path / "register.csv"
    _write_register(path, _sample_rows())

    rows = load_register(path)

    assert [r.requirement_id for r in rows] == ["GOV-001", "GOV-002", "DIS-001"]
    assert rows[0].poc_selected is True
    assert rows[1].poc_selected is False
    assert rows[2].poc_selected is True
    assert rows[0].requirement_text_en == "Adopt a risk management policy."


def test_load_register_accepts_various_truthy_markers(tmp_path: Path) -> None:
    path = tmp_path / "register.csv"
    rows = _sample_rows()
    rows[0]["poc_selected"] = "YES"
    rows[1]["poc_selected"] = "true"
    rows[2]["poc_selected"] = "1"
    _write_register(path, rows)

    parsed = load_register(path)

    assert all(row.poc_selected for row in parsed)


def test_load_register_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_register(tmp_path / "missing.csv") == []


def test_load_register_skips_rows_with_no_requirement_id(tmp_path: Path) -> None:
    path = tmp_path / "register.csv"
    rows = _sample_rows()
    rows.append({"requirement_id": "   ", "poc_selected": "yes"})
    _write_register(path, rows)

    parsed = load_register(path)

    assert len(parsed) == 3


# --------------------------------------------------------- save_register ----


def test_save_register_is_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "register.csv"
    _write_register(path, _sample_rows())
    rows = load_register(path)

    save_register(path, rows)
    reparsed = load_register(path)

    assert reparsed == rows
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == REGISTER_COLUMNS


def test_save_register_writes_yes_for_true_selection(tmp_path: Path) -> None:
    path = tmp_path / "register.csv"
    save_register(
        path,
        [
            RegisterRow(requirement_id="A", poc_selected=True),
            RegisterRow(requirement_id="B", poc_selected=False),
        ],
    )
    content = path.read_text(encoding="utf-8")
    assert "A,,,,,,,yes," in content.replace(" ", "")
    assert "B,,,,,,,," in content.replace(" ", "")


def test_save_register_is_atomic_via_tmp_file(tmp_path: Path) -> None:
    """A completed save leaves the final file in place and no .tmp stragglers."""

    path = tmp_path / "register.csv"
    save_register(path, [RegisterRow(requirement_id="A")])
    assert path.exists()
    assert not (tmp_path / "register.csv.tmp").exists()


# ------------------------------------------------- apply_selection_updates ---


def test_apply_selection_updates_flips_only_requested_rows() -> None:
    rows = [
        RegisterRow(requirement_id="A", poc_selected=True),
        RegisterRow(requirement_id="B", poc_selected=False),
        RegisterRow(requirement_id="C", poc_selected=True),
    ]
    updated, missing = apply_selection_updates(rows, {"B": True, "C": False})

    by_id = {row.requirement_id: row for row in updated}
    assert by_id["A"].poc_selected is True  # untouched
    assert by_id["B"].poc_selected is True  # flipped on
    assert by_id["C"].poc_selected is False  # flipped off
    assert missing == []


def test_apply_selection_updates_surfaces_unknown_ids() -> None:
    rows = [RegisterRow(requirement_id="A", poc_selected=True)]
    _, missing = apply_selection_updates(rows, {"A": False, "X": True, "Y": False})
    assert set(missing) == {"X", "Y"}


def test_apply_selection_updates_preserves_row_order() -> None:
    rows = [
        RegisterRow(requirement_id="A"),
        RegisterRow(requirement_id="B"),
        RegisterRow(requirement_id="C"),
    ]
    updated, _ = apply_selection_updates(rows, {"C": True})
    assert [r.requirement_id for r in updated] == ["A", "B", "C"]


def test_apply_selection_updates_is_pure() -> None:
    """The function returns new rows and does not mutate the input."""

    row_a = RegisterRow(requirement_id="A", poc_selected=False)
    row_b = RegisterRow(requirement_id="B", poc_selected=True)

    updated, _ = apply_selection_updates([row_a, row_b], {"A": True, "B": False})

    assert row_a.poc_selected is False
    assert row_b.poc_selected is True
    assert updated[0].poc_selected is True
    assert updated[1].poc_selected is False


# --------------------------------------------- update_register_selections ---


def test_update_register_selections_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "register.csv"
    _write_register(path, _sample_rows())

    rows, missing = update_register_selections(path, {"GOV-002": True, "DIS-001": False})

    assert missing == []
    reread = load_register(path)
    by_id = {row.requirement_id: row for row in reread}
    assert by_id["GOV-001"].poc_selected is True
    assert by_id["GOV-002"].poc_selected is True
    assert by_id["DIS-001"].poc_selected is False
    # Returned view matches disk
    assert {r.requirement_id: r.poc_selected for r in rows} == {
        "GOV-001": True,
        "GOV-002": True,
        "DIS-001": False,
    }


def test_update_register_selections_returns_unknown_ids(tmp_path: Path) -> None:
    path = tmp_path / "register.csv"
    _write_register(path, _sample_rows())

    _, missing = update_register_selections(path, {"GOV-001": False, "NEW-999": True})

    assert missing == ["NEW-999"]
    # Known id was still applied
    reread = {row.requirement_id: row for row in load_register(path)}
    assert reread["GOV-001"].poc_selected is False


def test_update_register_selections_with_empty_map_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "register.csv"
    _write_register(path, _sample_rows())
    before = load_register(path)

    rows, missing = update_register_selections(path, {})

    assert missing == []
    assert rows == before


# ------------------------------- interop with load_selected_requirements ---


def test_selected_load_sees_the_update(tmp_path: Path) -> None:
    """``load_selected_requirements`` reflects the fresh selection after a save."""

    from jsc_review.review.registry import load_selected_requirements

    path = tmp_path / "register.csv"
    _write_register(path, _sample_rows())

    before = {r.requirement_id for r in load_selected_requirements(path)}
    assert before == {"GOV-001", "DIS-001"}

    update_register_selections(path, {"GOV-002": True, "DIS-001": False})

    after = {r.requirement_id for r in load_selected_requirements(path)}
    assert after == {"GOV-001", "GOV-002"}
