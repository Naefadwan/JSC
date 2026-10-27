"""Compare the register's selected rows with findings.json by requirement."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

reg = list(csv.DictReader(open("Data/checklists/requirements_register.csv", encoding="utf-8-sig")))
selected = [r for r in reg if r.get("selected", "").strip().lower() in ("true", "1", "yes", "y")]
print("selected rows:", len(selected))
print("by evaluation_method:", dict(Counter(r["evaluation_method"] for r in selected)))

rows = json.loads(Path("Data/reports/reviews/findings.json").read_text(encoding="utf-8"))
req_counts = Counter(r["result"]["requirement_id"] for r in rows)
print("unique req ids in findings:", len(req_counts))

by_req: dict[str, Counter[str]] = {}
for row in rows:
    rid = row["result"]["requirement_id"]
    hr = row["result"]["human_review"]
    by_req.setdefault(rid, Counter())[hr] += 1

method_of = {r["requirement_id"]: r["evaluation_method"] for r in reg}

print()
print("per-requirement referral:")
for rid in sorted(by_req):
    c = by_req[rid]
    required = c.get("Required", 0)
    print(f"  {rid:>8}  Required={required}/5  method={method_of.get(rid, '?')}")

print()
print("register selected but missing from findings:")
sel_ids = {r["requirement_id"] for r in selected}
found_ids = set(by_req)
print(sorted(sel_ids - found_ids))
print()
print("found but not selected (should be empty):")
print(sorted(found_ids - sel_ids))
