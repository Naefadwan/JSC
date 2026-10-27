import csv

with open("Data/checklists/requirements_register.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row.get("poc_selected", "").strip() == "yes":
            rid = row["requirement_id"]
            method = row.get("evaluation_method", "")
            rtype = row.get("requirement_type", "")
            ar = (row.get("requirement_text_ar") or "")[:150]
            en = (row.get("requirement_text_en") or "")[:150]
            print(f"{rid}  method={method}  type={rtype}")
            print(f"  AR: {ar}")
            print(f"  EN: {en}")
            print()
