"""Show the raw OCR text for a sample of pages so you can judge quality."""

import json
from pathlib import Path

from jsc_review.companies.autolink import iter_pages_from_ocr

samples = [
    ("1_10000_599_29-03-2026_14-49-00", 88),
    ("217_10216_360_17-03-2026_12-46-33", 28),
    ("163_10162_161_07-04-2026_12-47-10", 101),
]

for doc, page in samples:
    path = Path(f"Data/reports/ocr/{doc}.json")
    pages = dict(iter_pages_from_ocr(json.loads(path.read_text(encoding="utf-8"))))
    text = pages.get(page, "")
    print("=" * 70)
    print(f"doc:  {doc}")
    print(f"page: {page}")
    print(f"chars: {len(text)}")
    print("-" * 70)
    print(text[:1200])
    print()
