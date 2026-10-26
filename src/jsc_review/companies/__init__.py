"""Company registry and document-to-company mapping.

The project runs against different companies and different purposes. Rather
than guessing a company from a filename, we keep a typed registry parsed
from the local Excel directory and a persistent map of
``document_id -> company``. The compliance assessor then compares the per
company submission timestamps against the configured disclosure periods.
"""
