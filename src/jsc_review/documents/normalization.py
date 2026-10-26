"""Arabic-aware text normalization.

Report pages come out of PyMuPDF with hard line breaks, mixed Arabic and
English, decorative characters, and encoding variants of the same Arabic
letter. This normalizer gives a stable surface form that is used:

- When the model reads a passage, so layout artifacts do not confuse it.
- When a quote is verified against the page, so a minor spacing or alef
  variant does not block Met.

The transformations are limited to shape-equivalence changes that experienced
Arabic reviewers accept. Normalization is a view, not a replacement: the raw
text stays on disk and is still what the schema stores as evidence.
"""

from __future__ import annotations

import re
import unicodedata

# U+0610-U+061A, U+064B-U+065F, U+0670, U+06D6-U+06ED: harakat, hamza above,
# diacritical marks, Quranic annotation signs. Removing them does not change
# meaning for a disclosure review.
_TASHKEEL = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")

# Tatweel is a stretching character used for justification. It carries no
# linguistic value.
_TATWEEL = "\u0640"

# Soft hyphen and zero-width characters leak in from PDF layout and copy-paste.
_INVISIBLES = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

# Collapse any run of whitespace (including Arabic newlines) to a single space.
_WHITESPACE = re.compile(r"\s+")

# Arabic alef variants that print the same.
_ALEF_VARIANTS = {
    "\u0622": "\u0627",  # alef with madda above
    "\u0623": "\u0627",  # alef with hamza above
    "\u0625": "\u0627",  # alef with hamza below
    "\u0671": "\u0627",  # alef wasla
}

# Alef maksura is often written ي in body text.
_YA_VARIANTS = {"\u0649": "\u064a"}

# Arabic-Indic and Eastern Arabic-Indic digits to ASCII, so a parsed number
# like ٥ reads the same as 5 for retrieval and quote comparison.
_DIGIT_VARIANTS = {chr(0x0660 + i): str(i) for i in range(10)}
_DIGIT_VARIANTS.update({chr(0x06F0 + i): str(i) for i in range(10)})


def normalize_arabic(text: str) -> str:
    """Return a comparison-stable form of a passage.

    The function is idempotent: ``normalize_arabic(normalize_arabic(x)) == normalize_arabic(x)``.
    """

    if not text:
        return ""
    # NFKC first: compose characters, normalize presentation forms (Arabic
    # ligatures FEFB etc. decompose back to canonical letters).
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _INVISIBLES.sub("", normalized)
    normalized = _TASHKEEL.sub("", normalized)
    normalized = normalized.replace(_TATWEEL, "")
    normalized = _apply_mapping(normalized, _ALEF_VARIANTS)
    normalized = _apply_mapping(normalized, _YA_VARIANTS)
    normalized = _apply_mapping(normalized, _DIGIT_VARIANTS)
    normalized = _WHITESPACE.sub(" ", normalized)
    return normalized.strip()


def quote_matches_page(quote: str, page_text: str) -> bool:
    """Check whether a quote appears in a page under the normalized view.

    Match conditions, in order:
      1. Exact substring match in the raw text (always safe).
      2. Substring match after Arabic normalization is applied to both sides.

    Returning ``True`` under rule 2 means the quote appears in the page up to
    whitespace, alef variants, tashkeel, tatweel, invisibles, and digit shape.
    The reviewer still sees the raw page in the UI.
    """

    if not quote or not page_text:
        return False
    if quote in page_text:
        return True
    normalized_quote = normalize_arabic(quote)
    if not normalized_quote:
        return False
    return normalized_quote in normalize_arabic(page_text)


def _apply_mapping(text: str, mapping: dict[str, str]) -> str:
    for source, target in mapping.items():
        if source in text:
            text = text.replace(source, target)
    return text
