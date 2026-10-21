"""Deterministic threshold checks for ownership and customer disclosures.

A result is Met only when the disclosure wording and a qualifying percentage
are both in the embedded page text. OCR text stays ineligible for Met.
"""

import re
import unicodedata
from dataclasses import dataclass

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_SHARES = re.compile(r"(?<![0-9٠-٩])([0-9٠-٩]{1,3}(?:[.,][0-9٠-٩]{3})+)(?![0-9٠-٩])")
_PERCENT_MARK = re.compile(r"%|٪")
_DIGIT = set("0123456789٠١٢٣٤٥٦٧٨٩")
OWNER_CUES = (
    "كبار المساهم",
    "كبار مالكي",
    "يملكون ما نسبته",
    "نسبة مساهمتهم",
)
CUSTOMER_CUES = (
    "عملاء رئيس",
    "العملاء الرئيس",
    "موردين",
    "الموردين",
)


@dataclass(frozen=True)
class ThresholdMatch:
    """One threshold check before it becomes a review result."""

    status: str
    reason: str
    page: int | None
    excerpt: str
    verified: bool
    confidence: float
    from_ocr: bool


def match_owner_disclosure(
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    *,
    require_share_count: bool,
) -> ThresholdMatch:
    """Find a major-shareholder disclosure at or above 5%."""

    return _match(
        pages,
        ocr_pages,
        cues=OWNER_CUES,
        minimum=5.0,
        require_share_count=require_share_count,
        found_reason=(
            "The major-shareholder disclosure is in the embedded page, "
            "and at least one holding is at or above 5%."
        ),
        shares_reason=(
            "The major-shareholder disclosure is in the embedded page, "
            "with a holding at or above 5% and a share count."
        ),
        below_reason=(
            "The major-shareholder wording was found. No parsed holding "
            "is at or above 5%, so this stays Needs Review."
        ),
        missing_number_reason=(
            "The major-shareholder wording was found. The percentage cells "
            "were not fully read, so this stays Needs Review."
        ),
        missing_shares_reason=(
            "A holding at or above 5% was found. A share count for a major "
            "holder was not read from the embedded page, so this stays Needs Review."
        ),
        absent_reason=(
            "No recognized major-shareholder disclosure was found. "
            "Unfamiliar wording is Needs Review, not a confirmed miss."
        ),
        other_cues=CUSTOMER_CUES,
    )


def match_customer_disclosure(
    pages: dict[int, str],
    ocr_pages: dict[int, str],
) -> ThresholdMatch:
    """Find a supplier or customer disclosure at or above 10%."""

    return _match(
        pages,
        ocr_pages,
        cues=CUSTOMER_CUES,
        minimum=10.0,
        require_share_count=False,
        found_reason=(
            "The supplier or customer disclosure is in the embedded page, "
            "and at least one dependence figure is at or above 10%."
        ),
        shares_reason="",
        below_reason=(
            "The supplier or customer wording was found. No parsed dependence "
            "figure is at or above 10%, so this stays Needs Review."
        ),
        missing_number_reason=(
            "The supplier or customer wording was found. The percentage cells "
            "were not fully read, so this stays Needs Review."
        ),
        missing_shares_reason="",
        absent_reason=(
            "No recognized supplier or customer concentration disclosure was found. "
            "Unfamiliar wording is Needs Review, not a confirmed miss."
        ),
        other_cues=OWNER_CUES,
    )


def _match(
    pages: dict[int, str],
    ocr_pages: dict[int, str],
    *,
    cues: tuple[str, ...],
    minimum: float,
    require_share_count: bool,
    found_reason: str,
    shares_reason: str,
    below_reason: str,
    missing_number_reason: str,
    missing_shares_reason: str,
    absent_reason: str,
    other_cues: tuple[str, ...],
) -> ThresholdMatch:
    embedded = _best_page(pages, cues, other_cues, minimum, require_share_count)
    if embedded is not None:
        return _from_page(
            embedded,
            from_ocr=False,
            minimum=minimum,
            require_share_count=require_share_count,
            found_reason=shares_reason if require_share_count else found_reason,
            below_reason=below_reason,
            missing_number_reason=missing_number_reason,
            missing_shares_reason=missing_shares_reason,
        )
    ocr = _best_page(ocr_pages, cues, other_cues, minimum, require_share_count)
    if ocr is not None:
        found = _from_page(
            ocr,
            from_ocr=True,
            minimum=minimum,
            require_share_count=require_share_count,
            found_reason=found_reason,
            below_reason=below_reason,
            missing_number_reason=missing_number_reason,
            missing_shares_reason=missing_shares_reason,
        )
        return ThresholdMatch(
            status="Needs Review",
            reason=(
                "The disclosure appeared only in OCR text. OCR text cannot "
                "support Met until its error rate is within 5%. " + found.reason
            ),
            page=found.page,
            excerpt=found.excerpt,
            verified=False,
            confidence=0.4,
            from_ocr=True,
        )
    return ThresholdMatch(
        status="Needs Review",
        reason=absent_reason,
        page=None,
        excerpt="",
        verified=False,
        confidence=0.3,
        from_ocr=False,
    )


def _from_page(
    found: tuple[int, str, str, list[float], bool],
    *,
    from_ocr: bool,
    minimum: float,
    require_share_count: bool,
    found_reason: str,
    below_reason: str,
    missing_number_reason: str,
    missing_shares_reason: str,
) -> ThresholdMatch:
    page_number, text, cue, percents, has_shares = found
    qualifying = [value for value in percents if value >= minimum]
    excerpt = _window(text, cue)
    if not excerpt:
        return ThresholdMatch(
            status="Needs Review",
            reason=missing_number_reason,
            page=page_number,
            excerpt="",
            verified=False,
            confidence=0.4,
            from_ocr=from_ocr,
        )
    if qualifying and (has_shares or not require_share_count):
        shown = ", ".join(f"{value:g}%" for value in qualifying[:4])
        return ThresholdMatch(
            status="Met",
            reason=f"{found_reason} Parsed figures: {shown}.",
            page=page_number,
            excerpt=excerpt,
            verified=not from_ocr and excerpt in text,
            confidence=0.9,
            from_ocr=from_ocr,
        )
    if qualifying and require_share_count:
        reason = missing_shares_reason
        confidence = 0.55
    elif percents:
        reason = below_reason
        confidence = 0.5
    else:
        reason = missing_number_reason
        confidence = 0.45
    return ThresholdMatch(
        status="Needs Review",
        reason=reason,
        page=page_number,
        excerpt=excerpt,
        verified=not from_ocr and excerpt in text,
        confidence=confidence,
        from_ocr=from_ocr,
    )


def _best_page(
    pages: dict[int, str],
    cues: tuple[str, ...],
    other_cues: tuple[str, ...],
    minimum: float,
    require_share_count: bool,
) -> tuple[int, str, str, list[float], bool] | None:
    ranked: list[tuple[int, int, int, str, str, list[float], bool]] = []
    for page_number in sorted(pages):
        text = pages[page_number]
        cue = _cue(text, cues)
        if cue is None:
            continue
        percents = _percentages_for_cues(text, cues, other_cues)
        shares = _has_share_count(text)
        qualifying = sum(value >= minimum for value in percents)
        complete = int(qualifying > 0 and (shares or not require_share_count))
        ranked.append((complete, qualifying, -page_number, text, cue, percents, shares))
    if not ranked:
        return None
    _complete, _qualifying, page_key, text, cue, percents, shares = max(ranked)
    return -page_key, text, cue, percents, shares


def _cue(text: str, cues: tuple[str, ...]) -> str | None:
    plain = _strip_format_characters(text)
    for cue in cues:
        if cue in plain:
            return cue
    return None


def _percentages_for_cues(
    text: str,
    cues: tuple[str, ...],
    other_cues: tuple[str, ...],
) -> list[float]:
    plain = _strip_format_characters(text)
    values: list[float] = []
    for start, value in _figures(plain):
        own = _nearest(plain, start, cues)
        other = _nearest(plain, start, other_cues)
        if own is None or (other is not None and other <= own):
            continue
        values.append(value)
    return values


def _figures(text: str) -> list[tuple[int, float]]:
    figures: list[tuple[int, float]] = []
    for mark in _PERCENT_MARK.finditer(text):
        value = _number_after(text, mark.end())
        if value is None:
            value = _number_before(text, mark.start())
        if value is not None and 0 < value <= 100:
            figures.append((mark.start(), value))
    return figures


def _number_after(text: str, index: int) -> float | None:
    while index < len(text) and text[index] in " \t\u00a0":
        index += 1
    if index >= len(text) or text[index] not in _DIGIT:
        return None
    return _decimal_from(text, index)


def _number_before(text: str, index: int) -> float | None:
    cursor = index - 1
    newlines = 0
    while cursor >= 0 and text[cursor] in " \t\u00a0\n":
        if text[cursor] == "\n":
            newlines += 1
            if newlines > 1:
                return None
        cursor -= 1
    if cursor < 0 or text[cursor] not in _DIGIT:
        return None
    start = cursor
    while start >= 0 and (text[start] in _DIGIT or text[start] in ".,"):
        start -= 1
    return _percent_value(text[start + 1 : cursor + 1])


def _decimal_from(text: str, index: int) -> float | None:
    end = index
    while end < len(text) and text[end] in _DIGIT:
        end += 1
    if end < len(text) and text[end] in ".,":
        fraction = end + 1
        while fraction < len(text) and text[fraction] in _DIGIT:
            fraction += 1
        end = fraction
    if end == index:
        return None
    return _percent_value(text[index:end])


def _nearest(text: str, position: int, cues: tuple[str, ...]) -> int | None:
    best: int | None = None
    for cue in cues:
        start = 0
        while True:
            found = text.find(cue, start)
            if found < 0:
                break
            distance = abs(position - found)
            if best is None or distance < best:
                best = distance
            start = found + len(cue)
    return best


def _percent_value(token: str) -> float | None:
    normalized = token.translate(_ARABIC_DIGITS)
    if "," in normalized and "." in normalized:
        return None
    if "," in normalized:
        whole, fraction = normalized.split(",", 1)
        if not whole.isdigit() or not fraction.isdigit():
            return None
        if len(fraction) == 3:
            return None
        return float(f"{whole}.{fraction}")
    try:
        return float(normalized)
    except ValueError:
        return None


def _has_share_count(text: str) -> bool:
    plain = _strip_format_characters(text)
    matches = list(_SHARES.finditer(plain))
    for match in matches:
        nearby_before = plain[match.start() - 1] if match.start() else ""
        nearby_after = plain[match.end()] if match.end() < len(plain) else ""
        if nearby_before in {"%", "٪"} or nearby_after in {"%", "٪"}:
            continue
        digits = match.group(1).translate(_ARABIC_DIGITS).replace(",", "").replace(".", "")
        if digits.isdigit() and int(digits) >= 1000:
            return True
    return False


def _window(text: str, cue: str) -> str:
    index = text.find(cue)
    if index < 0:
        return ""
    start = max(0, index - 60)
    end = min(len(text), index + len(cue) + 180)
    return text[start:end].strip()


def _strip_format_characters(text: str) -> str:
    """Drop invisible format marks so a percentage glued to a mark still parses."""

    return "".join(char for char in text if unicodedata.category(char) != "Cf")
