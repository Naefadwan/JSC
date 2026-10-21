"""Character and word error rates for local OCR."""

import json
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

_WHITESPACE = re.compile(r"\s+")
_Item = TypeVar("_Item")


class OcrMetricError(ValueError):
    """Raised when the versioned OCR policy cannot be read."""


@lru_cache
def ocr_max_character_error_rate() -> float:
    """Return the confirmed maximum character error rate for verified evidence."""

    policy_path = Path(__file__).resolve().parents[1] / "review" / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    threshold = policy["ocr_max_character_error_rate"]
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise OcrMetricError("OCR character-error threshold must be a number")
    return float(threshold)


def normalize_ocr_text(text: str) -> str:
    """Collapse whitespace so line breaks are not counted as recognition errors."""

    return _WHITESPACE.sub(" ", text).strip()


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """Return edit distance divided by the reference length, or None if unscoresable."""

    expected = normalize_ocr_text(reference)
    predicted = normalize_ocr_text(hypothesis)
    if not expected:
        return None
    return levenshtein(expected, predicted) / len(expected)


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Return word-level edit distance divided by the reference word count."""

    expected = normalize_ocr_text(reference).split(" ")
    predicted = normalize_ocr_text(hypothesis).split(" ")
    expected = [word for word in expected if word]
    predicted = [word for word in predicted if word]
    if not expected:
        return None
    return levenshtein(expected, predicted) / len(expected)


def supports_verified_evidence(character_error_rate_value: float | None) -> bool:
    """Allow OCR text to support Met only at or below the confirmed error rate."""

    if character_error_rate_value is None:
        return False
    return character_error_rate_value <= ocr_max_character_error_rate()


def levenshtein(reference: Sequence[_Item], hypothesis: Sequence[_Item]) -> int:
    """Return the edit distance between two strings or token lists."""

    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for index, reference_item in enumerate(reference, start=1):
        current = [index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, start=1):
            insertion = current[hypothesis_index - 1] + 1
            deletion = previous[hypothesis_index] + 1
            substitution = previous[hypothesis_index - 1] + (reference_item != hypothesis_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]
