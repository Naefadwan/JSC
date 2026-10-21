"""Tests for OCR error rates and the verified-evidence limit."""

from jsc_review.documents.ocr_metrics import (
    character_error_rate,
    ocr_max_character_error_rate,
    supports_verified_evidence,
    word_error_rate,
)


def test_identical_text_has_zero_error() -> None:
    assert character_error_rate("مجلس الإدارة", "مجلس الإدارة") == 0.0
    assert word_error_rate("board members", "board members") == 0.0


def test_character_error_counts_one_substitution() -> None:
    assert character_error_rate("abcd", "abxd") == 0.25


def test_word_error_counts_one_replaced_word() -> None:
    assert word_error_rate("the board adopted a policy", "the board accepted a policy") == 0.2


def test_whitespace_differences_are_not_errors() -> None:
    assert character_error_rate("the board\nadopted", "the  board adopted") == 0.0


def test_empty_reference_is_not_scored() -> None:
    assert character_error_rate("   ", "text") is None
    assert word_error_rate("", "text") is None


def test_unmeasured_ocr_cannot_support_verified_evidence() -> None:
    assert supports_verified_evidence(None) is False


def test_error_rate_above_the_confirmed_limit_cannot_support_met() -> None:
    limit = ocr_max_character_error_rate()

    assert limit == 0.05
    assert supports_verified_evidence(limit) is True
    assert supports_verified_evidence(limit + 0.01) is False
