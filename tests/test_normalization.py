"""Tests for Arabic-aware text normalization."""

from __future__ import annotations

from jsc_review.documents.normalization import normalize_arabic, quote_matches_page


def test_alef_variants_fold_to_alef() -> None:
    assert normalize_arabic("أحمد") == normalize_arabic("احمد")
    assert normalize_arabic("إدارة") == normalize_arabic("ادارة")
    assert normalize_arabic("آمل") == normalize_arabic("امل")


def test_tashkeel_is_stripped() -> None:
    assert normalize_arabic("مُحَمَّد") == "محمد"


def test_tatweel_is_stripped() -> None:
    assert normalize_arabic("يــســتــعــمــل") == "يستعمل"


def test_invisibles_are_removed() -> None:
    assert normalize_arabic("شركة\u200cالمنتجات") == "شركةالمنتجات"
    assert normalize_arabic("word\ufeffmore") == "wordmore"


def test_whitespace_is_collapsed() -> None:
    assert normalize_arabic("أ    ب\n\tج") == "ا ب ج"


def test_alef_maksura_folds_to_ya() -> None:
    assert normalize_arabic("على") == normalize_arabic("علي")


def test_arabic_digits_fold_to_ascii() -> None:
    assert normalize_arabic("نسبة ٢٨٪") == "نسبة 28٪"
    assert normalize_arabic("نسبة ۲۸٪") == "نسبة 28٪"


def test_normalization_is_idempotent() -> None:
    text = "إدارة المُحَاسَبَة ٥٪  ثم على"
    once = normalize_arabic(text)
    twice = normalize_arabic(once)
    assert once == twice


def test_empty_input_returns_empty() -> None:
    assert normalize_arabic("") == ""
    assert normalize_arabic("   ") == ""


def test_quote_matches_page_handles_raw_substring() -> None:
    page = "This is a paragraph about financial risk policy."
    assert quote_matches_page("financial risk policy", page) is True


def test_quote_matches_page_handles_alef_variant() -> None:
    page = "إدارة المخاطر"
    quote = "ادارة المخاطر"
    assert quote_matches_page(quote, page) is True


def test_quote_matches_page_handles_mismatched_whitespace() -> None:
    page = "كبار\nالمساهمين\nنسبة 28٪"
    quote = "كبار المساهمين نسبة 28٪"
    assert quote_matches_page(quote, page) is True


def test_quote_matches_page_handles_tashkeel_in_page() -> None:
    page = "مُحَمَّد عبد الله"
    quote = "محمد عبد الله"
    assert quote_matches_page(quote, page) is True


def test_quote_matches_page_rejects_absent_text() -> None:
    page = "لا توجد سياسة"
    quote = "توجد سياسة واضحة"
    assert quote_matches_page(quote, page) is False


def test_quote_matches_page_handles_empty_input() -> None:
    assert quote_matches_page("", "page") is False
    assert quote_matches_page("quote", "") is False
