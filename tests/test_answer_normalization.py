"""Tests for benchmark answer normalization."""

from data.benchmarks.answer_normalization import normalize_answer


def test_normalize_answer_lowercases_punctuation_and_whitespace() -> None:
    assert normalize_answer("  Chicago! ") == "chicago"
    assert normalize_answer("New   York, U.S.A.") == "new york u s a"


def test_normalize_answer_keeps_none_stable() -> None:
    assert normalize_answer(None) == ""
