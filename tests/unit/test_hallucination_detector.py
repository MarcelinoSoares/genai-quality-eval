"""Unit tests for HallucinationDetector — covers edge cases and all risk levels."""

from __future__ import annotations

import pytest

from genai_quality_eval.hallucination.risk_detector import (
    HallucinationDetector,
    RiskLevel,
)


@pytest.fixture
def detector():
    return HallucinationDetector()


def test_token_overlap_empty_answer(detector):
    # All tokens are stopwords → filtered to [] → returns 0.0
    score = detector._token_overlap_score("is the and a", "some real context here")
    assert score == 0.0


def test_ngram_overlap_empty_ngrams(detector):
    # Single content word → can't form any 3-gram → returns 0.0
    score = detector._ngram_overlap_score("hello", "hello world foo bar")
    assert score == 0.0


def test_extract_suspicious_sentences_skips_stopword_only_sentence(detector):
    # "Is the." → after stopword filter tokens = {} → skipped, not added
    answer = "Is the. Quantum entanglement drives nuclear fusion reactions."
    context = "The sky is blue."
    suspicious = detector._extract_suspicious_sentences(answer, context)
    assert any("Quantum" in s for s in suspicious)
    # The stopword-only sentence must not appear
    assert not any(s.strip() in ("Is the", "Is the.") for s in suspicious)


def test_classify_risk_low(detector):
    assert detector._classify_risk(0.10) == RiskLevel.LOW
    assert detector._classify_risk(0.29) == RiskLevel.LOW


def test_classify_risk_medium(detector):
    assert detector._classify_risk(0.30) == RiskLevel.MEDIUM
    assert detector._classify_risk(0.59) == RiskLevel.MEDIUM


def test_classify_risk_high(detector):
    assert detector._classify_risk(0.60) == RiskLevel.HIGH
    assert detector._classify_risk(1.00) == RiskLevel.HIGH


def test_score_low_risk_verbatim(detector):
    context = "Retrieval augmented generation retrieves relevant documents for the model."
    answer = context  # verbatim copy → near-zero ungroundedness
    result = detector.score(answer=answer, context=context)
    assert result.risk_level == RiskLevel.LOW
    assert result.passed is True


def test_score_high_risk_off_topic(detector):
    context = "The sky is blue on a clear day."
    answer = "Quantum entanglement drives photosynthesis reactions inside living cells."
    result = detector.score(answer=answer, context=context)
    assert result.risk_level == RiskLevel.HIGH
    assert result.passed is False
