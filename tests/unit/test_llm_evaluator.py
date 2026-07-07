"""Unit tests for LLMEvaluator — covers branches not hit by regression tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call

import pytest

from genai_quality_eval.evaluators.llm_evaluator import (
    EvaluationResult,
    LLMEvaluator,
    LLMParseError,
)


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


def test_average_score_empty_scores():
    result = EvaluationResult(question="q", answer="a")
    assert result.average_score == 0.0


def test_average_score_computed():
    result = EvaluationResult(question="q", answer="a", scores={"relevance": 0.8, "coherence": 0.6})
    assert result.average_score == pytest.approx(0.7)


def test_evaluation_result_passed_flag():
    result = EvaluationResult(question="q", answer="a", scores={"relevance": 0.9}, threshold=0.75)
    assert result.passed is True


def test_evaluation_result_failed_reason_forces_not_passed():
    result = EvaluationResult(
        question="q",
        answer="a",
        scores={"relevance": 0.9},
        threshold=0.75,
        failed_reason="Parse error: empty response",
    )
    assert result.passed is False


# ---------------------------------------------------------------------------
# _build_client
# ---------------------------------------------------------------------------


def test_build_client_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    evaluator = LLMEvaluator.__new__(LLMEvaluator)
    evaluator.model = "gpt-4o"
    evaluator.threshold = 0.75
    assert evaluator._build_client() is None


def test_build_client_success(monkeypatch):
    mock_client = MagicMock()
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = mock_client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    evaluator = LLMEvaluator.__new__(LLMEvaluator)
    evaluator.model = "gpt-4o"
    assert evaluator._build_client() is mock_client


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------


def _make_evaluator() -> LLMEvaluator:
    ev = LLMEvaluator.__new__(LLMEvaluator)
    ev.model = "gpt-4o"
    ev.threshold = 0.75
    ev.max_retries = 3
    ev.timeout = 30.0
    ev._client = None
    return ev


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.87", 0.87),
        ("  0.87  ", 0.87),
        ("0.82 because the answer is relevant", 0.82),
        ("Score: 0.75", 0.75),
        ("The answer scores 1.0 out of 1.0", 1.0),
        ("0", 0.0),
        ("1", 1.0),
    ],
)
def test_parse_score_valid(raw, expected):
    ev = _make_evaluator()
    assert ev._parse_score(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "excellent!",
        "N/A",
        "the model cannot score this",
    ],
)
def test_parse_score_no_number_raises(raw):
    ev = _make_evaluator()
    with pytest.raises(LLMParseError, match="Cannot extract score|Empty response"):
        ev._parse_score(raw)


def test_parse_score_out_of_range_raises():
    ev = _make_evaluator()
    with pytest.raises(LLMParseError, match="out of range"):
        ev._parse_score("1.5")


# ---------------------------------------------------------------------------
# _call_llm
# ---------------------------------------------------------------------------


def test_call_llm_raises_when_no_client():
    evaluator = _make_evaluator()
    with pytest.raises(RuntimeError, match="OpenAI client not available"):
        evaluator._call_llm("some prompt")


def test_call_llm_parses_float_from_response():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "  0.87  "
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    evaluator = _make_evaluator()
    evaluator._client = mock_client

    assert evaluator._call_llm("test prompt") == pytest.approx(0.87)


def test_call_llm_parses_score_from_verbose_response():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "0.82 because the answer is relevant"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    evaluator = _make_evaluator()
    evaluator._client = mock_client

    assert evaluator._call_llm("test prompt") == pytest.approx(0.82)


def test_call_llm_does_not_retry_parse_error():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "excellent!"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    evaluator = _make_evaluator()
    evaluator._client = mock_client

    with pytest.raises(LLMParseError):
        evaluator._call_llm("test prompt")

    # parse errors are deterministic — should not retry
    assert mock_client.chat.completions.create.call_count == 1


def test_call_llm_retries_on_transient_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    fail_response = MagicMock()
    fail_response.choices[0].message.content = "0.80"
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError("rate limit"),
        RuntimeError("timeout"),
        fail_response,
    ]

    evaluator = _make_evaluator()
    evaluator._client = mock_client
    evaluator.max_retries = 3

    assert evaluator._call_llm("test prompt") == pytest.approx(0.80)
    assert mock_client.chat.completions.create.call_count == 3


def test_call_llm_raises_after_max_retries_exhausted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("service unavailable")

    evaluator = _make_evaluator()
    evaluator._client = mock_client
    evaluator.max_retries = 3

    with pytest.raises(RuntimeError, match="service unavailable"):
        evaluator._call_llm("test prompt")

    assert mock_client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_without_context_skips_faithfulness(monkeypatch):
    monkeypatch.setattr(LLMEvaluator, "_build_client", lambda self: None)
    monkeypatch.setattr(LLMEvaluator, "_call_llm", lambda self, p: 0.80)

    evaluator = LLMEvaluator(model="gpt-4o")
    result = evaluator.evaluate(question="q?", answer="a")

    assert "faithfulness" not in result.scores
    assert "relevance" in result.scores
    assert "coherence" in result.scores


def test_evaluate_returns_failed_result_on_parse_error(monkeypatch):
    monkeypatch.setattr(LLMEvaluator, "_build_client", lambda self: None)
    monkeypatch.setattr(
        LLMEvaluator,
        "_call_llm",
        lambda self, p: (_ for _ in ()).throw(LLMParseError("Cannot extract score from: 'excellent!'")),
    )

    evaluator = LLMEvaluator(model="gpt-4o")
    result = evaluator.evaluate(question="q?", answer="a")

    assert result.passed is False
    assert result.failed_reason is not None
    assert "Parse error" in result.failed_reason


def test_evaluate_returns_failed_result_on_llm_error(monkeypatch):
    monkeypatch.setattr(LLMEvaluator, "_build_client", lambda self: None)
    monkeypatch.setattr(
        LLMEvaluator,
        "_call_llm",
        lambda self, p: (_ for _ in ()).throw(RuntimeError("service down")),
    )

    evaluator = LLMEvaluator(model="gpt-4o")
    result = evaluator.evaluate(question="q?", answer="a")

    assert result.passed is False
    assert result.failed_reason is not None
    assert "LLM call failed" in result.failed_reason
