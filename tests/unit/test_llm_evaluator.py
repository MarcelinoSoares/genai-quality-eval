"""Unit tests for LLMEvaluator — covers branches not hit by regression tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from genai_quality_eval.evaluators.llm_evaluator import EvaluationResult, LLMEvaluator


def test_average_score_empty_scores():
    result = EvaluationResult(question="q", answer="a")
    assert result.average_score == 0.0


def test_average_score_computed():
    result = EvaluationResult(
        question="q", answer="a", scores={"relevance": 0.8, "coherence": 0.6}
    )
    assert result.average_score == pytest.approx(0.7)


def test_evaluation_result_passed_flag():
    result = EvaluationResult(
        question="q", answer="a", scores={"relevance": 0.9}, threshold=0.75
    )
    assert result.passed is True


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


def test_call_llm_raises_when_no_client():
    evaluator = LLMEvaluator.__new__(LLMEvaluator)
    evaluator._client = None
    evaluator.model = "gpt-4o"
    with pytest.raises(RuntimeError, match="OpenAI client not available"):
        evaluator._call_llm("some prompt")


def test_call_llm_parses_float_from_response():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "  0.87  "
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    evaluator = LLMEvaluator.__new__(LLMEvaluator)
    evaluator._client = mock_client
    evaluator.model = "gpt-4o"

    assert evaluator._call_llm("test prompt") == pytest.approx(0.87)


def test_evaluate_without_context_skips_faithfulness(monkeypatch):
    monkeypatch.setattr(LLMEvaluator, "_build_client", lambda self: None)
    monkeypatch.setattr(LLMEvaluator, "_call_llm", lambda self, p: 0.80)

    evaluator = LLMEvaluator(model="gpt-4o")
    result = evaluator.evaluate(question="q?", answer="a")

    assert "faithfulness" not in result.scores
    assert "relevance" in result.scores
    assert "coherence" in result.scores
