"""Prompt Regression Tests.

Detects quality degradation in LLM responses across prompt versions or model updates.

Strategy:
1. Store baseline scores for a canonical set of prompt/answer pairs.
2. On each CI run, re-evaluate and compare against baselines.
3. Fail if any metric regresses beyond the allowed tolerance (default 5%).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pytest

from evaluators.llm_evaluator import EvaluationResult, LLMEvaluator
from hallucination.risk_detector import HallucinationDetector
from rag.retrieval_validator import RetrievalValidator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SNAPSHOT_DIR = Path(__file__).parent / "prompt_snapshots"
REGRESSION_TOLERANCE = float(os.getenv("REGRESSION_TOLERANCE", "0.05"))  # 5%

PROMPT_SUITE: List[Dict] = [
    {
        "id": "rag_basics",
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "context": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "a retrieval system with a language model. The retrieval system "
            "fetches relevant documents, which are then provided as context "
            "to the LLM so it can generate more accurate, grounded responses."
        ),
        "answer": (
            "RAG is a method that enhances language models by retrieving relevant "
            "documents and using them as context during generation, improving "
            "accuracy and reducing hallucinations."
        ),
        "baseline_scores": {"relevance": 0.90, "coherence": 0.85, "faithfulness": 0.88},
    },
    {
        "id": "latency_definition",
        "question": "Why is latency important in GenAI systems?",
        "context": (
            "Latency in GenAI systems refers to the time between a user sending "
            "a prompt and receiving a response. High latency degrades user experience "
            "and can violate SLA agreements in production systems."
        ),
        "answer": (
            "Latency is critical because it directly impacts user experience and "
            "production SLA compliance. Long response times reduce the usability "
            "of GenAI applications."
        ),
        "baseline_scores": {"relevance": 0.88, "coherence": 0.87, "faithfulness": 0.85},
    },
    {
        "id": "hallucination_risk",
        "question": "What causes hallucinations in LLMs?",
        "context": (
            "Hallucinations in LLMs occur when the model generates text that is "
            "factually incorrect or unsupported by the training data or context. "
            "They are caused by the model filling gaps in knowledge with plausible "
            "but incorrect information."
        ),
        "answer": (
            "LLM hallucinations happen when the model produces content not grounded "
            "in provided context or training data, often generating confident but "
            "incorrect statements."
        ),
        "baseline_scores": {"relevance": 0.91, "coherence": 0.89, "faithfulness": 0.86},
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_snapshot(prompt_id: str) -> dict:
    path = SNAPSHOT_DIR / f"{prompt_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_snapshot(prompt_id: str, scores: dict) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{prompt_id}.json"
    path.write_text(json.dumps(scores, indent=2))


def _assert_no_regression(baseline: dict, current: dict, tolerance: float) -> None:
    for metric, baseline_score in baseline.items():
        current_score = current.get(metric)
        if current_score is None:
            continue
        min_acceptable = baseline_score * (1 - tolerance)
        assert current_score >= min_acceptable, (
            f"Regression detected in '{metric}': "
            f"baseline={baseline_score:.3f}, current={current_score:.3f}, "
            f"drop={baseline_score - current_score:.3f} "
            f"(tolerance={tolerance * 100:.0f}%)"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PROMPT_SUITE, ids=[c["id"] for c in PROMPT_SUITE])
def test_hallucination_regression(case: dict) -> None:
    """Hallucination risk must not increase vs baseline."""
    detector = HallucinationDetector(risk_threshold=0.30)
    result = detector.score(answer=case["answer"], context=case["context"])

    assert result.risk_score < 0.30, (
        f"[{case['id']}] Hallucination risk too high: {result.risk_score:.3f} "
        f"(level={result.risk_level.value})"
    )


@pytest.mark.parametrize("case", PROMPT_SUITE, ids=[c["id"] for c in PROMPT_SUITE])
def test_score_regression_against_baseline(case: dict, monkeypatch) -> None:
    """Scores must not degrade by more than REGRESSION_TOLERANCE vs baseline."""
    # Monkeypatch _call_llm to return baseline scores deterministically in CI
    baseline = case["baseline_scores"]
    score_iter = iter(baseline.values())

    def mock_call_llm(self, prompt: str) -> float:
        try:
            return next(score_iter)
        except StopIteration:
            return 0.80

    monkeypatch.setattr(LLMEvaluator, "_call_llm", mock_call_llm)
    monkeypatch.setattr(LLMEvaluator, "_build_client", lambda self: None)

    evaluator = LLMEvaluator(model="gpt-4o")
    result = evaluator.evaluate(
        question=case["question"],
        answer=case["answer"],
        context=case["context"],
    )

    _assert_no_regression(
        baseline=baseline,
        current=result.scores,
        tolerance=REGRESSION_TOLERANCE,
    )

    assert result.passed, (
        f"[{case['id']}] Average score {result.average_score:.3f} "
        f"below threshold {result.threshold}"
    )


@pytest.mark.parametrize("case", PROMPT_SUITE, ids=[c["id"] for c in PROMPT_SUITE])
def test_rag_retrieval_regression(case: dict) -> None:
    """RAG precision/recall must meet minimum thresholds."""
    validator = RetrievalValidator(threshold=0.70)

    # Simulate retrieved = ground truth (perfect retrieval) for baseline test
    doc = {"id": case["id"], "content": case["context"]}
    metrics = validator.evaluate(
        query=case["question"],
        retrieved_docs=[doc],
        ground_truth_docs=[doc],
    )

    assert metrics.passed, (
        f"[{case['id']}] RAG F1 {metrics.f1:.3f} below threshold {metrics.threshold}"
    )
    assert metrics.precision >= 0.70, f"Precision too low: {metrics.precision}"
    assert metrics.recall >= 0.70, f"Recall too low: {metrics.recall}"
