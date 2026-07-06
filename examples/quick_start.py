"""Quick-start demo — runs fully offline, no API key required.

Demonstrates all four evaluation modules:
  1. HallucinationDetector  — context-grounding heuristics
  2. RetrievalValidator     — RAG retrieval metrics
  3. LatencyTracker         — p50/p95/p99 SLA enforcement
  4. LLMEvaluator           — LLM-as-judge (offline subclass, no API key)

Run:
    PYTHONPATH=. python examples/quick_start.py
"""

from __future__ import annotations

import time

from genai_quality_eval.evaluators.llm_evaluator import LLMEvaluator
from genai_quality_eval.hallucination.risk_detector import HallucinationDetector
from genai_quality_eval.metrics.latency.latency_tracker import LatencyTracker
from genai_quality_eval.rag.retrieval_validator import RetrievalValidator


class _OfflineLLMEvaluator(LLMEvaluator):
    """Offline subclass — returns hardcoded scores, no API key needed."""

    def _build_client(self):
        return None

    def _call_llm(self, prompt: str) -> float:
        first_line = prompt.split("\n")[0].lower()
        if "faithfully" in first_line or "faithfulness" in first_line:
            return 0.95
        if "coherence" in first_line:
            return 0.88
        return 0.92


# ---------------------------------------------------------------------------
# 1. Hallucination Detector
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Hallucination Detector")
print("=" * 60)

detector = HallucinationDetector(risk_threshold=0.30)

grounded_result = detector.score(
    answer="RAG combines retrieval systems with language models to ground responses in retrieved documents.",
    context="Retrieval-Augmented Generation (RAG) enhances language models by fetching relevant documents at inference time and including them as context.",
)
print(
    f"  [grounded]    risk_score={grounded_result.risk_score:.3f}  level={grounded_result.risk_level.value}  passed={grounded_result.passed}"
)

hallucinated_result = detector.score(
    answer="The Eiffel Tower was built in 1850 by Napoleon Bonaparte as a military fortress.",
    context="The Eiffel Tower was completed in 1889 and designed by Gustave Eiffel as the entrance arch for the 1889 World's Fair.",
)
print(
    f"  [hallucinated] risk_score={hallucinated_result.risk_score:.3f}  level={hallucinated_result.risk_level.value}  passed={hallucinated_result.passed}"
)
if hallucinated_result.unverified_claims:
    print(f"  unverified: {hallucinated_result.unverified_claims[0][:80]}")

# ---------------------------------------------------------------------------
# 2. RAG Retrieval Validator
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. RAG Retrieval Validator")
print("=" * 60)

validator = RetrievalValidator(threshold=0.70)

perfect_metrics = validator.evaluate(
    query="Explain transformer architecture",
    retrieved_docs=[
        {
            "id": "doc1",
            "content": "Transformers use self-attention mechanisms to process sequences in parallel.",
        },
        {
            "id": "doc2",
            "content": "The encoder-decoder structure allows transformers to handle sequence-to-sequence tasks.",
        },
    ],
    ground_truth_docs=[
        {
            "id": "doc1",
            "content": "Transformers use self-attention mechanisms to process sequences in parallel.",
        },
        {
            "id": "doc2",
            "content": "The encoder-decoder structure allows transformers to handle sequence-to-sequence tasks.",
        },
    ],
)
print(
    f"  [perfect]  precision={perfect_metrics.precision:.2f}  recall={perfect_metrics.recall:.2f}  f1={perfect_metrics.f1:.2f}  passed={perfect_metrics.passed}"
)

partial_metrics = validator.evaluate(
    query="Explain transformer architecture",
    retrieved_docs=[
        {
            "id": "doc1",
            "content": "Transformers use self-attention mechanisms to process sequences in parallel.",
        },
    ],
    ground_truth_docs=[
        {
            "id": "doc1",
            "content": "Transformers use self-attention mechanisms to process sequences in parallel.",
        },
        {
            "id": "doc2",
            "content": "The encoder-decoder structure allows transformers to handle sequence-to-sequence tasks.",
        },
    ],
)
print(
    f"  [partial]  precision={partial_metrics.precision:.2f}  recall={partial_metrics.recall:.2f}  f1={partial_metrics.f1:.2f}  passed={partial_metrics.passed}"
)

# ---------------------------------------------------------------------------
# 3. Latency Tracker
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. Latency Tracker")
print("=" * 60)

tracker = LatencyTracker(sla_threshold_ms=500)

simulated_durations_ms = [120, 145, 132, 158, 143, 201, 178, 165, 149, 310]
for duration_ms in simulated_durations_ms:
    with tracker.measure() as m:  # noqa: F841
        time.sleep(duration_ms / 1000)

report = tracker.report()
print(
    f"  calls={report['count']}  mean={report['mean_ms']:.1f}ms  p50={report['p50_ms']:.1f}ms  p95={report['p95_ms']:.1f}ms  p99={report['p99_ms']:.1f}ms"
)
print(
    f"  sla_threshold={report['sla_threshold_ms']:.0f}ms  violations={report['sla_violations']}"
)

try:
    tracker.assert_p95_sla()
    print("  p95 SLA: PASSED")
except Exception as e:
    print(f"  p95 SLA: FAILED — {e}")

# ---------------------------------------------------------------------------
# 4. LLM Evaluator (offline demo)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. LLM Evaluator (offline demo)")
print("=" * 60)

evaluator = _OfflineLLMEvaluator(model="gpt-4o", threshold=0.75)

result = evaluator.evaluate(
    question="What is RAG?",
    answer="RAG enhances LLMs by retrieving relevant documents as context before generating responses.",
    context="Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model.",
)
print(f"  scores={result.scores}")
print(f"  average={result.average_score:.3f}  passed={result.passed}")

print()
print("All modules exercised successfully.")
