"""LLM Response Quality Evaluator.

Scores LLM responses across three dimensions:
- Relevance: how well the answer addresses the question
- Coherence: logical structure and readability
- Faithfulness: factual grounding relative to provided context
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvaluationResult:
    question: str
    answer: str
    context: Optional[str] = None
    scores: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    passed: bool = False
    threshold: float = 0.75

    def __post_init__(self) -> None:
        if self.scores:
            avg = sum(self.scores.values()) / len(self.scores)
            self.passed = avg >= self.threshold

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


class LLMEvaluator:
    """Evaluates LLM response quality using an LLM-as-judge pattern."""

    RELEVANCE_PROMPT = (
        "Rate the relevance of the answer to the question on a scale from 0.0 to 1.0.\n"
        "Question: {question}\nAnswer: {answer}\nScore (float only):"
    )
    COHERENCE_PROMPT = (
        "Rate the coherence and clarity of the following answer on a scale from 0.0 to 1.0.\n"
        "Answer: {answer}\nScore (float only):"
    )
    FAITHFULNESS_PROMPT = (
        "Rate how faithfully the answer is grounded in the provided context on a scale from 0.0 to 1.0.\n"
        "Context: {context}\nAnswer: {answer}\nScore (float only):"
    )

    def __init__(self, model: str = "gpt-4o", threshold: float = 0.75) -> None:
        self.model = model
        self.threshold = threshold
        self._client = self._build_client()

    def _build_client(self):
        """Build the LLM client. Override to inject custom clients in tests."""
        try:
            from openai import OpenAI  # type: ignore

            return OpenAI()
        except ImportError:
            return None

    def _call_llm(self, prompt: str) -> float:
        """Call the LLM and parse a float score from its response."""
        if self._client is None:
            raise RuntimeError("OpenAI client not available. Install 'openai' package.")
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        return float(raw)

    def evaluate(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate an LLM answer and return structured scores."""
        start = time.perf_counter()
        scores: dict[str, float] = {}

        scores["relevance"] = self._call_llm(
            self.RELEVANCE_PROMPT.format(question=question, answer=answer)
        )
        scores["coherence"] = self._call_llm(self.COHERENCE_PROMPT.format(answer=answer))
        if context:
            scores["faithfulness"] = self._call_llm(
                self.FAITHFULNESS_PROMPT.format(context=context, answer=answer)
            )

        latency_ms = (time.perf_counter() - start) * 1000

        return EvaluationResult(
            question=question,
            answer=answer,
            context=context,
            scores=scores,
            latency_ms=latency_ms,
            threshold=self.threshold,
        )
