"""Hallucination Risk Detector.

Detects potential hallucinations in LLM responses by comparing the answer
against the provided context using token overlap and semantic heuristics.

Risk levels:
  LOW    : score < 0.30
  MEDIUM : 0.30 <= score < 0.60
  HIGH   : score >= 0.60
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "it",
        "its",
        "that",
        "this",
        "which",
        "so",
        "can",
        "not",
        "more",
        "they",
        "them",
        "their",
        "then",
        "than",
        "when",
        "what",
        "how",
    }
)


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class HallucinationResult:
    answer: str
    context: str
    risk_score: float  # 0.0 (grounded) -> 1.0 (ungrounded)
    risk_level: RiskLevel
    unverified_claims: List[str]
    passed: bool  # True when risk_score < threshold


def _stem(word: str) -> str:
    """Minimal suffix-stripping stemmer (handles the most common English endings)."""
    for suffix in (
        "ing",
        "tions",
        "ion",
        "ness",
        "ment",
        "ments",
        "al",
        "ed",
        "ly",
        "er",
        "es",
        "e",
        "s",
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


class HallucinationDetector:
    """Detects hallucination risk using context-grounding analysis."""

    def __init__(
        self,
        risk_threshold: float = 0.30,
        min_ngram: int = 3,
    ) -> None:
        self.risk_threshold = risk_threshold
        self.min_ngram = min_ngram

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.sub(r"[^\w\s-]", "", text.lower()).split()
        return [_stem(t) for t in tokens if t not in _STOPWORDS]

    def _ngrams(self, tokens: List[str], n: int) -> set:
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    def _token_overlap_score(self, answer: str, context: str) -> float:
        """Fraction of answer tokens NOT found in context (ungroundedness proxy)."""
        answer_tokens = set(self._tokenize(answer))
        context_tokens = set(self._tokenize(context))
        if not answer_tokens:
            return 0.0
        ungrounded = answer_tokens - context_tokens
        return len(ungrounded) / len(answer_tokens)

    def _ngram_overlap_score(self, answer: str, context: str, n: int = 3) -> float:
        """Fraction of answer n-grams NOT present in context."""
        answer_ng = self._ngrams(self._tokenize(answer), n)
        context_ng = self._ngrams(self._tokenize(context), n)
        if not answer_ng:
            return 0.0
        ungrounded = answer_ng - context_ng
        return len(ungrounded) / len(answer_ng)

    def _extract_suspicious_sentences(self, answer: str, context: str) -> List[str]:
        """Return sentences whose tokens are poorly covered by context."""
        context_tokens = set(self._tokenize(context))
        suspicious = []
        for sentence in answer.replace(".", ".|").replace("!", "!|").replace("?", "?|").split("|"):
            sentence = sentence.strip()
            if not sentence:
                continue
            tokens = set(self._tokenize(sentence))
            if not tokens:
                continue
            overlap = len(tokens & context_tokens) / len(tokens)
            if overlap < 0.4:  # less than 40% of words found in context
                suspicious.append(sentence)
        return suspicious

    def _classify_risk(self, score: float) -> RiskLevel:
        if score < 0.30:
            return RiskLevel.LOW
        if score < 0.60:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        answer: str,
        context: str,
    ) -> HallucinationResult:
        """Compute hallucination risk for an answer given a context.

        Args:
            answer:  The LLM-generated answer to evaluate.
            context: The retrieved context the answer should be grounded in.

        Returns:
            HallucinationResult with risk_score, risk_level and suspicious claims.
        """
        token_score = self._token_overlap_score(answer, context)
        ngram_score = self._ngram_overlap_score(answer, context, n=self.min_ngram)

        # Weighted combination: token overlap is the primary signal;
        # n-gram overlap adds signal only for near-verbatim hallucinations.
        risk_score = round(0.9 * token_score + 0.1 * ngram_score, 4)
        risk_level = self._classify_risk(risk_score)
        unverified = self._extract_suspicious_sentences(answer, context)

        return HallucinationResult(
            answer=answer,
            context=context,
            risk_score=risk_score,
            risk_level=risk_level,
            unverified_claims=unverified,
            passed=risk_score < self.risk_threshold,
        )
