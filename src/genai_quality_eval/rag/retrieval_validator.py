"""RAG Retrieval Validator.

Measures retrieval quality for Retrieval-Augmented Generation pipelines:
- Precision@K: fraction of retrieved docs that are relevant
- Recall@K: fraction of relevant docs that were retrieved
- Mean Reciprocal Rank (MRR): rank of the first relevant document
- Context Coverage: semantic coverage of retrieved context vs ground truth
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RetrievalMetrics:
    precision: float
    recall: float
    mrr: float
    f1: float
    context_coverage: float
    passed: bool = False
    threshold: float = 0.70

    def __post_init__(self) -> None:
        self.passed = self.f1 >= self.threshold


class RetrievalValidator:
    """Validates RAG retrieval quality against ground-truth document sets."""

    def __init__(self, threshold: float = 0.70) -> None:
        self.threshold = threshold

    def _doc_id(self, doc) -> str:
        """Extract a stable identifier from a document (str or dict)."""
        if isinstance(doc, dict):
            return doc.get("id") or doc.get("content", str(doc))
        return str(doc)

    def precision_at_k(self, retrieved: List, relevant_ids: set) -> float:
        """Fraction of retrieved documents that are relevant."""
        if not retrieved:
            return 0.0
        hits = sum(1 for d in retrieved if self._doc_id(d) in relevant_ids)
        return hits / len(retrieved)

    def recall_at_k(self, retrieved: List, relevant_ids: set) -> float:
        """Fraction of relevant documents that were retrieved."""
        if not relevant_ids:
            return 0.0
        retrieved_ids = {self._doc_id(d) for d in retrieved}
        return len(retrieved_ids & relevant_ids) / len(relevant_ids)

    def mean_reciprocal_rank(self, retrieved: List, relevant_ids: set) -> float:
        """MRR: inverse rank of the first relevant result (1-indexed)."""
        for rank, doc in enumerate(retrieved, start=1):
            if self._doc_id(doc) in relevant_ids:
                return 1.0 / rank
        return 0.0

    def context_coverage(self, retrieved: List, ground_truth: List) -> float:
        """Simple token-overlap coverage between retrieved and ground-truth text."""
        retrieved_text = " ".join(
            d.get("content", str(d)) if isinstance(d, dict) else str(d) for d in retrieved
        )
        gt_text = " ".join(
            d.get("content", str(d)) if isinstance(d, dict) else str(d) for d in ground_truth
        )
        gt_tokens = set(gt_text.lower().split())
        retrieved_tokens = set(retrieved_text.lower().split())
        if not gt_tokens:
            return 1.0
        return len(retrieved_tokens & gt_tokens) / len(gt_tokens)

    def evaluate(
        self,
        query: str,
        retrieved_docs: List,
        ground_truth_docs: List,
    ) -> RetrievalMetrics:
        """Compute all retrieval metrics and return a structured result."""
        relevant_ids = {self._doc_id(d) for d in ground_truth_docs}

        p = self.precision_at_k(retrieved_docs, relevant_ids)
        r = self.recall_at_k(retrieved_docs, relevant_ids)
        mrr = self.mean_reciprocal_rank(retrieved_docs, relevant_ids)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        coverage = self.context_coverage(retrieved_docs, ground_truth_docs)

        return RetrievalMetrics(
            precision=round(p, 4),
            recall=round(r, 4),
            mrr=round(mrr, 4),
            f1=round(f1, 4),
            context_coverage=round(coverage, 4),
            threshold=self.threshold,
        )
