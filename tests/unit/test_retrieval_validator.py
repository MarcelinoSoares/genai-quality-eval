"""Unit tests for RetrievalValidator — covers empty-input edge cases."""

from __future__ import annotations

import pytest

from rag.retrieval_validator import RetrievalValidator


@pytest.fixture
def validator():
    return RetrievalValidator(threshold=0.70)


def test_doc_id_dict_without_id_key(validator):
    doc = {"content": "some text"}
    assert validator._doc_id(doc) == "some text"


def test_doc_id_dict_with_id_key(validator):
    doc = {"id": "doc-1", "content": "text"}
    assert validator._doc_id(doc) == "doc-1"


def test_doc_id_string(validator):
    assert validator._doc_id("plain-string") == "plain-string"


def test_precision_at_k_empty_retrieved(validator):
    assert validator.precision_at_k([], {"doc-1"}) == 0.0


def test_recall_at_k_empty_relevant_ids(validator):
    assert validator.recall_at_k([{"id": "doc-1"}], set()) == 0.0


def test_mean_reciprocal_rank_no_relevant_found(validator):
    retrieved = [{"id": "doc-a"}, {"id": "doc-b"}]
    assert validator.mean_reciprocal_rank(retrieved, {"doc-x"}) == 0.0


def test_context_coverage_empty_ground_truth(validator):
    assert validator.context_coverage([{"content": "anything"}], []) == 1.0


def test_mean_reciprocal_rank_first_hit(validator):
    retrieved = [{"id": "doc-1"}, {"id": "doc-2"}]
    assert validator.mean_reciprocal_rank(retrieved, {"doc-1"}) == pytest.approx(1.0)


def test_mean_reciprocal_rank_second_hit(validator):
    retrieved = [{"id": "doc-x"}, {"id": "doc-1"}]
    assert validator.mean_reciprocal_rank(retrieved, {"doc-1"}) == pytest.approx(0.5)


def test_evaluate_perfect_retrieval(validator):
    doc = {"id": "d1", "content": "context text here"}
    metrics = validator.evaluate("query", [doc], [doc])
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.passed is True
