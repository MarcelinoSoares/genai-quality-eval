# Design Decisions

Engineering choices behind the GenAI Quality Evaluation Framework and the reasoning for each.

---

## 1. Heuristic hallucination detection — no LLM required

**Decision:** `HallucinationDetector` uses token overlap and n-gram overlap heuristics instead of a second LLM call.

**Why:** LLM-based hallucination checkers are slow, expensive, and introduce a dependency on an external service that may itself hallucinate. Token-overlap heuristics are deterministic, fast, and reproducible — making them safe to run in every CI job without incurring API costs or flakiness. The trade-off is lower semantic accuracy: the heuristic misses paraphrased hallucinations. For a CI gate, a conservative false-positive rate is acceptable; the goal is to catch obvious groundedness failures, not achieve human-level accuracy.

---

## 2. Versioned baseline scores committed alongside tests

**Decision:** `PROMPT_SUITE` in `test_prompt_regression.py` stores `baseline_scores` as literals in the test file, tracked in version control.

**Why:** Baselines must be intentional. Storing them as a separate artefact (database, JSON file, external service) decouples them from the code change that justified updating them, making regressions invisible. Committing baselines alongside tests means every baseline update appears in code review as a deliberate diff, alongside the model or prompt change that caused it. The `REGRESSION_TOLERANCE` env var provides a controlled escape hatch for major model upgrades without requiring a baseline rewrite.

---

## 3. LLM-as-judge monkeypatched in CI

**Decision:** In regression tests, `LLMEvaluator._call_llm` is monkeypatched to return baseline scores deterministically. No real LLM call is made in CI.

**Why:** CI must be fast, cheap, and reproducible. Real LLM calls introduce latency (~1–3 s per call), cost, and non-determinism — all unacceptable for a quality gate that runs on every push. The monkeypatching strategy tests the scoring and regression logic in full while keeping the test suite dependency-free. Live LLM evaluation is reserved for local pre-release validation or scheduled nightly runs.

---

## 4. Quality gate thresholds are explicit, not inferred

**Decision:** Thresholds (faithfulness ≥ 0.80, hallucination risk < 0.30, latency p95 ≤ 3000 ms, etc.) are hardcoded in the CI pipeline and documented in the README, not learned from historical data.

**Why:** Learned thresholds can silently shift as the system degrades — a model that consistently underperforms will produce a "normal" baseline that is actually low quality. Explicit thresholds encode an engineering contract: if the team lowers a threshold, it appears as a deliberate commit. This makes quality regressions visible and creates a shared language between engineers and stakeholders about what "good" means.

---

## 5. F1 as the primary RAG retrieval gate, not precision alone

**Decision:** `RetrievalValidator.evaluate()` sets `passed = f1 >= threshold`, not `precision >= threshold`.

**Why:** Precision-only gating rewards shallow retrieval — a system that returns one highly relevant document can achieve precision 1.0 while missing 80% of what the user needed. F1 penalises both over-retrieval (low precision) and under-retrieval (low recall), making it a better single-metric proxy for retrieval health. MRR and context coverage are surfaced alongside F1 for diagnostic purposes but are not part of the pass/fail gate.

---

## 6. `SLAViolationError` is a subclass of `AssertionError`

**Decision:** `SLAViolationError(AssertionError)` instead of `SLAViolationError(Exception)`.

**Why:** Latency assertions (`assert_sla()`, `assert_p95_sla()`) are quality assertions in the same spirit as `pytest.fail()`. Making `SLAViolationError` a subclass of `AssertionError` means pytest treats it identically to any other assertion failure — it produces a clean diff-style output, is captured by `pytest.raises(AssertionError)`, and integrates naturally into test reports without special handling.

---

## 7. Flat module structure under `src/genai_quality_eval/`

**Decision:** Each evaluator lives in its own sub-package (`evaluators/`, `rag/`, `hallucination/`, `metrics/latency/`) rather than a single flat module.

**Why:** Evaluators are independently usable. A team that only needs latency tracking should not have to import — or depend on — the OpenAI client or the sentence-transformers library. The sub-package structure makes that independence explicit at the import level and allows future optional-dependency gating (e.g., `pip install genai-quality-eval[llm]`) without restructuring the codebase.
