# GenAI Quality Evaluation Framework

> CI-ready quality gates for LLM · RAG · Latency · Hallucination · Regression

[![CI](https://github.com/MarcelinoSoares/genai-quality-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcelinoSoares/genai-quality-eval/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **pure-Python** evaluation framework for Generative AI systems, designed to bring Quality Engineering practices into LLM and RAG delivery pipelines.

It provides modular evaluators for response quality, hallucination risk, retrieval accuracy, latency SLAs, and prompt regression testing — so teams can detect regressions, enforce quality thresholds, and make GenAI releases more reliable.

No web server. No database. Just composable modules that can run locally, inside automated tests, or as CI/CD quality gates.

---

## Why this framework?

| Problem | Solution |
|---|---|
| "Did the model regress after the upgrade?" | Prompt regression suite with versioned baselines and ≤ 5% tolerance |
| "Is this answer hallucinated?" | Heuristic context-grounding score, no LLM required |
| "Are we hitting our latency SLA?" | Context-manager tracker with p50/p95/p99 and `assert_sla()` |
| "How good is our RAG retrieval?" | Precision@K, Recall@K, MRR, F1, and context coverage in one call |
| "What does the judge score say?" | LLM-as-judge with relevance, coherence, and faithfulness |

---

## Quality Engineering Use Cases

- Detect prompt regressions after model, prompt, or retrieval changes
- Enforce latency SLAs for GenAI features in CI/CD pipelines
- Validate RAG retrieval quality before production deployment
- Identify hallucination risk using context-grounding heuristics
- Compare LLM response quality across versions or providers
- Generate quality signals for release readiness decisions

---

## Architecture

```
genai-quality-eval/
├── src/
│   └── genai_quality_eval/
│       ├── evaluators/
│       │   └── llm_evaluator.py        # LLM-as-judge: relevance, coherence, faithfulness
│       ├── rag/
│       │   └── retrieval_validator.py  # Precision@K, Recall@K, MRR, F1, context coverage
│       ├── metrics/latency/
│       │   └── latency_tracker.py      # Context-manager timer · p50/p95/p99 · SLA assertions
│       └── hallucination/
│           └── risk_detector.py        # Token + n-gram overlap heuristics, no LLM required
├── tests/
│   ├── unit/                   # Fast unit tests (no LLM calls)
│   └── regression/
│       └── test_prompt_regression.py   # Versioned baseline regression suite
├── examples/
│   └── quick_start.py          # Offline demo — all 4 modules, no API key required
├── docs/
│   └── design-decisions.md     # Engineering rationale for key architectural choices
└── .github/workflows/ci.yml    # lint → unit → regression ‖ benchmarks → quality-gate
```

Each module is **independently usable** — import only what you need. See [`docs/design-decisions.md`](docs/design-decisions.md) for the engineering rationale behind key choices.

---

## Quickstart

```bash
git clone https://github.com/MarcelinoSoares/genai-quality-eval.git
cd genai-quality-eval

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run everything
PYTHONPATH=src pytest tests/ -v --tb=short

# Unit tests only (no LLM calls, runs offline)
PYTHONPATH=src pytest tests/ --ignore=tests/regression -v --tb=short

# Regression suite only
PYTHONPATH=src pytest tests/regression/ -v --tb=long

# Filter by name
PYTHONPATH=src pytest tests/ -k "latency or benchmark" -v
```

---

## Examples

See the [`examples/`](examples/) directory for runnable scripts covering all modules:

- LLM-as-judge evaluation
- Hallucination risk detection
- RAG retrieval validation
- Latency SLA checks
- Prompt regression testing

Run the quick-start demo offline (no API key required):

```bash
PYTHONPATH=src python examples/quick_start.py
```

---

## Modules

### LLM Evaluator — `genai_quality_eval.evaluators.llm_evaluator`

LLM-as-judge pattern. Calls an OpenAI model to score three dimensions and returns a structured result with a `passed` flag.

```python
from genai_quality_eval.evaluators.llm_evaluator import LLMEvaluator

evaluator = LLMEvaluator(model="gpt-4o", threshold=0.75)
result = evaluator.evaluate(
    question="What is RAG?",
    answer="RAG enhances LLMs by retrieving relevant documents as context.",
    context="RAG combines retrieval systems with language models...",
)

print(result.scores)        # {"relevance": 0.92, "coherence": 0.88, "faithfulness": 0.95}
print(result.average_score) # 0.916
print(result.passed)        # True  (average >= threshold)
print(result.latency_ms)    # 1243.7
```

**Scores:**
- `relevance` — how well the answer addresses the question
- `coherence` — logical structure and readability
- `faithfulness` — factual grounding relative to provided context (only scored when `context` is supplied)

`passed = average_score >= threshold` (default `0.75`)

---

### Hallucination Detector — `genai_quality_eval.hallucination.risk_detector`

No LLM required. Measures how grounded an answer is in its context using token overlap and n-gram overlap heuristics.

```python
from genai_quality_eval.hallucination.risk_detector import HallucinationDetector

detector = HallucinationDetector(risk_threshold=0.30)
result = detector.score(
    answer="The Eiffel Tower was built in 1850.",
    context="The Eiffel Tower was completed in 1889.",
)

print(result.risk_score)        # 0.73
print(result.risk_level)        # RiskLevel.HIGH
print(result.unverified_claims) # ["The Eiffel Tower was built in 1850."]
print(result.passed)            # False  (risk_score >= threshold)
```

**Risk score formula:**

```
risk_score = 0.9 × token_ungroundedness + 0.1 × ngram_ungroundedness
```

| Risk Level | Score Range |
|---|---|
| LOW | < 0.30 |
| MEDIUM | 0.30 – 0.59 |
| HIGH | ≥ 0.60 |

Suspicious sentences (< 40% token overlap with context) are surfaced in `unverified_claims`.

---

### RAG Retrieval Validator — `genai_quality_eval.rag.retrieval_validator`

Validates retrieval quality for RAG pipelines. Documents are matched by their `id` field (dict) or string identity.

```python
from genai_quality_eval.rag.retrieval_validator import RetrievalValidator

validator = RetrievalValidator(threshold=0.70)
metrics = validator.evaluate(
    query="Explain transformer architecture",
    retrieved_docs=[{"id": "doc1", "content": "..."}],
    ground_truth_docs=[{"id": "doc1", "content": "..."}, {"id": "doc2", "content": "..."}],
)

print(metrics.precision)        # 1.0
print(metrics.recall)           # 0.5
print(metrics.mrr)              # 1.0
print(metrics.f1)               # 0.6667
print(metrics.context_coverage) # 0.84
print(metrics.passed)           # False  (F1 < threshold)
```

**Metrics computed:**

| Metric | Description |
|---|---|
| `precision` | Fraction of retrieved docs that are relevant |
| `recall` | Fraction of relevant docs that were retrieved |
| `mrr` | Mean Reciprocal Rank — rank of the first relevant result |
| `f1` | Harmonic mean of precision and recall |
| `context_coverage` | Token-overlap coverage of retrieved vs ground-truth text |

`passed = f1 >= threshold` (default `0.70`)

---

### Latency Tracker — `genai_quality_eval.metrics.latency.latency_tracker`

Context-manager timer with percentile statistics and SLA enforcement. `SLAViolationError` (subclass of `AssertionError`) is raised on violation.

```python
from genai_quality_eval.metrics.latency.latency_tracker import LatencyTracker, SLAViolationError

tracker = LatencyTracker(sla_threshold_ms=2000)

for prompt in prompts:
    with tracker.measure() as m:
        response = llm.generate(prompt)
    m.assert_sla()  # raises SLAViolationError if this call exceeded 2000ms

# Aggregate stats after N calls
report = tracker.report()
# {
#   "count": 50, "mean_ms": 1320.4,
#   "p50_ms": 1280.0, "p95_ms": 1890.0, "p99_ms": 2100.0,
#   "sla_threshold_ms": 2000.0, "sla_violations": 1
# }

tracker.assert_p95_sla()  # raises if p95 > 2000ms
```

---

## Regression Testing

The regression suite in `tests/regression/test_prompt_regression.py` defines a `PROMPT_SUITE` of canonical question/answer/context triples with versioned `baseline_scores` committed alongside the tests.

**In CI, `LLMEvaluator._call_llm` is monkeypatched** — baselines are returned deterministically, so regression tests run offline with no OpenAI key needed.

A test fails when any score drops more than `REGRESSION_TOLERANCE` (default 5%) below its baseline:

```
Regression detected in 'faithfulness':
  baseline=0.880, current=0.820, drop=0.060 (tolerance=5%)
```

To update baselines after an intentional model change, regenerate the baseline scores, review the diff, and commit the updated values. Use `REGRESSION_TOLERANCE` to widen the tolerance window during transition:

```bash
PYTHONPATH=src REGRESSION_TOLERANCE=0.10 pytest tests/regression/ -v   # 10% tolerance
```

**Three test classes run per suite entry:**

| Test | What it checks |
|---|---|
| `test_score_regression_against_baseline` | Scores must not drop > 5% vs versioned baselines |
| `test_hallucination_regression` | Hallucination risk must stay below HIGH (< 0.60) |
| `test_rag_retrieval_regression` | RAG Precision, Recall, F1 must all meet ≥ 0.70 |

---

## Example Quality Gate Output

```json
{
  "faithfulness": 0.87,
  "hallucination_risk": 0.18,
  "rag_f1": 0.76,
  "latency_p95_ms": 1840,
  "prompt_regression_drop": 0.02,
  "quality_gate": "passed"
}
```

This output can be used in CI/CD pipelines to block risky GenAI releases when quality thresholds are not met.

---

## CI/CD Pipeline

Every push to `main` / `develop` and every pull request triggers the full evaluation pipeline:

```
[Push / PR]
    │
    ▼
[1] Lint & Type Check          ruff check · ruff format · mypy
    │
    ▼
[2] Unit Tests                 pytest tests/ --ignore=tests/regression
    │                          coverage reports uploaded as artifacts
    ├───────────────────────────────────────┐
    ▼                                       ▼
[3] Prompt Regression Tests     [4] Latency Benchmarks
    pytest tests/regression/        pytest -k "latency or benchmark"
    REGRESSION_TOLERANCE=0.05       pytest-benchmark --benchmark-json
    │                                       │
    └──────────────────┬────────────────────┘
                       ▼
               [5] Quality Gate
                   Summary report to GitHub Step Summary
```

Jobs 3 and 4 run in parallel after unit tests pass.

### Quality gate thresholds

| Metric | Threshold |
|---|---|
| Faithfulness | ≥ 0.80 |
| Hallucination risk | < 0.30 |
| Latency p95 | ≤ 3000 ms |
| Prompt regression | ≤ 5% degradation |
| RAG retrieval F1 | ≥ 0.70 |

---

## Tech Stack

| Layer | Libraries |
|---|---|
| LLM providers | `openai`, `anthropic` |
| RAG orchestration | `langchain`, `langchain-openai` |
| Semantic similarity | `sentence-transformers` |
| NLP metrics | `evaluate` (HuggingFace), `rouge-score`, `nltk` |
| Testing | `pytest`, `pytest-cov`, `pytest-benchmark`, `pytest-mock` |
| Linting / types | `ruff`, `mypy` |
| Utilities | `numpy`, `pandas`, `rich`, `python-dotenv` |
| CI | GitHub Actions |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for `LLMEvaluator` in live mode (OpenAI) |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic models in live mode |
| `REGRESSION_TOLERANCE` | `0.05` | Maximum allowed score drop in regression tests |

---

## Author

**Marcelino Soares de Oliveira**
Senior SDET / Quality Engineer at Thoughtworks · GenAI Quality · Test Automation · CI/CD
[LinkedIn](https://www.linkedin.com/in/marcelinosoares) · [GitHub](https://github.com/MarcelinoSoares)

---

## License

MIT — see [LICENSE](LICENSE) for details.
