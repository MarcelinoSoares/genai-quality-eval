# GenAI Quality Evaluation Framework

> LLM + RAG Testing, Latency, Accuracy & Regression

[![CI](https://github.com/MarcelinoSoares/genai-quality-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcelinoSoares/genai-quality-eval/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready quality evaluation framework for **Generative AI systems**, covering LLM response assessment, RAG/retrieval validation, prompt regression testing, latency metrics, accuracy scoring, and hallucination risk detection — with a fully automated CI/CD pipeline.

---

## Features

| Module | Description |
|---|---|
| `evaluators/` | LLM response quality scoring (relevance, coherence, faithfulness) |
| `tests/regression/` | Prompt regression tests — detect quality degradation across model versions |
| `rag/` | RAG pipeline validation: retrieval precision, context coverage, answer grounding |
| `metrics/latency/` | End-to-end latency measurement and SLA threshold assertions |
| `metrics/accuracy/` | Accuracy, F1, BLEU, ROUGE and semantic similarity scoring |
| `hallucination/` | Hallucination risk detection via fact-checking and confidence calibration |
| `.github/workflows/` | CI/CD pipeline running all evaluations on every push/PR |

---

## Project Structure

```
genai-quality-eval/
├── evaluators/
│   ├── llm_evaluator.py        # LLM response quality scorer
│   ├── coherence_checker.py    # Coherence & relevance checks
│   └── faithfulness_scorer.py  # Answer faithfulness to context
├── rag/
│   ├── retrieval_validator.py  # Retrieval precision & recall
│   ├── context_coverage.py     # Context coverage metrics
│   └── answer_grounding.py     # Answer grounding verification
├── metrics/
│   ├── latency/
│   │   ├── latency_tracker.py  # Response time measurement
│   │   └── sla_assertions.py   # SLA threshold enforcement
│   └── accuracy/
│       ├── semantic_scorer.py  # Semantic similarity (cosine/BERTScore)
│       └── nlp_metrics.py      # BLEU, ROUGE, F1 computation
├── hallucination/
│   ├── risk_detector.py        # Hallucination risk scoring
│   └── fact_checker.py         # Fact-checking pipeline
├── tests/
│   ├── regression/
│   │   ├── test_prompt_regression.py
│   │   └── prompt_snapshots/   # Baseline prompt snapshots
│   ├── test_evaluators.py
│   ├── test_rag.py
│   ├── test_metrics.py
│   └── test_hallucination.py
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/MarcelinoSoares/genai-quality-eval.git
cd genai-quality-eval

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v --tb=short

# Run only regression tests
pytest tests/regression/ -v

# Run with latency report
pytest tests/ -v --benchmark-autosave
```

---

## Evaluation Modules

### LLM Response Evaluator

```python
from evaluators.llm_evaluator import LLMEvaluator

evaluator = LLMEvaluator(model="gpt-4o")
result = evaluator.evaluate(
    question="What is RAG?",
    answer="RAG stands for Retrieval-Augmented Generation...",
    context="..."
)
print(result.scores)  # {relevance: 0.92, coherence: 0.88, faithfulness: 0.95}
```

### RAG Validation

```python
from rag.retrieval_validator import RetrievalValidator

validator = RetrievalValidator()
metrics = validator.evaluate(
    query="Explain transformer architecture",
    retrieved_docs=[...],
    ground_truth_docs=[...]
)
print(metrics)  # {precision: 0.85, recall: 0.90, mrr: 0.87}
```

### Latency Tracking

```python
from metrics.latency.latency_tracker import LatencyTracker

tracker = LatencyTracker(sla_threshold_ms=2000)
with tracker.measure() as m:
    response = llm.generate(prompt)

m.assert_sla()  # Raises AssertionError if > 2000ms
print(m.duration_ms)  # e.g. 1430.5
```

### Hallucination Risk Detection

```python
from hallucination.risk_detector import HallucinationDetector

detector = HallucinationDetector()
risk = detector.score(
    answer="The Eiffel Tower was built in 1850.",
    context="The Eiffel Tower was completed in 1889."
)
print(risk)  # {risk_score: 0.91, risk_level: "HIGH"}
```

---

## CI/CD Pipeline

Every push and pull request triggers the full evaluation suite:

```
[Push/PR] → [Lint & Type Check] → [Unit Tests] → [Regression Tests] → [Metrics Report] → [Quality Gate]
```

Quality gate thresholds (configurable in `pyproject.toml`):
- Minimum faithfulness score: **0.80**
- Maximum latency p95: **3000ms**
- Hallucination risk threshold: **< 0.30**
- Prompt regression tolerance: **5%** degradation

---

## Tech Stack

- **Python 3.11+**
- **pytest** + **pytest-benchmark** — test runner & latency benchmarks
- **LangChain** — LLM & RAG orchestration
- **sentence-transformers** — semantic similarity
- **evaluate** (HuggingFace) — BLEU, ROUGE, BERTScore
- **OpenAI / Anthropic / Ollama** — LLM provider adapters
- **GitHub Actions** — CI/CD automation

---

## Author

**Marcelino Soares de Oliveira**  
QA at Thoughtworks | Agile Testing Specialist | Doctor in Computer Science  
[LinkedIn](https://www.linkedin.com/in/marcelinosoares) · [GitHub](https://github.com/MarcelinoSoares)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
