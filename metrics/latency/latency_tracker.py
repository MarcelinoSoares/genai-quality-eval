"""Latency Tracker for GenAI pipelines.

Measures end-to-end response time and enforces SLA thresholds.
Supports context manager usage and percentile statistics across runs.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List


class SLAViolationError(AssertionError):
    """Raised when a measured latency exceeds the SLA threshold."""


@dataclass
class LatencyMeasurement:
    duration_ms: float
    sla_threshold_ms: float
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.passed = self.duration_ms <= self.sla_threshold_ms

    def assert_sla(self) -> None:
        """Raise SLAViolationError if duration exceeds the threshold."""
        if not self.passed:
            raise SLAViolationError(
                f"Latency {self.duration_ms:.1f}ms exceeded SLA of "
                f"{self.sla_threshold_ms:.1f}ms "
                f"(+{self.duration_ms - self.sla_threshold_ms:.1f}ms over)"
            )


class LatencyTracker:
    """Tracks LLM call latency and enforces SLA thresholds."""

    def __init__(self, sla_threshold_ms: float = 3000.0) -> None:
        self.sla_threshold_ms = sla_threshold_ms
        self._history: List[float] = []

    @contextmanager
    def measure(self) -> Generator[LatencyMeasurement, None, None]:
        """Context manager that measures execution time of the wrapped block.

        Usage::

            tracker = LatencyTracker(sla_threshold_ms=2000)
            with tracker.measure() as m:
                response = llm.generate(prompt)
            m.assert_sla()
        """
        result = LatencyMeasurement(
            duration_ms=0.0,
            sla_threshold_ms=self.sla_threshold_ms,
        )
        start = time.perf_counter()
        try:
            yield result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.duration_ms = round(elapsed_ms, 3)
            result.passed = elapsed_ms <= self.sla_threshold_ms
            self._history.append(elapsed_ms)

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0-100) of recorded latencies."""
        if not self._history:
            return 0.0
        sorted_vals = sorted(self._history)
        idx = max(0, int(len(sorted_vals) * p / 100) - 1)
        return round(sorted_vals[idx], 3)

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def mean(self) -> float:
        if not self._history:
            return 0.0
        return round(statistics.mean(self._history), 3)

    def assert_p95_sla(self) -> None:
        """Assert that the p95 latency is within the SLA threshold."""
        p95 = self.p95
        if p95 > self.sla_threshold_ms:
            raise SLAViolationError(
                f"p95 latency {p95:.1f}ms exceeded SLA of {self.sla_threshold_ms:.1f}ms"
            )

    def report(self) -> dict:
        """Return a summary dictionary of latency statistics."""
        return {
            "count": len(self._history),
            "mean_ms": self.mean,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "sla_threshold_ms": self.sla_threshold_ms,
            "sla_violations": sum(
                1 for d in self._history if d > self.sla_threshold_ms
            ),
        }
