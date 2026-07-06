"""Unit tests for LatencyTracker — covers all branches including SLA violations."""

from __future__ import annotations

import pytest

from genai_quality_eval.metrics.latency.latency_tracker import (
    LatencyMeasurement,
    LatencyTracker,
    SLAViolationError,
)


def test_measurement_passes_when_within_sla():
    m = LatencyMeasurement(duration_ms=100.0, sla_threshold_ms=3000.0)
    assert m.passed is True


def test_measurement_fails_when_over_sla():
    m = LatencyMeasurement(duration_ms=5000.0, sla_threshold_ms=3000.0)
    assert m.passed is False


def test_assert_sla_raises_on_violation():
    m = LatencyMeasurement(duration_ms=5000.0, sla_threshold_ms=3000.0)
    with pytest.raises(SLAViolationError):
        m.assert_sla()


def test_assert_sla_passes_when_within_threshold():
    m = LatencyMeasurement(duration_ms=100.0, sla_threshold_ms=3000.0)
    m.assert_sla()  # must not raise


def test_measure_context_manager_records_duration():
    tracker = LatencyTracker(sla_threshold_ms=3000.0)
    with tracker.measure() as m:
        pass
    assert m.duration_ms >= 0.0
    assert len(tracker._history) == 1


def test_percentile_empty_history():
    tracker = LatencyTracker()
    assert tracker.percentile(95) == 0.0


def test_mean_empty_history():
    tracker = LatencyTracker()
    assert tracker.mean == 0.0


def test_p50_p95_p99_with_data():
    tracker = LatencyTracker(sla_threshold_ms=5000.0)
    tracker._history = [100.0, 200.0, 300.0, 400.0, 500.0]
    assert tracker.p50 > 0.0
    assert tracker.p95 >= tracker.p50
    assert tracker.p99 >= tracker.p95


def test_assert_p95_sla_passes():
    tracker = LatencyTracker(sla_threshold_ms=5000.0)
    tracker._history = [100.0, 200.0]
    tracker.assert_p95_sla()  # must not raise


def test_assert_p95_sla_raises():
    tracker = LatencyTracker(sla_threshold_ms=100.0)
    tracker._history = [5000.0, 6000.0]
    with pytest.raises(SLAViolationError):
        tracker.assert_p95_sla()


def test_report_structure():
    tracker = LatencyTracker(sla_threshold_ms=3000.0)
    tracker._history = [100.0, 200.0, 5000.0]
    report = tracker.report()
    assert report["count"] == 3
    assert report["sla_violations"] == 1
    assert "p95_ms" in report
    assert "mean_ms" in report


def test_sla_violation_error_is_assertion_error():
    assert issubclass(SLAViolationError, AssertionError)
