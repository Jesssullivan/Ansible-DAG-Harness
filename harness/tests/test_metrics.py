"""Tests for golden metrics tracking."""

import pytest
from harness.db.state import StateDB
from harness.metrics.golden import (
    GoldenMetric,
    GoldenMetricsTracker,
    MetricType,
    DEFAULT_GOLDEN_METRICS,
)


class TestGoldenMetric:
    """Test GoldenMetric dataclass."""

    @pytest.mark.unit
    def test_create_metric(self):
        """Create a metric with all fields."""
        metric = GoldenMetric(
            name="test_metric",
            metric_type=MetricType.LATENCY,
            baseline_value=100.0,
            warning_threshold=1.5,
            critical_threshold=2.0,
            description="Test metric",
            unit="ms"
        )

        assert metric.name == "test_metric"
        assert metric.metric_type == MetricType.LATENCY
        assert metric.baseline_value == 100.0
        assert metric.warning_threshold == 1.5
        assert metric.critical_threshold == 2.0
        assert metric.description == "Test metric"
        assert metric.unit == "ms"

    @pytest.mark.unit
    def test_default_metrics_exist(self):
        """Default metrics should be defined."""
        assert len(DEFAULT_GOLDEN_METRICS) > 0

        # Check some expected metrics
        names = {m.name for m in DEFAULT_GOLDEN_METRICS}
        assert "workflow_completion_time" in names
        assert "molecule_test_time" in names
        assert "active_regressions" in names


class TestGoldenMetricsTracker:
    """Test GoldenMetricsTracker."""

    @pytest.mark.unit
    def test_tracker_initialization(self, db: StateDB):
        """Tracker should initialize with default metrics."""
        tracker = GoldenMetricsTracker(db)

        assert len(tracker.metrics) > 0
        assert "workflow_completion_time" in tracker.metrics

    @pytest.mark.unit
    def test_record_latency_ok(self, db: StateDB):
        """Record latency metric within baseline."""
        tracker = GoldenMetricsTracker(db)

        # workflow_completion_time baseline is 120s
        status = tracker.record("workflow_completion_time", 100.0)
        assert status == "ok"

    @pytest.mark.unit
    def test_record_latency_warning(self, db: StateDB):
        """Record latency metric at warning level."""
        tracker = GoldenMetricsTracker(db)

        # warning_threshold is 1.5, so 180s is warning
        status = tracker.record("workflow_completion_time", 200.0)
        assert status == "warning"

    @pytest.mark.unit
    def test_record_latency_critical(self, db: StateDB):
        """Record latency metric at critical level."""
        tracker = GoldenMetricsTracker(db)

        # critical_threshold is 2.0, so 240s is critical
        status = tracker.record("workflow_completion_time", 300.0)
        assert status == "critical"

    @pytest.mark.unit
    def test_record_gauge_ok(self, db: StateDB):
        """Record gauge metric below warning."""
        tracker = GoldenMetricsTracker(db)

        # active_regressions warning is 3
        status = tracker.record("active_regressions", 1.0)
        assert status == "ok"

    @pytest.mark.unit
    def test_record_gauge_warning(self, db: StateDB):
        """Record gauge metric at warning level."""
        tracker = GoldenMetricsTracker(db)

        # active_regressions warning is 3
        status = tracker.record("active_regressions", 5.0)
        assert status == "warning"

    @pytest.mark.unit
    def test_record_gauge_critical(self, db: StateDB):
        """Record gauge metric at critical level."""
        tracker = GoldenMetricsTracker(db)

        # active_regressions critical is 10
        status = tracker.record("active_regressions", 15.0)
        assert status == "critical"

    @pytest.mark.unit
    def test_record_unknown_metric(self, db: StateDB):
        """Recording unknown metric should raise error."""
        tracker = GoldenMetricsTracker(db)

        with pytest.raises(ValueError) as exc_info:
            tracker.record("unknown_metric", 100.0)

        assert "Unknown metric" in str(exc_info.value)

    @pytest.mark.unit
    def test_record_with_context(self, db: StateDB):
        """Record metric with context metadata."""
        tracker = GoldenMetricsTracker(db)

        status = tracker.record(
            "workflow_completion_time",
            100.0,
            context={"role": "common", "wave": 0}
        )
        assert status == "ok"

    @pytest.mark.unit
    def test_get_recent(self, db: StateDB):
        """Get recent metric values."""
        tracker = GoldenMetricsTracker(db)

        # Record a few values
        tracker.record("workflow_completion_time", 100.0)
        tracker.record("workflow_completion_time", 110.0)
        tracker.record("workflow_completion_time", 120.0)

        recent = tracker.get_recent("workflow_completion_time", hours=1)
        assert len(recent) == 3

    @pytest.mark.unit
    def test_get_latest(self, db: StateDB):
        """Get most recent metric value."""
        tracker = GoldenMetricsTracker(db)

        tracker.record("workflow_completion_time", 100.0)
        tracker.record("workflow_completion_time", 150.0)

        latest = tracker.get_latest("workflow_completion_time")
        assert latest is not None
        assert latest["value"] == 150.0

    @pytest.mark.unit
    def test_get_latest_no_data(self, db: StateDB):
        """Get latest when no data exists."""
        tracker = GoldenMetricsTracker(db)

        latest = tracker.get_latest("workflow_completion_time")
        assert latest is None

    @pytest.mark.unit
    def test_get_status_summary(self, db: StateDB):
        """Get status summary of all metrics."""
        tracker = GoldenMetricsTracker(db)

        # Record some values
        tracker.record("workflow_completion_time", 100.0)
        tracker.record("active_regressions", 5.0)

        summary = tracker.get_status_summary()
        assert isinstance(summary, dict)
        assert summary["workflow_completion_time"] == "ok"
        assert summary["active_regressions"] == "warning"
        # Unrecorded metrics should be unknown
        assert summary["molecule_test_time"] == "unknown"

    @pytest.mark.unit
    def test_get_health(self, db: StateDB):
        """Get overall health status."""
        tracker = GoldenMetricsTracker(db)

        # No data - should be unknown
        health = tracker.get_health()
        assert health["overall"] == "unknown"

        # Record OK metric
        tracker.record("workflow_completion_time", 100.0)
        health = tracker.get_health()
        assert health["ok"] >= 1

        # Record critical metric
        tracker.record("active_regressions", 15.0)
        health = tracker.get_health()
        assert health["overall"] == "critical"
        assert health["critical"] >= 1

    @pytest.mark.unit
    def test_update_baseline(self, db: StateDB):
        """Update baseline for a metric."""
        tracker = GoldenMetricsTracker(db)

        original = tracker.metrics["workflow_completion_time"].baseline_value
        tracker.update_baseline("workflow_completion_time", 200.0)

        assert tracker.metrics["workflow_completion_time"].baseline_value == 200.0

        # Now 100.0 is well under baseline
        status = tracker.record("workflow_completion_time", 100.0)
        assert status == "ok"

    @pytest.mark.unit
    def test_update_baseline_unknown(self, db: StateDB):
        """Update baseline for unknown metric should raise."""
        tracker = GoldenMetricsTracker(db)

        with pytest.raises(ValueError):
            tracker.update_baseline("unknown_metric", 100.0)

    @pytest.mark.unit
    def test_register_metric(self, db: StateDB):
        """Register a custom metric."""
        tracker = GoldenMetricsTracker(db)

        custom = GoldenMetric(
            name="custom_metric",
            metric_type=MetricType.COUNT,
            baseline_value=10.0,
            warning_threshold=2.0,
            critical_threshold=5.0
        )
        tracker.register_metric(custom)

        assert "custom_metric" in tracker.metrics
        status = tracker.record("custom_metric", 15.0)
        assert status == "ok"  # 15 / 10 = 1.5 < 2.0 warning

    @pytest.mark.unit
    def test_list_metrics(self, db: StateDB):
        """List all registered metrics."""
        tracker = GoldenMetricsTracker(db)
        metrics = tracker.list_metrics()

        assert len(metrics) == len(DEFAULT_GOLDEN_METRICS)

    @pytest.mark.unit
    def test_get_trend(self, db: StateDB):
        """Get trend analysis."""
        tracker = GoldenMetricsTracker(db)

        # Record multiple values
        for i in range(10):
            tracker.record("workflow_completion_time", 100.0 + i * 5)

        trend = tracker.get_trend("workflow_completion_time", hours=1)
        assert trend["count"] == 10
        assert trend["min"] == 100.0
        assert trend["max"] == 145.0

    @pytest.mark.unit
    def test_get_trend_no_data(self, db: StateDB):
        """Trend with no data."""
        tracker = GoldenMetricsTracker(db)

        trend = tracker.get_trend("workflow_completion_time", hours=1)
        assert trend["count"] == 0
        assert trend["average"] is None

    @pytest.mark.unit
    def test_purge_old(self, db: StateDB):
        """Purge old records."""
        tracker = GoldenMetricsTracker(db)

        # Record some values
        tracker.record("workflow_completion_time", 100.0)

        # Purge (with days=0 to keep all recent)
        # This mostly tests that the method works
        count = tracker.purge_old(days=30)
        assert count >= 0  # Nothing old to purge
