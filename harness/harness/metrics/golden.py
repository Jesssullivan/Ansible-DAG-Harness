"""Golden metrics tracking for harness performance baselines."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from harness.db.state import StateDB


class MetricType(str, Enum):
    """Type of metric being tracked."""

    LATENCY = "latency"
    COUNT = "count"
    RATE = "rate"
    GAUGE = "gauge"


@dataclass
class GoldenMetric:
    """A golden metric with baseline and thresholds."""

    name: str
    metric_type: MetricType
    baseline_value: float
    warning_threshold: float  # Multiplier for baseline (e.g., 1.5 = 50% worse)
    critical_threshold: float  # Multiplier for baseline (e.g., 2.0 = 100% worse)
    description: str | None = None
    unit: str | None = None


# Default golden metrics for the harness
DEFAULT_GOLDEN_METRICS = [
    GoldenMetric(
        name="workflow_completion_time",
        metric_type=MetricType.LATENCY,
        baseline_value=120.0,  # seconds
        warning_threshold=1.5,  # 50% slower
        critical_threshold=2.0,  # 100% slower
        description="Time to complete box-up-role workflow",
        unit="seconds",
    ),
    GoldenMetric(
        name="molecule_test_time",
        metric_type=MetricType.LATENCY,
        baseline_value=180.0,
        warning_threshold=1.5,
        critical_threshold=2.0,
        description="Time to run molecule tests for a role",
        unit="seconds",
    ),
    GoldenMetric(
        name="pytest_test_time",
        metric_type=MetricType.LATENCY,
        baseline_value=60.0,
        warning_threshold=1.5,
        critical_threshold=2.0,
        description="Time to run pytest tests",
        unit="seconds",
    ),
    GoldenMetric(
        name="db_query_time_p99",
        metric_type=MetricType.LATENCY,
        baseline_value=0.1,  # 100ms
        warning_threshold=2.0,
        critical_threshold=5.0,
        description="99th percentile database query time",
        unit="seconds",
    ),
    GoldenMetric(
        name="active_regressions",
        metric_type=MetricType.GAUGE,
        baseline_value=0.0,
        warning_threshold=3.0,  # Warning if 3+ regressions
        critical_threshold=10.0,  # Critical if 10+ regressions
        description="Number of active test regressions",
    ),
    GoldenMetric(
        name="cycle_detection_time",
        metric_type=MetricType.LATENCY,
        baseline_value=0.05,  # 50ms
        warning_threshold=2.0,
        critical_threshold=5.0,
        description="Time to detect cycles in dependency graph",
        unit="seconds",
    ),
    GoldenMetric(
        name="worktree_creation_time",
        metric_type=MetricType.LATENCY,
        baseline_value=30.0,  # seconds
        warning_threshold=1.5,
        critical_threshold=2.0,
        description="Time to create a git worktree",
        unit="seconds",
    ),
    GoldenMetric(
        name="gitlab_api_latency",
        metric_type=MetricType.LATENCY,
        baseline_value=2.0,  # seconds
        warning_threshold=2.0,
        critical_threshold=5.0,
        description="GitLab API response time",
        unit="seconds",
    ),
    GoldenMetric(
        name="sync_roles_time",
        metric_type=MetricType.LATENCY,
        baseline_value=10.0,  # seconds
        warning_threshold=1.5,
        critical_threshold=2.0,
        description="Time to sync roles from filesystem",
        unit="seconds",
    ),
    GoldenMetric(
        name="pending_workflows",
        metric_type=MetricType.GAUGE,
        baseline_value=0.0,
        warning_threshold=5.0,  # Warning if 5+ pending
        critical_threshold=20.0,  # Critical if 20+ pending
        description="Number of pending workflow executions",
    ),
]


class GoldenMetricsTracker:
    """Track and evaluate golden metrics."""

    def __init__(self, db: StateDB):
        """
        Initialize the metrics tracker.

        Args:
            db: StateDB instance for storing metrics
        """
        self.db = db
        self.metrics = {m.name: m for m in DEFAULT_GOLDEN_METRICS}
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure golden_metrics table exists."""
        with self.db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS golden_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    baseline REAL,
                    status TEXT CHECK (status IN ('ok', 'warning', 'critical')),
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context TEXT  -- JSON metadata
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_golden_metrics_name
                ON golden_metrics(name, recorded_at)
            """)

    def record(self, name: str, value: float, context: dict | None = None) -> str:
        """
        Record a metric value and evaluate against baseline.

        Args:
            name: Name of the metric (must be registered)
            value: Current value to record
            context: Optional context metadata (JSON-serializable)

        Returns:
            Status: 'ok', 'warning', or 'critical'

        Raises:
            ValueError: If metric name is not registered
        """
        metric = self.metrics.get(name)
        if not metric:
            raise ValueError(
                f"Unknown metric: {name}. Available: {', '.join(sorted(self.metrics.keys()))}"
            )

        # Evaluate status based on metric type
        if metric.metric_type == MetricType.GAUGE:
            # For gauges, thresholds are absolute values
            if value >= metric.critical_threshold:
                status = "critical"
            elif value >= metric.warning_threshold:
                status = "warning"
            else:
                status = "ok"
        else:
            # For latency/rate/count, thresholds are multipliers
            ratio = value / metric.baseline_value if metric.baseline_value else 1.0

            if ratio >= metric.critical_threshold:
                status = "critical"
            elif ratio >= metric.warning_threshold:
                status = "warning"
            else:
                status = "ok"

        # Store in database
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO golden_metrics (name, value, baseline, status, context)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    name,
                    value,
                    metric.baseline_value,
                    status,
                    json.dumps(context) if context else None,
                ),
            )

        return status

    def get_recent(self, name: str, hours: int = 24) -> list[dict[str, Any]]:
        """
        Get recent metric values.

        Args:
            name: Metric name
            hours: Number of hours to look back (default 24)

        Returns:
            List of metric records with timestamp, value, status, etc.
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        # SQLite CURRENT_TIMESTAMP uses space separator, not 'T'
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM golden_metrics
                WHERE name = ? AND recorded_at > ?
                ORDER BY recorded_at DESC
            """,
                (name, cutoff_str),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_latest(self, name: str) -> dict[str, Any] | None:
        """
        Get the most recent value for a metric.

        Args:
            name: Metric name

        Returns:
            Most recent metric record or None
        """
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM golden_metrics
                WHERE name = ?
                ORDER BY recorded_at DESC
                LIMIT 1
            """,
                (name,),
            ).fetchone()

        return dict(row) if row else None

    def get_status_summary(self) -> dict[str, str]:
        """
        Get current status of all metrics.

        Returns:
            Dict mapping metric names to their latest status
        """
        summary = {}

        with self.db.connection() as conn:
            for name in self.metrics:
                row = conn.execute(
                    """
                    SELECT status FROM golden_metrics
                    WHERE name = ?
                    ORDER BY recorded_at DESC
                    LIMIT 1
                """,
                    (name,),
                ).fetchone()

                summary[name] = row["status"] if row else "unknown"

        return summary

    def get_health(self) -> dict[str, Any]:
        """
        Get overall health based on golden metrics.

        Returns:
            Dict with overall health status and breakdown
        """
        summary = self.get_status_summary()

        critical_count = sum(1 for s in summary.values() if s == "critical")
        warning_count = sum(1 for s in summary.values() if s == "warning")
        ok_count = sum(1 for s in summary.values() if s == "ok")
        unknown_count = sum(1 for s in summary.values() if s == "unknown")

        if critical_count > 0:
            overall = "critical"
        elif warning_count > 0:
            overall = "warning"
        elif ok_count > 0:
            overall = "healthy"
        else:
            overall = "unknown"

        return {
            "overall": overall,
            "critical": critical_count,
            "warning": warning_count,
            "ok": ok_count,
            "unknown": unknown_count,
            "metrics": summary,
        }

    def update_baseline(self, name: str, new_baseline: float) -> None:
        """
        Update the baseline value for a metric.

        This is useful when performance improves and you want to
        reset the baseline to the new normal.

        Args:
            name: Metric name
            new_baseline: New baseline value

        Raises:
            ValueError: If metric name is not registered
        """
        if name not in self.metrics:
            raise ValueError(f"Unknown metric: {name}")

        self.metrics[name].baseline_value = new_baseline
        self.db.log_audit(
            "golden_metric",
            0,
            "update_baseline",
            new_value={"name": name, "baseline": new_baseline},
        )

    def register_metric(self, metric: GoldenMetric) -> None:
        """
        Register a new metric.

        Args:
            metric: GoldenMetric instance to register
        """
        self.metrics[metric.name] = metric

    def list_metrics(self) -> list[GoldenMetric]:
        """
        List all registered metrics.

        Returns:
            List of GoldenMetric instances
        """
        return list(self.metrics.values())

    def get_trend(self, name: str, hours: int = 24) -> dict[str, Any]:
        """
        Get trend information for a metric.

        Args:
            name: Metric name
            hours: Number of hours to analyze

        Returns:
            Dict with trend information (average, min, max, count)
        """
        records = self.get_recent(name, hours=hours)

        if not records:
            return {
                "metric": name,
                "count": 0,
                "average": None,
                "min": None,
                "max": None,
                "latest": None,
                "trend": "unknown",
            }

        values = [r["value"] for r in records]
        latest = values[0]
        average = sum(values) / len(values)

        # Determine trend by comparing first and second half
        mid = len(values) // 2
        if mid > 0:
            first_half_avg = sum(values[mid:]) / len(values[mid:])
            second_half_avg = sum(values[:mid]) / mid

            if second_half_avg > first_half_avg * 1.1:
                trend = "increasing"
            elif second_half_avg < first_half_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        return {
            "metric": name,
            "count": len(values),
            "average": average,
            "min": min(values),
            "max": max(values),
            "latest": latest,
            "trend": trend,
        }

    def purge_old(self, days: int = 30) -> int:
        """
        Purge old metric records.

        Args:
            days: Records older than this many days will be deleted

        Returns:
            Number of records deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        # SQLite CURRENT_TIMESTAMP uses space separator, not 'T'
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

        with self.db.connection() as conn:
            count = conn.execute(
                """
                SELECT COUNT(*) as c FROM golden_metrics
                WHERE recorded_at < ?
            """,
                (cutoff_str,),
            ).fetchone()["c"]

            conn.execute(
                """
                DELETE FROM golden_metrics
                WHERE recorded_at < ?
            """,
                (cutoff_str,),
            )

        return count
