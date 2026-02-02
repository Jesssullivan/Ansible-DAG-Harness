"""Cost alert thresholds and notifications.

Provides configurable cost alerts based on session, daily, and total spending.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from harness.costs.tracker import TokenUsageTracker


class CostAlertLevel(str, Enum):
    """Severity level for cost alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CostThreshold:
    """A cost threshold configuration.

    Attributes:
        name: Threshold identifier
        level: Alert severity level
        amount: Dollar amount that triggers alert
        period: Time period (session, daily, weekly, monthly, total)
        description: Human-readable description
    """

    name: str
    level: CostAlertLevel
    amount: Decimal
    period: str  # "session", "daily", "weekly", "monthly", "total"
    description: str = ""


@dataclass
class CostAlert:
    """A triggered cost alert.

    Attributes:
        threshold: The threshold that was exceeded
        current_amount: Current cost amount
        exceeded_by: Amount over threshold
        session_id: Session that triggered (if applicable)
        timestamp: When alert was triggered
        context: Additional context
    """

    threshold: CostThreshold
    current_amount: Decimal
    exceeded_by: Decimal
    session_id: str | None
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> str:
        """Generate alert message."""
        return (
            f"[{self.threshold.level.value.upper()}] Cost alert: "
            f"{self.threshold.name} exceeded. "
            f"Current: ${self.current_amount:.4f}, "
            f"Threshold: ${self.threshold.amount:.4f}, "
            f"Over by: ${self.exceeded_by:.4f}"
        )


# Default cost thresholds
DEFAULT_COST_THRESHOLDS: list[CostThreshold] = [
    # Session-level alerts
    CostThreshold(
        name="session_warning",
        level=CostAlertLevel.WARNING,
        amount=Decimal("1.00"),
        period="session",
        description="Single session cost exceeds $1",
    ),
    CostThreshold(
        name="session_critical",
        level=CostAlertLevel.CRITICAL,
        amount=Decimal("5.00"),
        period="session",
        description="Single session cost exceeds $5",
    ),
    # Daily alerts
    CostThreshold(
        name="daily_warning",
        level=CostAlertLevel.WARNING,
        amount=Decimal("10.00"),
        period="daily",
        description="Daily cost exceeds $10",
    ),
    CostThreshold(
        name="daily_critical",
        level=CostAlertLevel.CRITICAL,
        amount=Decimal("50.00"),
        period="daily",
        description="Daily cost exceeds $50",
    ),
    # Weekly alerts
    CostThreshold(
        name="weekly_warning",
        level=CostAlertLevel.WARNING,
        amount=Decimal("50.00"),
        period="weekly",
        description="Weekly cost exceeds $50",
    ),
    CostThreshold(
        name="weekly_critical",
        level=CostAlertLevel.CRITICAL,
        amount=Decimal("200.00"),
        period="weekly",
        description="Weekly cost exceeds $200",
    ),
    # Monthly alerts
    CostThreshold(
        name="monthly_warning",
        level=CostAlertLevel.WARNING,
        amount=Decimal("200.00"),
        period="monthly",
        description="Monthly cost exceeds $200",
    ),
    CostThreshold(
        name="monthly_critical",
        level=CostAlertLevel.CRITICAL,
        amount=Decimal("1000.00"),
        period="monthly",
        description="Monthly cost exceeds $1000",
    ),
]


# Type for alert handlers
AlertHandler = Callable[[CostAlert], None]


class CostAlertManager:
    """Manages cost thresholds and triggers alerts.

    Monitors costs against configured thresholds and invokes
    registered handlers when thresholds are exceeded.
    """

    def __init__(
        self,
        tracker: TokenUsageTracker,
        thresholds: list[CostThreshold] | None = None,
    ):
        """Initialize alert manager.

        Args:
            tracker: TokenUsageTracker for cost queries
            thresholds: Custom thresholds (uses defaults if None, pass [] for no thresholds)
        """
        self.tracker = tracker
        self.thresholds = DEFAULT_COST_THRESHOLDS.copy() if thresholds is None else list(thresholds)
        self._handlers: list[AlertHandler] = []
        self._triggered_alerts: dict[str, CostAlert] = {}  # Dedup by threshold name
        self._alert_history: list[CostAlert] = []

    def add_handler(self, handler: AlertHandler) -> None:
        """Register an alert handler.

        Args:
            handler: Callable that receives CostAlert instances
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: AlertHandler) -> None:
        """Remove a registered handler.

        Args:
            handler: Handler to remove
        """
        if handler in self._handlers:
            self._handlers.remove(handler)

    def add_threshold(self, threshold: CostThreshold) -> None:
        """Add a custom threshold.

        Args:
            threshold: Threshold to add
        """
        self.thresholds.append(threshold)

    def remove_threshold(self, name: str) -> bool:
        """Remove a threshold by name.

        Args:
            name: Threshold name to remove

        Returns:
            True if threshold was removed
        """
        original_len = len(self.thresholds)
        self.thresholds = [t for t in self.thresholds if t.name != name]
        return len(self.thresholds) < original_len

    def check_session(self, session_id: str) -> list[CostAlert]:
        """Check session costs against thresholds.

        Args:
            session_id: Session to check

        Returns:
            List of triggered alerts
        """
        summary = self.tracker.get_session_cost(session_id)
        alerts = []

        for threshold in self.thresholds:
            if threshold.period != "session":
                continue

            if summary.total_cost > threshold.amount:
                alert = CostAlert(
                    threshold=threshold,
                    current_amount=summary.total_cost,
                    exceeded_by=summary.total_cost - threshold.amount,
                    session_id=session_id,
                    timestamp=datetime.utcnow(),
                    context={
                        "input_tokens": summary.total_input_tokens,
                        "output_tokens": summary.total_output_tokens,
                        "record_count": summary.record_count,
                    },
                )
                alerts.append(alert)
                self._trigger_alert(alert)

        return alerts

    def check_daily(self) -> list[CostAlert]:
        """Check daily costs against thresholds.

        Returns:
            List of triggered alerts
        """
        from datetime import timedelta

        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        summary = self.tracker.get_total_cost(
            start_date=today,
            end_date=today + timedelta(days=1),
        )
        alerts = []

        for threshold in self.thresholds:
            if threshold.period != "daily":
                continue

            if summary.total_cost > threshold.amount:
                alert = CostAlert(
                    threshold=threshold,
                    current_amount=summary.total_cost,
                    exceeded_by=summary.total_cost - threshold.amount,
                    session_id=None,
                    timestamp=datetime.utcnow(),
                    context={
                        "period_start": today.isoformat(),
                        "input_tokens": summary.total_input_tokens,
                        "output_tokens": summary.total_output_tokens,
                    },
                )
                alerts.append(alert)
                self._trigger_alert(alert)

        return alerts

    def check_weekly(self) -> list[CostAlert]:
        """Check weekly costs against thresholds.

        Returns:
            List of triggered alerts
        """
        from datetime import timedelta

        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today - timedelta(days=today.weekday())  # Monday
        summary = self.tracker.get_total_cost(
            start_date=week_start,
            end_date=today + timedelta(days=1),
        )
        alerts = []

        for threshold in self.thresholds:
            if threshold.period != "weekly":
                continue

            if summary.total_cost > threshold.amount:
                alert = CostAlert(
                    threshold=threshold,
                    current_amount=summary.total_cost,
                    exceeded_by=summary.total_cost - threshold.amount,
                    session_id=None,
                    timestamp=datetime.utcnow(),
                    context={
                        "period_start": week_start.isoformat(),
                        "input_tokens": summary.total_input_tokens,
                        "output_tokens": summary.total_output_tokens,
                    },
                )
                alerts.append(alert)
                self._trigger_alert(alert)

        return alerts

    def check_monthly(self) -> list[CostAlert]:
        """Check monthly costs against thresholds.

        Returns:
            List of triggered alerts
        """
        from datetime import timedelta

        today = datetime.utcnow()
        month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        summary = self.tracker.get_total_cost(
            start_date=month_start,
            end_date=today + timedelta(days=1),
        )
        alerts = []

        for threshold in self.thresholds:
            if threshold.period != "monthly":
                continue

            if summary.total_cost > threshold.amount:
                alert = CostAlert(
                    threshold=threshold,
                    current_amount=summary.total_cost,
                    exceeded_by=summary.total_cost - threshold.amount,
                    session_id=None,
                    timestamp=datetime.utcnow(),
                    context={
                        "period_start": month_start.isoformat(),
                        "input_tokens": summary.total_input_tokens,
                        "output_tokens": summary.total_output_tokens,
                    },
                )
                alerts.append(alert)
                self._trigger_alert(alert)

        return alerts

    def check_all(self, session_id: str | None = None) -> list[CostAlert]:
        """Check all thresholds.

        Args:
            session_id: Optional session to check session-level thresholds

        Returns:
            List of all triggered alerts
        """
        alerts = []

        if session_id:
            alerts.extend(self.check_session(session_id))

        alerts.extend(self.check_daily())
        alerts.extend(self.check_weekly())
        alerts.extend(self.check_monthly())

        return alerts

    def _trigger_alert(self, alert: CostAlert) -> None:
        """Trigger an alert and invoke handlers.

        Args:
            alert: Alert to trigger
        """
        # Store in history
        self._alert_history.append(alert)

        # Dedup: only invoke handlers once per threshold per hour
        key = f"{alert.threshold.name}:{alert.timestamp.hour}"
        if key in self._triggered_alerts:
            return

        self._triggered_alerts[key] = alert

        # Invoke all registered handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                # Don't let handler errors break alert processing
                pass

    def get_alert_history(
        self, limit: int = 100, level: CostAlertLevel | None = None
    ) -> list[CostAlert]:
        """Get recent alert history.

        Args:
            limit: Maximum alerts to return
            level: Filter by severity level

        Returns:
            List of recent alerts
        """
        alerts = self._alert_history
        if level:
            alerts = [a for a in alerts if a.threshold.level == level]
        return alerts[-limit:]

    def clear_history(self) -> None:
        """Clear alert history."""
        self._alert_history.clear()
        self._triggered_alerts.clear()

    def get_current_status(self) -> dict[str, Any]:
        """Get current cost status across all periods.

        Returns:
            Status dict with costs and threshold info
        """
        from datetime import timedelta

        today = datetime.utcnow()
        today_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        month_start = today_start.replace(day=1)

        daily = self.tracker.get_total_cost(
            start_date=today_start, end_date=today + timedelta(days=1)
        )
        weekly = self.tracker.get_total_cost(
            start_date=week_start, end_date=today + timedelta(days=1)
        )
        monthly = self.tracker.get_total_cost(
            start_date=month_start, end_date=today + timedelta(days=1)
        )

        def get_threshold_status(period: str, current: Decimal) -> dict[str, Any]:
            """Get threshold status for a period."""
            thresholds = [t for t in self.thresholds if t.period == period]
            warning = next((t for t in thresholds if t.level == CostAlertLevel.WARNING), None)
            critical = next((t for t in thresholds if t.level == CostAlertLevel.CRITICAL), None)

            status = "ok"
            if critical and current > critical.amount:
                status = "critical"
            elif warning and current > warning.amount:
                status = "warning"

            return {
                "current": float(current),
                "status": status,
                "warning_threshold": float(warning.amount) if warning else None,
                "critical_threshold": float(critical.amount) if critical else None,
            }

        return {
            "daily": get_threshold_status("daily", daily.total_cost),
            "weekly": get_threshold_status("weekly", weekly.total_cost),
            "monthly": get_threshold_status("monthly", monthly.total_cost),
            "active_alerts": len(self._triggered_alerts),
            "total_alerts_triggered": len(self._alert_history),
        }
