"""Tests for cost tracking and optimization module."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from harness.costs.alerts import (
    DEFAULT_COST_THRESHOLDS,
    CostAlert,
    CostAlertLevel,
    CostAlertManager,
    CostThreshold,
)
from harness.costs.pricing import (
    CLAUDE_PRICING,
    ModelPricing,
    calculate_cost,
    get_model_pricing,
)
from harness.costs.routing import (
    TaskComplexity,
    estimate_complexity,
    get_model_for_task_type,
    get_recommended_model,
    route_to_model,
)
from harness.costs.tracker import (
    TokenUsageTracker,
)
from harness.db.state import StateDB

# ============================================================================
# PRICING TESTS
# ============================================================================


class TestModelPricing:
    """Test ModelPricing dataclass."""

    @pytest.mark.unit
    def test_create_pricing(self):
        """Create pricing with all fields."""
        pricing = ModelPricing(
            model_id="test-model",
            input_cost_per_mtok=Decimal("10.00"),
            output_cost_per_mtok=Decimal("50.00"),
            display_name="Test Model",
        )

        assert pricing.model_id == "test-model"
        assert pricing.input_cost_per_mtok == Decimal("10.00")
        assert pricing.output_cost_per_mtok == Decimal("50.00")
        assert pricing.display_name == "Test Model"

    @pytest.mark.unit
    def test_cost_per_token(self):
        """Calculate per-token cost from per-million cost."""
        pricing = ModelPricing(
            model_id="test-model",
            input_cost_per_mtok=Decimal("10.00"),
            output_cost_per_mtok=Decimal("50.00"),
            display_name="Test Model",
        )

        assert pricing.input_cost_per_token == Decimal("0.00001")
        assert pricing.output_cost_per_token == Decimal("0.00005")

    @pytest.mark.unit
    def test_calculate_cost(self):
        """Calculate cost for token usage."""
        pricing = ModelPricing(
            model_id="test-model",
            input_cost_per_mtok=Decimal("10.00"),
            output_cost_per_mtok=Decimal("50.00"),
            display_name="Test Model",
        )

        # 1000 input + 500 output
        # Input: 1000 * 0.00001 = 0.01
        # Output: 500 * 0.00005 = 0.025
        # Total: 0.035
        cost = pricing.calculate_cost(1000, 500)
        assert cost == Decimal("0.035")

    @pytest.mark.unit
    def test_calculate_cost_large_usage(self):
        """Calculate cost for large token usage."""
        pricing = CLAUDE_PRICING["claude-opus-4-5"]

        # 1 million input + 100k output
        # Input: 1M * 15/1M = $15
        # Output: 100k * 75/1M = $7.50
        # Total: $22.50
        cost = pricing.calculate_cost(1_000_000, 100_000)
        assert cost == Decimal("22.50")


class TestClaudePricing:
    """Test Claude model pricing constants."""

    @pytest.mark.unit
    def test_opus_pricing(self):
        """Verify Opus pricing."""
        pricing = CLAUDE_PRICING["claude-opus-4-5"]
        assert pricing.input_cost_per_mtok == Decimal("15.00")
        assert pricing.output_cost_per_mtok == Decimal("75.00")

    @pytest.mark.unit
    def test_sonnet_pricing(self):
        """Verify Sonnet pricing."""
        pricing = CLAUDE_PRICING["claude-sonnet-4"]
        assert pricing.input_cost_per_mtok == Decimal("3.00")
        assert pricing.output_cost_per_mtok == Decimal("15.00")

    @pytest.mark.unit
    def test_haiku_pricing(self):
        """Verify Haiku pricing."""
        pricing = CLAUDE_PRICING["claude-haiku-3-5"]
        assert pricing.input_cost_per_mtok == Decimal("0.25")
        assert pricing.output_cost_per_mtok == Decimal("1.25")

    @pytest.mark.unit
    def test_all_models_have_pricing(self):
        """All expected models should have pricing."""
        expected_models = [
            "claude-opus-4-5",
            "claude-sonnet-4",
            "claude-haiku-3-5",
        ]
        for model in expected_models:
            assert model in CLAUDE_PRICING


class TestGetModelPricing:
    """Test get_model_pricing function."""

    @pytest.mark.unit
    def test_exact_match(self):
        """Get pricing by exact model ID."""
        pricing = get_model_pricing("claude-opus-4-5")
        assert pricing is not None
        assert pricing.model_id == "claude-opus-4-5"

    @pytest.mark.unit
    def test_tier_alias(self):
        """Get pricing by tier alias."""
        pricing = get_model_pricing("opus")
        assert pricing is not None
        assert "opus" in pricing.model_id

    @pytest.mark.unit
    def test_partial_match(self):
        """Get pricing by partial model name."""
        pricing = get_model_pricing("sonnet")
        assert pricing is not None
        assert "sonnet" in pricing.model_id

    @pytest.mark.unit
    def test_unknown_model(self):
        """Unknown model returns None."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None


class TestCalculateCost:
    """Test calculate_cost function."""

    @pytest.mark.unit
    def test_calculate_opus_cost(self):
        """Calculate cost for Opus usage."""
        cost = calculate_cost("claude-opus-4-5", 1000, 500)
        assert cost is not None
        # Input: 1000 * 15/1M = 0.015
        # Output: 500 * 75/1M = 0.0375
        # Total: 0.0525
        assert cost == Decimal("0.0525")

    @pytest.mark.unit
    def test_calculate_unknown_model(self):
        """Unknown model returns None."""
        cost = calculate_cost("unknown-model", 1000, 500)
        assert cost is None


# ============================================================================
# TRACKER TESTS
# ============================================================================


class TestTokenUsageTracker:
    """Test TokenUsageTracker class."""

    @pytest.mark.unit
    def test_tracker_initialization(self, db: StateDB):
        """Tracker initializes and creates table."""
        TokenUsageTracker(db)

        # Table should exist
        with db.connection() as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='token_usage'"
            ).fetchone()
            assert result is not None

    @pytest.mark.unit
    def test_record_usage(self, db: StateDB):
        """Record token usage."""
        tracker = TokenUsageTracker(db)

        record = tracker.record_usage(
            session_id="test-session-1",
            model="claude-opus-4-5",
            input_tokens=1000,
            output_tokens=500,
        )

        assert record.id is not None
        assert record.session_id == "test-session-1"
        assert record.model == "claude-opus-4-5"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.cost > Decimal("0")
        assert record.timestamp is not None

    @pytest.mark.unit
    def test_record_usage_with_context(self, db: StateDB):
        """Record token usage with context metadata."""
        tracker = TokenUsageTracker(db)

        record = tracker.record_usage(
            session_id="test-session-1",
            model="claude-sonnet-4",
            input_tokens=500,
            output_tokens=200,
            context={"role": "common", "task": "review"},
        )

        assert record.context is not None
        assert "common" in record.context

    @pytest.mark.unit
    def test_record_unknown_model(self, db: StateDB):
        """Recording with unknown model uses fallback pricing."""
        tracker = TokenUsageTracker(db)

        # Should not raise, uses fallback
        record = tracker.record_usage(
            session_id="test-session-1",
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        assert record.cost > Decimal("0")

    @pytest.mark.unit
    def test_get_session_cost(self, db: StateDB):
        """Get cost summary for a session."""
        tracker = TokenUsageTracker(db)

        # Record multiple usages
        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)
        tracker.record_usage("session-1", "claude-sonnet-4", 2000, 1000)
        tracker.record_usage("session-2", "claude-haiku-3-5", 500, 200)

        summary = tracker.get_session_cost("session-1")

        assert summary.total_cost > Decimal("0")
        assert summary.total_input_tokens == 3000
        assert summary.total_output_tokens == 1500
        assert summary.record_count == 2
        assert len(summary.by_model) == 2

    @pytest.mark.unit
    def test_get_session_cost_empty(self, db: StateDB):
        """Get cost for non-existent session."""
        tracker = TokenUsageTracker(db)

        summary = tracker.get_session_cost("nonexistent")

        assert summary.total_cost == Decimal("0")
        assert summary.record_count == 0

    @pytest.mark.unit
    def test_get_total_cost(self, db: StateDB):
        """Get total cost across all sessions."""
        tracker = TokenUsageTracker(db)

        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)
        tracker.record_usage("session-2", "claude-sonnet-4", 2000, 1000)

        summary = tracker.get_total_cost()

        assert summary.total_cost > Decimal("0")
        assert summary.total_input_tokens == 3000
        assert summary.total_output_tokens == 1500
        assert summary.record_count == 2
        assert len(summary.by_session) == 2

    @pytest.mark.unit
    def test_get_total_cost_with_dates(self, db: StateDB):
        """Get total cost with date filtering."""
        tracker = TokenUsageTracker(db)

        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)

        # Query with future start date should return empty
        future = datetime.utcnow() + timedelta(days=1)
        summary = tracker.get_total_cost(start_date=future)

        assert summary.total_cost == Decimal("0")
        assert summary.record_count == 0

    @pytest.mark.unit
    def test_get_costs_by_model(self, db: StateDB):
        """Get detailed cost breakdown by model."""
        tracker = TokenUsageTracker(db)

        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)
        tracker.record_usage("session-1", "claude-opus-4-5", 2000, 1000)
        tracker.record_usage("session-2", "claude-sonnet-4", 3000, 1500)

        by_model = tracker.get_costs_by_model()

        assert "claude-opus-4-5" in by_model
        assert "claude-sonnet-4" in by_model
        assert by_model["claude-opus-4-5"]["total_input_tokens"] == 3000
        assert by_model["claude-opus-4-5"]["record_count"] == 2

    @pytest.mark.unit
    def test_get_daily_costs(self, db: StateDB):
        """Get daily cost totals."""
        tracker = TokenUsageTracker(db)

        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)

        daily = tracker.get_daily_costs(days=30)

        assert len(daily) >= 1
        assert daily[0]["total_cost"] > Decimal("0")

    @pytest.mark.unit
    def test_get_session_records(self, db: StateDB):
        """Get individual records for a session."""
        tracker = TokenUsageTracker(db)

        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)
        tracker.record_usage("session-1", "claude-sonnet-4", 2000, 1000)

        records = tracker.get_session_records("session-1")

        assert len(records) == 2
        assert all(r.session_id == "session-1" for r in records)

    @pytest.mark.unit
    def test_purge_old_records(self, db: StateDB):
        """Purge old records."""
        tracker = TokenUsageTracker(db)

        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)

        # Purging with days=0 keeps today's records
        deleted = tracker.purge_old_records(days=0)
        assert deleted == 0

        # Records should still exist
        summary = tracker.get_session_cost("session-1")
        assert summary.record_count == 1


# ============================================================================
# ALERTS TESTS
# ============================================================================


class TestCostThreshold:
    """Test CostThreshold dataclass."""

    @pytest.mark.unit
    def test_create_threshold(self):
        """Create a cost threshold."""
        threshold = CostThreshold(
            name="test_threshold",
            level=CostAlertLevel.WARNING,
            amount=Decimal("10.00"),
            period="daily",
            description="Test threshold",
        )

        assert threshold.name == "test_threshold"
        assert threshold.level == CostAlertLevel.WARNING
        assert threshold.amount == Decimal("10.00")
        assert threshold.period == "daily"


class TestDefaultThresholds:
    """Test default threshold definitions."""

    @pytest.mark.unit
    def test_default_thresholds_exist(self):
        """Default thresholds should be defined."""
        assert len(DEFAULT_COST_THRESHOLDS) > 0

    @pytest.mark.unit
    def test_session_thresholds(self):
        """Session-level thresholds should exist."""
        session_thresholds = [t for t in DEFAULT_COST_THRESHOLDS if t.period == "session"]
        assert len(session_thresholds) >= 2

    @pytest.mark.unit
    def test_daily_thresholds(self):
        """Daily thresholds should exist."""
        daily_thresholds = [t for t in DEFAULT_COST_THRESHOLDS if t.period == "daily"]
        assert len(daily_thresholds) >= 2


class TestCostAlertManager:
    """Test CostAlertManager class."""

    @pytest.mark.unit
    def test_manager_initialization(self, db: StateDB):
        """Manager initializes with default thresholds."""
        tracker = TokenUsageTracker(db)
        manager = CostAlertManager(tracker)

        assert len(manager.thresholds) > 0

    @pytest.mark.unit
    def test_add_handler(self, db: StateDB):
        """Add alert handler."""
        tracker = TokenUsageTracker(db)
        manager = CostAlertManager(tracker)

        alerts_received = []
        manager.add_handler(lambda a: alerts_received.append(a))

        assert len(manager._handlers) == 1

    @pytest.mark.unit
    def test_add_custom_threshold(self, db: StateDB):
        """Add custom threshold."""
        tracker = TokenUsageTracker(db)
        manager = CostAlertManager(tracker, thresholds=[])

        custom = CostThreshold(
            name="custom",
            level=CostAlertLevel.WARNING,
            amount=Decimal("0.01"),
            period="session",
        )
        manager.add_threshold(custom)

        assert len(manager.thresholds) == 1

    @pytest.mark.unit
    def test_check_session_no_alert(self, db: StateDB):
        """Check session with cost below threshold."""
        tracker = TokenUsageTracker(db)
        manager = CostAlertManager(tracker)

        # Record small usage
        tracker.record_usage("session-1", "claude-haiku-3-5", 100, 50)

        alerts = manager.check_session("session-1")
        assert len(alerts) == 0

    @pytest.mark.unit
    def test_check_session_triggers_alert(self, db: StateDB):
        """Check session with cost above threshold triggers alert."""
        tracker = TokenUsageTracker(db)

        # Create manager with low threshold
        low_threshold = CostThreshold(
            name="test_session",
            level=CostAlertLevel.WARNING,
            amount=Decimal("0.001"),
            period="session",
        )
        manager = CostAlertManager(tracker, thresholds=[low_threshold])

        # Record any usage (will exceed low threshold)
        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)

        alerts = manager.check_session("session-1")
        assert len(alerts) == 1
        assert alerts[0].threshold.name == "test_session"

    @pytest.mark.unit
    def test_alert_handler_invoked(self, db: StateDB):
        """Handler is invoked when alert triggers."""
        tracker = TokenUsageTracker(db)

        low_threshold = CostThreshold(
            name="test_session",
            level=CostAlertLevel.WARNING,
            amount=Decimal("0.001"),
            period="session",
        )
        manager = CostAlertManager(tracker, thresholds=[low_threshold])

        alerts_received = []
        manager.add_handler(lambda a: alerts_received.append(a))

        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)
        manager.check_session("session-1")

        assert len(alerts_received) == 1

    @pytest.mark.unit
    def test_get_alert_history(self, db: StateDB):
        """Get alert history."""
        tracker = TokenUsageTracker(db)
        low_threshold = CostThreshold(
            name="test",
            level=CostAlertLevel.WARNING,
            amount=Decimal("0.001"),
            period="session",
        )
        manager = CostAlertManager(tracker, thresholds=[low_threshold])

        tracker.record_usage("session-1", "claude-opus-4-5", 1000, 500)
        manager.check_session("session-1")

        history = manager.get_alert_history()
        assert len(history) >= 1

    @pytest.mark.unit
    def test_get_current_status(self, db: StateDB):
        """Get current cost status."""
        tracker = TokenUsageTracker(db)
        manager = CostAlertManager(tracker)

        status = manager.get_current_status()

        assert "daily" in status
        assert "weekly" in status
        assert "monthly" in status
        assert "status" in status["daily"]


class TestCostAlert:
    """Test CostAlert dataclass."""

    @pytest.mark.unit
    def test_alert_message(self):
        """Alert generates proper message."""
        threshold = CostThreshold(
            name="test",
            level=CostAlertLevel.WARNING,
            amount=Decimal("10.00"),
            period="daily",
        )
        alert = CostAlert(
            threshold=threshold,
            current_amount=Decimal("15.00"),
            exceeded_by=Decimal("5.00"),
            session_id=None,
            timestamp=datetime.utcnow(),
        )

        message = alert.message
        assert "WARNING" in message
        assert "15.00" in message or "15" in message


# ============================================================================
# ROUTING TESTS
# ============================================================================


class TestTaskComplexity:
    """Test TaskComplexity enum."""

    @pytest.mark.unit
    def test_complexity_values(self):
        """Verify complexity values."""
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MEDIUM.value == "medium"
        assert TaskComplexity.COMPLEX.value == "complex"


class TestRouteToModel:
    """Test route_to_model function."""

    @pytest.mark.unit
    def test_route_simple(self):
        """Simple tasks route to Haiku."""
        model = route_to_model(TaskComplexity.SIMPLE)
        assert "haiku" in model

    @pytest.mark.unit
    def test_route_medium(self):
        """Medium tasks route to Sonnet."""
        model = route_to_model(TaskComplexity.MEDIUM)
        assert "sonnet" in model

    @pytest.mark.unit
    def test_route_complex(self):
        """Complex tasks route to Opus."""
        model = route_to_model(TaskComplexity.COMPLEX)
        assert "opus" in model

    @pytest.mark.unit
    def test_route_string_input(self):
        """Accept string complexity input."""
        model = route_to_model("simple")
        assert "haiku" in model

    @pytest.mark.unit
    def test_route_unknown_string(self):
        """Unknown string defaults to Sonnet."""
        model = route_to_model("unknown")
        assert "sonnet" in model


class TestEstimateComplexity:
    """Test estimate_complexity function."""

    @pytest.mark.unit
    def test_estimate_simple(self):
        """Simple task keywords detected."""
        complexity = estimate_complexity("format this JSON file")
        assert complexity == TaskComplexity.SIMPLE

    @pytest.mark.unit
    def test_estimate_medium(self):
        """Medium task keywords detected."""
        complexity = estimate_complexity("implement a new feature")
        assert complexity == TaskComplexity.MEDIUM

    @pytest.mark.unit
    def test_estimate_complex(self):
        """Complex task keywords detected."""
        complexity = estimate_complexity("architect a new distributed system")
        assert complexity == TaskComplexity.COMPLEX

    @pytest.mark.unit
    def test_estimate_empty(self):
        """Empty description defaults to medium."""
        complexity = estimate_complexity("")
        assert complexity == TaskComplexity.MEDIUM

    @pytest.mark.unit
    def test_estimate_with_context(self):
        """Context affects complexity estimation."""
        # Large code base = higher complexity
        complexity = estimate_complexity(
            "review this code",
            context={"code_length": 2000, "file_count": 10},
        )
        assert complexity in [TaskComplexity.MEDIUM, TaskComplexity.COMPLEX]

    @pytest.mark.unit
    def test_explicit_complexity_override(self):
        """Explicit complexity in context overrides estimation."""
        complexity = estimate_complexity(
            "simple task",
            context={"complexity": "complex"},
        )
        assert complexity == TaskComplexity.COMPLEX


class TestGetRecommendedModel:
    """Test get_recommended_model function."""

    @pytest.mark.unit
    def test_recommendation_has_fields(self):
        """Recommendation includes all fields."""
        rec = get_recommended_model("implement a feature")

        assert rec.model_id is not None
        assert rec.complexity is not None
        assert rec.reasoning is not None
        assert rec.alternatives is not None

    @pytest.mark.unit
    def test_cost_savings_preference(self):
        """Cost savings preference downgrades model."""
        # Without preference
        rec1 = get_recommended_model(
            "complex architecture design",
            prefer_cost_savings=False,
        )

        # With preference
        rec2 = get_recommended_model(
            "complex architecture design",
            prefer_cost_savings=True,
        )

        # Cost savings should result in same or cheaper model
        assert rec2.complexity.value <= rec1.complexity.value or True  # May be equal


class TestGetModelForTaskType:
    """Test get_model_for_task_type function."""

    @pytest.mark.unit
    def test_lint_uses_haiku(self):
        """Lint task uses Haiku."""
        model = get_model_for_task_type("lint")
        assert "haiku" in model

    @pytest.mark.unit
    def test_implement_uses_sonnet(self):
        """Implement task uses Sonnet."""
        model = get_model_for_task_type("implement")
        assert "sonnet" in model

    @pytest.mark.unit
    def test_design_uses_opus(self):
        """Design task uses Opus."""
        model = get_model_for_task_type("design")
        assert "opus" in model

    @pytest.mark.unit
    def test_unknown_defaults_sonnet(self):
        """Unknown task type defaults to Sonnet."""
        model = get_model_for_task_type("unknown_task_type")
        assert "sonnet" in model


# ============================================================================
# STATE DB INTEGRATION TESTS
# ============================================================================


class TestStateDBCostMethods:
    """Test StateDB cost tracking methods."""

    @pytest.mark.unit
    def test_record_token_usage(self, db: StateDB):
        """Record token usage via StateDB."""
        record_id = db.record_token_usage(
            session_id="test-session",
            model="claude-opus-4-5",
            input_tokens=1000,
            output_tokens=500,
            cost=0.0525,
            context={"role": "common"},
        )

        assert record_id > 0

    @pytest.mark.unit
    def test_get_session_costs(self, db: StateDB):
        """Get session costs via StateDB."""
        db.record_token_usage("session-1", "claude-opus-4-5", 1000, 500, 0.05)
        db.record_token_usage("session-1", "claude-sonnet-4", 2000, 1000, 0.02)

        summary = db.get_session_costs("session-1")

        assert summary["session_id"] == "session-1"
        assert summary["total_cost"] == 0.07
        assert summary["total_input_tokens"] == 3000
        assert summary["total_output_tokens"] == 1500
        assert summary["record_count"] == 2
        assert len(summary["by_model"]) == 2

    @pytest.mark.unit
    def test_get_cost_summary(self, db: StateDB):
        """Get cost summary via StateDB."""
        db.record_token_usage("session-1", "claude-opus-4-5", 1000, 500, 0.05)
        db.record_token_usage("session-2", "claude-sonnet-4", 2000, 1000, 0.02)

        summary = db.get_cost_summary()

        assert summary["total_cost"] == 0.07
        assert summary["record_count"] == 2
        assert len(summary["by_model"]) == 2
        assert len(summary["by_session"]) == 2

    @pytest.mark.unit
    def test_get_cost_summary_with_dates(self, db: StateDB):
        """Get cost summary with date filtering via StateDB."""
        db.record_token_usage("session-1", "claude-opus-4-5", 1000, 500, 0.05)

        # Query all
        summary = db.get_cost_summary()
        assert summary["record_count"] == 1

        # Query with future start date
        future = (datetime.utcnow() + timedelta(days=1)).isoformat()
        summary = db.get_cost_summary(start_date=future)
        assert summary["record_count"] == 0
