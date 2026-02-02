"""Cost tracking and optimization for Claude API usage.

This module provides:
- Pricing constants for Claude models (Opus, Sonnet, Haiku)
- Token usage tracking per session
- Cost aggregation and reporting
- Cost alerts and thresholds
- Model tier routing based on task complexity
"""

from harness.costs.alerts import (
    DEFAULT_COST_THRESHOLDS,
    CostAlert,
    CostAlertLevel,
    CostAlertManager,
)
from harness.costs.pricing import (
    CLAUDE_PRICING,
    ModelPricing,
    calculate_cost,
    get_model_pricing,
)
from harness.costs.routing import (
    TaskComplexity,
    get_recommended_model,
    route_to_model,
)
from harness.costs.tracker import TokenUsageTracker

__all__ = [
    # Pricing
    "ModelPricing",
    "CLAUDE_PRICING",
    "get_model_pricing",
    "calculate_cost",
    # Tracker
    "TokenUsageTracker",
    # Alerts
    "CostAlert",
    "CostAlertLevel",
    "CostAlertManager",
    "DEFAULT_COST_THRESHOLDS",
    # Routing
    "TaskComplexity",
    "route_to_model",
    "get_recommended_model",
]
