"""Pricing constants for Claude models.

Pricing is per 1 million tokens (MTok).
All costs are in USD.
"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelPricing:
    """Pricing information for a Claude model.

    Attributes:
        model_id: The model identifier (e.g., "claude-opus-4-5")
        input_cost_per_mtok: Cost per million input tokens in USD
        output_cost_per_mtok: Cost per million output tokens in USD
        display_name: Human-readable model name
    """

    model_id: str
    input_cost_per_mtok: Decimal
    output_cost_per_mtok: Decimal
    display_name: str

    @property
    def input_cost_per_token(self) -> Decimal:
        """Cost per single input token."""
        return self.input_cost_per_mtok / Decimal("1000000")

    @property
    def output_cost_per_token(self) -> Decimal:
        """Cost per single output token."""
        return self.output_cost_per_mtok / Decimal("1000000")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Calculate total cost for given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD as Decimal
        """
        input_cost = Decimal(input_tokens) * self.input_cost_per_token
        output_cost = Decimal(output_tokens) * self.output_cost_per_token
        return input_cost + output_cost


# Claude model pricing as of 2025 (per 1M tokens)
# https://www.anthropic.com/pricing
CLAUDE_PRICING: dict[str, ModelPricing] = {
    # Claude Opus 4.5 - Most capable
    "claude-opus-4-5": ModelPricing(
        model_id="claude-opus-4-5",
        input_cost_per_mtok=Decimal("15.00"),
        output_cost_per_mtok=Decimal("75.00"),
        display_name="Claude Opus 4.5",
    ),
    "claude-opus-4-5-20251101": ModelPricing(
        model_id="claude-opus-4-5-20251101",
        input_cost_per_mtok=Decimal("15.00"),
        output_cost_per_mtok=Decimal("75.00"),
        display_name="Claude Opus 4.5 (20251101)",
    ),
    # Claude Sonnet 4 - Balanced performance/cost
    "claude-sonnet-4": ModelPricing(
        model_id="claude-sonnet-4",
        input_cost_per_mtok=Decimal("3.00"),
        output_cost_per_mtok=Decimal("15.00"),
        display_name="Claude Sonnet 4",
    ),
    "claude-sonnet-4-20250514": ModelPricing(
        model_id="claude-sonnet-4-20250514",
        input_cost_per_mtok=Decimal("3.00"),
        output_cost_per_mtok=Decimal("15.00"),
        display_name="Claude Sonnet 4 (20250514)",
    ),
    # Claude Haiku 3.5 - Fast and cost-effective
    "claude-haiku-3-5": ModelPricing(
        model_id="claude-haiku-3-5",
        input_cost_per_mtok=Decimal("0.25"),
        output_cost_per_mtok=Decimal("1.25"),
        display_name="Claude Haiku 3.5",
    ),
    "claude-haiku-3-5-20241022": ModelPricing(
        model_id="claude-haiku-3-5-20241022",
        input_cost_per_mtok=Decimal("0.25"),
        output_cost_per_mtok=Decimal("1.25"),
        display_name="Claude Haiku 3.5 (20241022)",
    ),
    # Legacy models (Sonnet 3.5)
    "claude-3-5-sonnet-20241022": ModelPricing(
        model_id="claude-3-5-sonnet-20241022",
        input_cost_per_mtok=Decimal("3.00"),
        output_cost_per_mtok=Decimal("15.00"),
        display_name="Claude 3.5 Sonnet",
    ),
}

# Model tier aliases for easier reference
MODEL_TIERS = {
    "opus": "claude-opus-4-5",
    "sonnet": "claude-sonnet-4",
    "haiku": "claude-haiku-3-5",
}


def get_model_pricing(model_id: str) -> ModelPricing | None:
    """Get pricing for a model by ID.

    Handles both exact model IDs and tier aliases.

    Args:
        model_id: Model identifier or tier alias

    Returns:
        ModelPricing if found, None otherwise
    """
    # Check for exact match
    if model_id in CLAUDE_PRICING:
        return CLAUDE_PRICING[model_id]

    # Check tier aliases
    if model_id.lower() in MODEL_TIERS:
        return CLAUDE_PRICING[MODEL_TIERS[model_id.lower()]]

    # Try partial match (e.g., "opus" in "claude-opus-4-5")
    model_lower = model_id.lower()
    for key, pricing in CLAUDE_PRICING.items():
        if model_lower in key.lower():
            return pricing

    return None


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    """Calculate cost for token usage on a specific model.

    Args:
        model_id: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD as Decimal, or None if model not found
    """
    pricing = get_model_pricing(model_id)
    if pricing is None:
        return None
    return pricing.calculate_cost(input_tokens, output_tokens)
