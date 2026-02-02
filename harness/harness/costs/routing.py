"""Model tier routing based on task complexity.

Routes tasks to optimal model tier (Opus, Sonnet, Haiku) based on:
- Task complexity estimation
- Cost optimization goals
- Performance requirements
"""

from dataclasses import dataclass
from enum import Enum


class TaskComplexity(str, Enum):
    """Task complexity levels for model routing."""

    SIMPLE = "simple"  # Haiku - basic tasks, formatting, simple queries
    MEDIUM = "medium"  # Sonnet - standard coding, analysis, moderate reasoning
    COMPLEX = "complex"  # Opus - advanced reasoning, multi-step, novel problems


@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning.

    Attributes:
        model_id: Recommended model identifier
        complexity: Assessed task complexity
        reasoning: Why this model was selected
        alternatives: Other viable models
    """

    model_id: str
    complexity: TaskComplexity
    reasoning: str
    alternatives: list[str]


# Model assignments for each complexity level
COMPLEXITY_MODEL_MAP = {
    TaskComplexity.SIMPLE: "claude-haiku-3-5",
    TaskComplexity.MEDIUM: "claude-sonnet-4",
    TaskComplexity.COMPLEX: "claude-opus-4-5",
}

# Keywords/patterns that suggest task complexity
COMPLEXITY_INDICATORS = {
    TaskComplexity.SIMPLE: [
        "format",
        "list",
        "simple",
        "basic",
        "convert",
        "translate",
        "summarize",
        "extract",
        "parse",
        "validate",
        "check syntax",
        "lint",
        "spell",
    ],
    TaskComplexity.MEDIUM: [
        "implement",
        "code",
        "write",
        "analyze",
        "refactor",
        "debug",
        "test",
        "review",
        "explain",
        "document",
        "standard",
        "typical",
        "normal",
    ],
    TaskComplexity.COMPLEX: [
        "architect",
        "design",
        "optimize",
        "complex",
        "advanced",
        "novel",
        "research",
        "multi-step",
        "reasoning",
        "strategy",
        "plan",
        "creative",
        "innovative",
        "critical",
    ],
}


def estimate_complexity(
    task_description: str,
    context: dict | None = None,
) -> TaskComplexity:
    """Estimate task complexity from description.

    Uses keyword matching and context to estimate complexity.
    Errs on the side of higher complexity when uncertain.

    Args:
        task_description: Description of the task
        context: Optional context (e.g., {"code_length": 500, "has_tests": True})

    Returns:
        Estimated TaskComplexity
    """
    if not task_description:
        return TaskComplexity.MEDIUM  # Default to medium

    description_lower = task_description.lower()

    # Score each complexity level based on keyword matches
    scores = {
        TaskComplexity.SIMPLE: 0,
        TaskComplexity.MEDIUM: 0,
        TaskComplexity.COMPLEX: 0,
    }

    for complexity, keywords in COMPLEXITY_INDICATORS.items():
        for keyword in keywords:
            if keyword in description_lower:
                scores[complexity] += 1

    # Apply context-based adjustments
    if context:
        # Longer code = higher complexity
        code_length = context.get("code_length", 0)
        if code_length > 1000:
            scores[TaskComplexity.COMPLEX] += 2
        elif code_length > 500:
            scores[TaskComplexity.MEDIUM] += 1

        # Multi-file tasks are more complex
        file_count = context.get("file_count", 1)
        if file_count > 5:
            scores[TaskComplexity.COMPLEX] += 2
        elif file_count > 2:
            scores[TaskComplexity.MEDIUM] += 1

        # Explicit complexity override
        if "complexity" in context:
            explicit = context["complexity"]
            if explicit in [c.value for c in TaskComplexity]:
                return TaskComplexity(explicit)

    # Find highest scoring complexity
    max_score = max(scores.values())

    if max_score == 0:
        # No matches - default to medium
        return TaskComplexity.MEDIUM

    # If tied, prefer higher complexity (conservative)
    for complexity in [TaskComplexity.COMPLEX, TaskComplexity.MEDIUM, TaskComplexity.SIMPLE]:
        if scores[complexity] == max_score:
            return complexity

    return TaskComplexity.MEDIUM


def route_to_model(
    task_complexity: TaskComplexity | str,
) -> str:
    """Route task to optimal model based on complexity.

    Args:
        task_complexity: Task complexity level (enum or string)

    Returns:
        Model identifier string
    """
    if isinstance(task_complexity, str):
        try:
            task_complexity = TaskComplexity(task_complexity)
        except ValueError:
            # Unknown complexity, default to Sonnet
            return COMPLEXITY_MODEL_MAP[TaskComplexity.MEDIUM]

    return COMPLEXITY_MODEL_MAP.get(task_complexity, COMPLEXITY_MODEL_MAP[TaskComplexity.MEDIUM])


def get_recommended_model(
    task_description: str,
    context: dict | None = None,
    prefer_cost_savings: bool = False,
) -> ModelRecommendation:
    """Get model recommendation with full reasoning.

    Args:
        task_description: Description of the task
        context: Optional context for complexity estimation
        prefer_cost_savings: If True, bias toward cheaper models

    Returns:
        ModelRecommendation with model, reasoning, and alternatives
    """
    complexity = estimate_complexity(task_description, context)

    # Apply cost savings preference
    if prefer_cost_savings:
        # Downgrade complexity by one level
        if complexity == TaskComplexity.COMPLEX:
            complexity = TaskComplexity.MEDIUM
        elif complexity == TaskComplexity.MEDIUM:
            complexity = TaskComplexity.SIMPLE

    model_id = route_to_model(complexity)

    # Generate reasoning
    if complexity == TaskComplexity.SIMPLE:
        reasoning = (
            "Task appears straightforward. Haiku provides fast, cost-effective "
            "responses for simple queries, formatting, and basic operations."
        )
        alternatives = ["claude-sonnet-4"]
    elif complexity == TaskComplexity.MEDIUM:
        reasoning = (
            "Task requires moderate reasoning and coding ability. Sonnet offers "
            "good balance of capability and cost for standard development tasks."
        )
        alternatives = ["claude-opus-4-5", "claude-haiku-3-5"]
    else:  # COMPLEX
        reasoning = (
            "Task involves advanced reasoning, complex architecture, or novel "
            "problems. Opus provides best-in-class capabilities for demanding tasks."
        )
        alternatives = ["claude-sonnet-4"]

    return ModelRecommendation(
        model_id=model_id,
        complexity=complexity,
        reasoning=reasoning,
        alternatives=alternatives,
    )


def get_model_for_task_type(task_type: str) -> str:
    """Get recommended model for common task types.

    Provides quick routing for known task categories.

    Args:
        task_type: One of: "lint", "format", "test", "review", "implement",
                   "refactor", "design", "research"

    Returns:
        Model identifier
    """
    task_type_map = {
        # Simple tasks -> Haiku
        "lint": "claude-haiku-3-5",
        "format": "claude-haiku-3-5",
        "validate": "claude-haiku-3-5",
        "parse": "claude-haiku-3-5",
        # Medium tasks -> Sonnet
        "test": "claude-sonnet-4",
        "review": "claude-sonnet-4",
        "implement": "claude-sonnet-4",
        "debug": "claude-sonnet-4",
        "document": "claude-sonnet-4",
        # Complex tasks -> Opus
        "refactor": "claude-opus-4-5",
        "design": "claude-opus-4-5",
        "architect": "claude-opus-4-5",
        "research": "claude-opus-4-5",
        "optimize": "claude-opus-4-5",
    }

    return task_type_map.get(task_type.lower(), "claude-sonnet-4")
