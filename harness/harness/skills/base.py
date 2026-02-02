"""Base skill classes for the DAG harness skills framework.

This module provides the foundational skill classes that other skills
inherit from. Skills are reusable, composable capabilities that agents
can invoke to perform common tasks.

Skill Lifecycle:
    1. Registration: Skills are registered with SkillRegistry
    2. Validation: Context and parameters are validated
    3. Execution: The skill action is executed
    4. Result: SkillResult is returned with status and data

Skills vs Hooks:
    - Hooks intercept agent behavior (passive)
    - Skills extend agent capabilities (active)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    """Status of skill execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Partially successful
    SKIPPED = "skipped"
    ERROR = "error"  # Unexpected error


class SkillError(Exception):
    """Exception raised by skill execution."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclass
class SkillContext:
    """Context passed to skill execution.

    Attributes:
        working_dir: Working directory for the skill
        agent_id: ID of the invoking agent
        session_id: Current session ID
        execution_id: Workflow execution ID
        timeout: Timeout in seconds
        env: Environment variables
        metadata: Additional context data
    """

    working_dir: Path
    agent_id: str | None = None
    session_id: str | None = None
    execution_id: int | None = None
    timeout: int = 300
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Result from skill execution.

    Attributes:
        status: Execution status
        message: Human-readable message
        data: Result data (skill-specific)
        errors: List of error messages
        warnings: List of warning messages
        duration_ms: Execution time in milliseconds
        timestamp: When execution completed
        metadata: Additional result metadata
    """

    status: SkillStatus
    message: str = ""
    data: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "message": self.message,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def success(
        cls,
        message: str = "Success",
        data: dict | None = None,
        **kwargs,
    ) -> "SkillResult":
        """Create a success result."""
        return cls(status=SkillStatus.SUCCESS, message=message, data=data, **kwargs)

    @classmethod
    def failure(
        cls,
        message: str = "Failed",
        errors: list[str] | None = None,
        **kwargs,
    ) -> "SkillResult":
        """Create a failure result."""
        return cls(
            status=SkillStatus.FAILURE,
            message=message,
            errors=errors or [],
            **kwargs,
        )

    @classmethod
    def error(
        cls,
        message: str = "Error",
        exception: Exception | None = None,
        **kwargs,
    ) -> "SkillResult":
        """Create an error result."""
        errors = [str(exception)] if exception else []
        return cls(
            status=SkillStatus.ERROR,
            message=message,
            errors=errors,
            **kwargs,
        )


@dataclass
class SkillAction:
    """Definition of a skill action.

    Attributes:
        name: Action name
        description: What the action does
        handler: Async function to execute
        parameters: Expected parameter names
        required_params: Required parameter names
    """

    name: str
    description: str
    handler: Callable[[SkillContext, dict], SkillResult]
    parameters: list[str] = field(default_factory=list)
    required_params: list[str] = field(default_factory=list)

    def validate_params(self, params: dict) -> list[str]:
        """Validate parameters.

        Args:
            params: Parameters to validate

        Returns:
            List of missing required parameters
        """
        missing = []
        for param in self.required_params:
            if param not in params:
                missing.append(param)
        return missing


class Skill(ABC):
    """Abstract base class for skills.

    Skills encapsulate reusable capabilities that agents can invoke.
    Each skill has a name, description, and one or more actions.

    Subclasses should:
    1. Set name and description
    2. Implement _register_actions to define actions
    3. Implement action handler methods
    """

    def __init__(
        self,
        enabled: bool = True,
        config: dict | None = None,
    ):
        """Initialize the skill.

        Args:
            enabled: Whether the skill is active
            config: Optional configuration
        """
        self.enabled = enabled
        self.config = config or {}
        self._actions: dict[str, SkillAction] = {}
        self._stats = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "total_duration_ms": 0,
        }

        # Register actions
        self._register_actions()

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this skill."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this skill does."""
        pass

    @abstractmethod
    def _register_actions(self) -> None:
        """Register skill actions. Called during __init__."""
        pass

    def register_action(self, action: SkillAction) -> None:
        """Register an action.

        Args:
            action: Action to register
        """
        self._actions[action.name] = action
        logger.debug(f"Registered action: {self.name}.{action.name}")

    def get_actions(self) -> list[str]:
        """Get list of available action names.

        Returns:
            List of action names
        """
        return list(self._actions.keys())

    def get_action(self, action_name: str) -> SkillAction | None:
        """Get an action by name.

        Args:
            action_name: Name of the action

        Returns:
            SkillAction if found
        """
        return self._actions.get(action_name)

    async def execute(
        self,
        action_name: str,
        context: SkillContext,
        params: dict | None = None,
    ) -> SkillResult:
        """Execute a skill action.

        Args:
            action_name: Name of the action to execute
            context: Execution context
            params: Action parameters

        Returns:
            SkillResult from execution
        """
        import time

        start_time = time.time()
        params = params or {}

        if not self.enabled:
            return SkillResult(
                status=SkillStatus.SKIPPED,
                message=f"Skill {self.name} is disabled",
            )

        action = self._actions.get(action_name)
        if not action:
            return SkillResult.error(
                f"Unknown action: {action_name}",
                metadata={"available_actions": list(self._actions.keys())},
            )

        # Validate parameters
        missing = action.validate_params(params)
        if missing:
            return SkillResult.error(
                f"Missing required parameters: {missing}",
                metadata={"required": action.required_params},
            )

        try:
            self._stats["executions"] += 1

            # Execute with timeout
            result = action.handler(context, params)
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(
                    result,
                    timeout=context.timeout,
                )

            # Record stats
            duration = int((time.time() - start_time) * 1000)
            result.duration_ms = duration
            self._stats["total_duration_ms"] += duration

            if result.status == SkillStatus.SUCCESS:
                self._stats["successes"] += 1
            else:
                self._stats["failures"] += 1

            return result

        except TimeoutError:
            self._stats["failures"] += 1
            return SkillResult.error(f"Action {action_name} timed out after {context.timeout}s")
        except SkillError as e:
            self._stats["failures"] += 1
            return SkillResult.failure(str(e), metadata=e.details)
        except Exception as e:
            self._stats["failures"] += 1
            logger.error(f"Skill {self.name}.{action_name} error: {e}")
            return SkillResult.error(str(e), exception=e)

    def get_stats(self) -> dict[str, Any]:
        """Get skill statistics.

        Returns:
            Dict with execution stats
        """
        stats = self._stats.copy()
        if stats["executions"] > 0:
            stats["success_rate"] = (stats["successes"] / stats["executions"]) * 100
            stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["executions"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_duration_ms"] = 0
        return stats

    def get_info(self) -> dict[str, Any]:
        """Get skill information.

        Returns:
            Dict with skill info
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "actions": [
                {
                    "name": action.name,
                    "description": action.description,
                    "parameters": action.parameters,
                    "required_params": action.required_params,
                }
                for action in self._actions.values()
            ],
        }


class SkillRegistry:
    """Registry for managing skills.

    The registry provides a central place to register, discover,
    and invoke skills.

    Usage:
        registry = SkillRegistry()
        registry.register(TestingSkill())
        registry.register(DependencySkill())

        result = await registry.execute(
            skill_name="testing",
            action_name="run_tests",
            context=context,
            params={"path": "./tests"}
        )
    """

    def __init__(self):
        """Initialize the registry."""
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill.

        Args:
            skill: Skill instance to register
        """
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name} ({len(skill.get_actions())} actions)")

    def unregister(self, skill_name: str) -> bool:
        """Unregister a skill.

        Args:
            skill_name: Name of skill to remove

        Returns:
            True if skill was found and removed
        """
        if skill_name in self._skills:
            del self._skills[skill_name]
            logger.info(f"Unregistered skill: {skill_name}")
            return True
        return False

    def get_skill(self, skill_name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            skill_name: Skill name

        Returns:
            Skill instance if found
        """
        return self._skills.get(skill_name)

    def list_skills(self) -> list[str]:
        """List registered skill names.

        Returns:
            List of skill names
        """
        return list(self._skills.keys())

    def get_all_actions(self) -> dict[str, list[str]]:
        """Get all actions grouped by skill.

        Returns:
            Dict mapping skill name to list of action names
        """
        return {name: skill.get_actions() for name, skill in self._skills.items()}

    async def execute(
        self,
        skill_name: str,
        action_name: str,
        context: SkillContext,
        params: dict | None = None,
    ) -> SkillResult:
        """Execute a skill action.

        Args:
            skill_name: Name of the skill
            action_name: Name of the action
            context: Execution context
            params: Action parameters

        Returns:
            SkillResult from execution
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return SkillResult.error(
                f"Unknown skill: {skill_name}",
                metadata={"available_skills": list(self._skills.keys())},
            )

        return await skill.execute(action_name, context, params)

    def get_info(self) -> dict[str, Any]:
        """Get information about all registered skills.

        Returns:
            Dict with skill information
        """
        return {
            "skills": [skill.get_info() for skill in self._skills.values()],
            "total_skills": len(self._skills),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all skills.

        Returns:
            Dict with skill stats
        """
        return {name: skill.get_stats() for name, skill in self._skills.items()}


# Type variable for generic skill creation
SkillT = TypeVar("SkillT", bound=Skill)


def create_simple_skill(
    name: str,
    description: str,
    actions: dict[str, Callable[[SkillContext, dict], SkillResult]],
) -> Skill:
    """Create a simple skill from a dict of action handlers.

    Args:
        name: Skill name
        description: Skill description
        actions: Dict mapping action name to handler function

    Returns:
        Skill instance
    """

    class SimpleSkill(Skill):
        @property
        def name(self) -> str:
            return name

        @property
        def description(self) -> str:
            return description

        def _register_actions(self) -> None:
            for action_name, handler in actions.items():
                self.register_action(
                    SkillAction(
                        name=action_name,
                        description=f"{action_name} action",
                        handler=handler,
                    )
                )

    return SimpleSkill()
