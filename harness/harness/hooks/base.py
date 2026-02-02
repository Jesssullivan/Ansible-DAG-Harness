"""Base hook classes for the DAG harness hooks framework.

This module provides the foundational hook classes that other hooks
inherit from. Hooks allow intercepting and modifying agent behavior
at various lifecycle points.

Hook Lifecycle:
    1. PreToolUse: Before a tool is executed
    2. PostToolUse: After a tool completes
    3. SubagentStart: When a subagent is spawned
    4. SubagentStop: When a subagent completes

Each hook can:
    - Modify inputs (PreToolUse)
    - Log/audit activity (all hooks)
    - Block actions (PreToolUse via HookResult)
    - Trigger notifications (SubagentStop)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class HookPriority(IntEnum):
    """Priority levels for hook execution order.

    Lower values execute first. Same priority executes in registration order.
    """

    CRITICAL = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    AUDIT = 100  # Audit hooks run last


class HookResult(Enum):
    """Result of hook execution."""

    CONTINUE = "continue"  # Continue with execution
    BLOCK = "block"  # Block the action
    MODIFY = "modify"  # Action was modified but should continue


@dataclass
class HookContext:
    """Context passed to all hooks.

    Attributes:
        agent_id: ID of the agent triggering the hook
        session_id: Current session ID
        execution_id: Workflow execution ID if applicable
        timestamp: When the hook was triggered
        metadata: Additional context-specific data
        parent_agent_id: Parent agent ID for subagents
        tool_name: Name of tool being used (for tool hooks)
        tool_input: Tool input data (for PreToolUse)
        tool_output: Tool output data (for PostToolUse)
    """

    agent_id: str
    session_id: str | None = None
    execution_id: int | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_agent_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None


class Hook(ABC):
    """Abstract base class for all hooks.

    Hooks are registered with the HookManager and called at appropriate
    lifecycle points. Each hook has a priority and can be enabled/disabled.

    Subclasses should implement the appropriate on_* methods for the
    hook type they represent.
    """

    def __init__(
        self,
        name: str,
        priority: HookPriority = HookPriority.NORMAL,
        enabled: bool = True,
    ):
        """Initialize a hook.

        Args:
            name: Unique identifier for this hook
            priority: Execution priority (lower runs first)
            enabled: Whether the hook is active
        """
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self._stats = {
            "invocations": 0,
            "errors": 0,
            "last_invocation": None,
            "last_error": None,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get hook execution statistics."""
        return self._stats.copy()

    def _record_invocation(self) -> None:
        """Record that the hook was invoked."""
        self._stats["invocations"] += 1
        self._stats["last_invocation"] = datetime.utcnow().isoformat()

    def _record_error(self, error: Exception) -> None:
        """Record that an error occurred."""
        self._stats["errors"] += 1
        self._stats["last_error"] = str(error)


class PreToolUseHook(Hook):
    """Hook that runs before a tool is executed.

    PreToolUse hooks can:
    - Modify the tool input before execution
    - Block tool execution entirely
    - Log/audit the tool invocation

    Return value from on_pre_tool_use determines behavior:
    - (HookResult.CONTINUE, input): Continue with original or modified input
    - (HookResult.BLOCK, None): Block tool execution
    - (HookResult.MODIFY, modified_input): Continue with modified input
    """

    @abstractmethod
    async def on_pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: HookContext,
    ) -> tuple[HookResult, dict[str, Any] | None]:
        """Called before a tool is executed.

        Args:
            tool_name: Name of the tool being used
            tool_input: Input parameters for the tool
            context: Hook context with agent/session info

        Returns:
            Tuple of (HookResult, modified_input or None)
        """
        pass


class PostToolUseHook(Hook):
    """Hook that runs after a tool completes.

    PostToolUse hooks can:
    - Log/audit tool output
    - Trigger follow-up actions
    - Record metrics

    The tool output cannot be modified by this hook.
    """

    @abstractmethod
    async def on_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
        context: HookContext,
    ) -> None:
        """Called after a tool completes execution.

        Args:
            tool_name: Name of the tool that was used
            tool_input: Input parameters that were passed
            tool_output: Output from the tool
            context: Hook context with agent/session info
        """
        pass


class SubagentStartHook(Hook):
    """Hook that runs when a subagent is spawned.

    SubagentStart hooks can:
    - Log/audit subagent creation
    - Inject context into the subagent
    - Block subagent creation (via HookResult.BLOCK)

    Return value from on_subagent_start determines behavior:
    - (HookResult.CONTINUE, context): Continue with optional modified context
    - (HookResult.BLOCK, None): Block subagent creation
    """

    @abstractmethod
    async def on_subagent_start(
        self,
        agent_id: str,
        task: str,
        context: HookContext,
    ) -> tuple[HookResult, dict[str, Any] | None]:
        """Called when a subagent is about to be spawned.

        Args:
            agent_id: ID of the new subagent
            task: Task description for the subagent
            context: Hook context with parent agent info

        Returns:
            Tuple of (HookResult, modified_context or None)
        """
        pass


class SubagentStopHook(Hook):
    """Hook that runs when a subagent completes.

    SubagentStop hooks can:
    - Log/audit subagent completion
    - Trigger notifications
    - Record metrics
    - Archive subagent session data

    This hook is informational only - it cannot modify the result.
    """

    @abstractmethod
    async def on_subagent_stop(
        self,
        agent_id: str,
        result: dict[str, Any],
        context: HookContext,
    ) -> None:
        """Called when a subagent completes.

        Args:
            agent_id: ID of the completed subagent
            result: Result dict with status, output, errors
            context: Hook context with agent info
        """
        pass


class HookManager:
    """Manages hook registration and execution.

    The HookManager maintains separate registries for each hook type
    and handles executing hooks in priority order.

    Usage:
        manager = HookManager()
        manager.register(my_pre_tool_hook)
        manager.register(my_audit_hook)

        # Execute hooks
        result, modified_input = await manager.execute_pre_tool_use(
            tool_name="Write",
            tool_input={"file_path": "/tmp/file.txt"},
            context=context
        )
    """

    def __init__(self):
        """Initialize the hook manager."""
        self._pre_tool_hooks: list[PreToolUseHook] = []
        self._post_tool_hooks: list[PostToolUseHook] = []
        self._subagent_start_hooks: list[SubagentStartHook] = []
        self._subagent_stop_hooks: list[SubagentStopHook] = []

    def register(self, hook: Hook) -> None:
        """Register a hook.

        Args:
            hook: Hook instance to register

        Raises:
            TypeError: If hook type is not recognized
        """
        if isinstance(hook, PreToolUseHook):
            self._pre_tool_hooks.append(hook)
            self._pre_tool_hooks.sort(key=lambda h: h.priority)
        elif isinstance(hook, PostToolUseHook):
            self._post_tool_hooks.append(hook)
            self._post_tool_hooks.sort(key=lambda h: h.priority)
        elif isinstance(hook, SubagentStartHook):
            self._subagent_start_hooks.append(hook)
            self._subagent_start_hooks.sort(key=lambda h: h.priority)
        elif isinstance(hook, SubagentStopHook):
            self._subagent_stop_hooks.append(hook)
            self._subagent_stop_hooks.sort(key=lambda h: h.priority)
        else:
            raise TypeError(f"Unknown hook type: {type(hook)}")

        logger.debug(f"Registered hook: {hook.name} (priority={hook.priority})")

    def unregister(self, hook_name: str) -> bool:
        """Unregister a hook by name.

        Args:
            hook_name: Name of the hook to remove

        Returns:
            True if hook was found and removed
        """
        for hook_list in [
            self._pre_tool_hooks,
            self._post_tool_hooks,
            self._subagent_start_hooks,
            self._subagent_stop_hooks,
        ]:
            for hook in hook_list:
                if hook.name == hook_name:
                    hook_list.remove(hook)
                    logger.debug(f"Unregistered hook: {hook_name}")
                    return True
        return False

    def get_hook(self, hook_name: str) -> Hook | None:
        """Get a hook by name.

        Args:
            hook_name: Name of the hook

        Returns:
            Hook instance if found, None otherwise
        """
        for hook_list in [
            self._pre_tool_hooks,
            self._post_tool_hooks,
            self._subagent_start_hooks,
            self._subagent_stop_hooks,
        ]:
            for hook in hook_list:
                if hook.name == hook_name:
                    return hook
        return None

    def list_hooks(self) -> dict[str, list[str]]:
        """List all registered hooks by type.

        Returns:
            Dict mapping hook type to list of hook names
        """
        return {
            "pre_tool_use": [h.name for h in self._pre_tool_hooks],
            "post_tool_use": [h.name for h in self._post_tool_hooks],
            "subagent_start": [h.name for h in self._subagent_start_hooks],
            "subagent_stop": [h.name for h in self._subagent_stop_hooks],
        }

    async def execute_pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: HookContext,
    ) -> tuple[HookResult, dict[str, Any]]:
        """Execute all PreToolUse hooks in priority order.

        Args:
            tool_name: Name of the tool being used
            tool_input: Original tool input
            context: Hook context

        Returns:
            Tuple of (final result, possibly modified input)
        """
        current_input = tool_input.copy()

        for hook in self._pre_tool_hooks:
            if not hook.enabled:
                continue

            try:
                hook._record_invocation()
                result, modified_input = await hook.on_pre_tool_use(
                    tool_name, current_input, context
                )

                if result == HookResult.BLOCK:
                    logger.info(f"Hook {hook.name} blocked tool {tool_name}")
                    return HookResult.BLOCK, current_input

                if result == HookResult.MODIFY and modified_input is not None:
                    current_input = modified_input
                    logger.debug(f"Hook {hook.name} modified input for {tool_name}")

            except Exception as e:
                hook._record_error(e)
                logger.error(f"Error in hook {hook.name}: {e}")
                # Continue with other hooks on error

        return HookResult.CONTINUE, current_input

    async def execute_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
        context: HookContext,
    ) -> None:
        """Execute all PostToolUse hooks in priority order.

        Args:
            tool_name: Name of the tool that was used
            tool_input: Tool input that was passed
            tool_output: Tool output
            context: Hook context
        """
        for hook in self._post_tool_hooks:
            if not hook.enabled:
                continue

            try:
                hook._record_invocation()
                await hook.on_post_tool_use(tool_name, tool_input, tool_output, context)
            except Exception as e:
                hook._record_error(e)
                logger.error(f"Error in hook {hook.name}: {e}")

    async def execute_subagent_start(
        self,
        agent_id: str,
        task: str,
        context: HookContext,
    ) -> tuple[HookResult, dict[str, Any] | None]:
        """Execute all SubagentStart hooks in priority order.

        Args:
            agent_id: ID of the new subagent
            task: Task description
            context: Hook context

        Returns:
            Tuple of (result, possibly modified context)
        """
        modified_context: dict[str, Any] | None = None

        for hook in self._subagent_start_hooks:
            if not hook.enabled:
                continue

            try:
                hook._record_invocation()
                result, ctx_update = await hook.on_subagent_start(agent_id, task, context)

                if result == HookResult.BLOCK:
                    logger.info(f"Hook {hook.name} blocked subagent {agent_id}")
                    return HookResult.BLOCK, None

                if ctx_update:
                    if modified_context is None:
                        modified_context = {}
                    modified_context.update(ctx_update)

            except Exception as e:
                hook._record_error(e)
                logger.error(f"Error in hook {hook.name}: {e}")

        return HookResult.CONTINUE, modified_context

    async def execute_subagent_stop(
        self,
        agent_id: str,
        result: dict[str, Any],
        context: HookContext,
    ) -> None:
        """Execute all SubagentStop hooks in priority order.

        Args:
            agent_id: ID of the completed subagent
            result: Result dict
            context: Hook context
        """
        for hook in self._subagent_stop_hooks:
            if not hook.enabled:
                continue

            try:
                hook._record_invocation()
                await hook.on_subagent_stop(agent_id, result, context)
            except Exception as e:
                hook._record_error(e)
                logger.error(f"Error in hook {hook.name}: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all hooks.

        Returns:
            Dict with hook stats by name
        """
        stats = {}
        for hook_list in [
            self._pre_tool_hooks,
            self._post_tool_hooks,
            self._subagent_start_hooks,
            self._subagent_stop_hooks,
        ]:
            for hook in hook_list:
                stats[hook.name] = hook.get_stats()
        return stats


# Convenience function for creating hook callbacks
def create_simple_pre_tool_hook(
    name: str,
    callback: Callable[[str, dict, HookContext], tuple[HookResult, dict | None]],
    priority: HookPriority = HookPriority.NORMAL,
) -> PreToolUseHook:
    """Create a simple PreToolUse hook from a callback function.

    Args:
        name: Hook name
        callback: Function to call (can be sync or async)
        priority: Hook priority

    Returns:
        PreToolUseHook instance
    """

    class SimplePreToolHook(PreToolUseHook):
        async def on_pre_tool_use(
            self, tool_name: str, tool_input: dict, context: HookContext
        ) -> tuple[HookResult, dict | None]:
            result = callback(tool_name, tool_input, context)
            if asyncio.iscoroutine(result):
                return await result
            return result

    return SimplePreToolHook(name=name, priority=priority)


def create_simple_subagent_stop_hook(
    name: str,
    callback: Callable[[str, dict, HookContext], None],
    priority: HookPriority = HookPriority.AUDIT,
) -> SubagentStopHook:
    """Create a simple SubagentStop hook from a callback function.

    Args:
        name: Hook name
        callback: Function to call (can be sync or async)
        priority: Hook priority

    Returns:
        SubagentStopHook instance
    """

    class SimpleSubagentStopHook(SubagentStopHook):
        async def on_subagent_stop(self, agent_id: str, result: dict, context: HookContext) -> None:
            res = callback(agent_id, result, context)
            if asyncio.iscoroutine(res):
                await res

    return SimpleSubagentStopHook(name=name, priority=priority)
