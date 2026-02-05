"""Claude Agent SDK integration for HOTL autonomous operation.

This module provides in-process integration with Claude Code via the
official `claude-agent-sdk` package, replacing subprocess spawning
with native Python SDK calls.

Features:
- ClaudeSDKClient for bidirectional, interactive conversations
- In-process MCP tools via @tool decorator
- Hook support for PreToolUse/PostToolUse events
- Session resumption support
- Async-first design with sync fallbacks
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from harness.hotl.agent_session import (
    AgentSession,
    AgentSessionManager,
    AgentStatus,
    FileChange,
    FileChangeType,
)

logger = logging.getLogger(__name__)


# Type definitions for SDK integration
class HookEvent(str, Enum):
    """Supported hook event types."""

    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    STOP = "Stop"


class PermissionMode(str, Enum):
    """Permission modes for controlling tool execution."""

    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS_PERMISSIONS = "bypassPermissions"


class ToolDecision(TypedDict, total=False):
    """Decision returned by PreToolUse hooks."""

    permissionDecision: str  # "allow" | "deny"
    permissionDecisionReason: str | None


class HookOutput(TypedDict, total=False):
    """Output structure for hook callbacks."""

    decision: str | None  # "block" to block the action
    systemMessage: str | None  # Warning message for user
    reason: str | None  # Feedback for Claude
    hookSpecificOutput: dict[str, Any] | None


@dataclass
class SDKAgentConfig:
    """Configuration for Claude Agent SDK integration.

    Attributes:
        system_prompt: Custom system prompt for the agent
        allowed_tools: List of tools Claude can use (Read, Write, Bash, etc.)
        permission_mode: Permission mode for tool execution
        max_turns: Maximum conversation turns
        max_budget_usd: Maximum budget in USD for the session
        model: Claude model to use
        cwd: Working directory for the agent
        env: Environment variables to pass
        resume_session_id: Session ID to resume from
        enable_file_checkpointing: Enable file change tracking for rewinding
    """

    system_prompt: str | None = None
    allowed_tools: list[str] = field(
        default_factory=lambda: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    )
    permission_mode: PermissionMode = PermissionMode.ACCEPT_EDITS
    max_turns: int | None = None
    max_budget_usd: float | None = None
    model: str | None = None
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    resume_session_id: str | None = None
    enable_file_checkpointing: bool = True
    default_timeout: int = 600  # 10 minutes


# SDK import handling with graceful fallback
_SDK_AVAILABLE = False
_SDK_IMPORT_ERROR: Exception | None = None

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ClaudeSDKError,
        CLIJSONDecodeError,
        CLINotFoundError,
        HookMatcher,
        ProcessError,
        ResultMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        create_sdk_mcp_server,
        tool,
    )

    _SDK_AVAILABLE = True
except ImportError as e:
    _SDK_IMPORT_ERROR = e
    logger.warning(f"claude-agent-sdk not available: {e}. SDK integration disabled.")

    # Placeholder classes for type hints when SDK not available
    class ClaudeSDKClient:  # type: ignore
        pass

    class ClaudeAgentOptions:  # type: ignore
        pass


def sdk_available() -> bool:
    """Check if the Claude Agent SDK is available."""
    return _SDK_AVAILABLE


def get_sdk_import_error() -> Exception | None:
    """Get the import error if SDK is not available."""
    return _SDK_IMPORT_ERROR


# =============================================================================
# IN-PROCESS MCP TOOLS
# =============================================================================


def create_hotl_mcp_tools(
    session_manager: AgentSessionManager,
    current_session_id_getter: Callable[[], str | None],
) -> Any | None:
    """Create in-process MCP server with HOTL tools.

    Args:
        session_manager: Session manager for updating session state
        current_session_id_getter: Callable that returns the current session ID

    Returns:
        McpSdkServerConfig if SDK available, None otherwise
    """
    if not _SDK_AVAILABLE:
        return None

    @tool("agent_report_progress", "Report progress on the current task", {"message": str})
    async def agent_report_progress(args: dict[str, Any]) -> dict[str, Any]:
        """Report progress on the current HOTL task.

        This tool allows the agent to report its progress, which is
        tracked in the session and can be used for monitoring.
        """
        session_id = current_session_id_getter()
        if not session_id:
            return {
                "content": [{"type": "text", "text": "Error: No active session"}],
                "is_error": True,
            }

        message = args.get("message", "")
        session = session_manager.get_session(session_id)
        if session:
            session.add_progress(message)
            session_manager.update_session(session)
            logger.info(f"Agent {session_id[:8]} progress: {message[:100]}")
            return {"content": [{"type": "text", "text": f"Progress reported: {message}"}]}
        return {
            "content": [{"type": "text", "text": f"Session not found: {session_id}"}],
            "is_error": True,
        }

    @tool(
        "agent_request_intervention",
        "Request human intervention when stuck or need approval",
        {"reason": str, "context": str},
    )
    async def agent_request_intervention(args: dict[str, Any]) -> dict[str, Any]:
        """Request human intervention for the current task.

        Use this when you encounter a problem you cannot solve autonomously,
        need approval for potentially destructive operations, or need
        clarification on the task requirements.
        """
        session_id = current_session_id_getter()
        if not session_id:
            return {
                "content": [{"type": "text", "text": "Error: No active session"}],
                "is_error": True,
            }

        reason = args.get("reason", "Unknown reason")
        context = args.get("context", "")

        session = session_manager.get_session(session_id)
        if session:
            full_reason = f"{reason}\n\nContext: {context}" if context else reason
            session.request_intervention(full_reason)
            session_manager.update_session(session)
            logger.warning(f"Agent {session_id[:8]} requested intervention: {reason}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Intervention requested. A human will review: {reason}",
                    }
                ]
            }
        return {
            "content": [{"type": "text", "text": f"Session not found: {session_id}"}],
            "is_error": True,
        }

    @tool(
        "agent_log_file_operation",
        "Log a file operation (create, modify, delete, rename)",
        {
            "file_path": str,
            "operation": str,  # create, modify, delete, rename
            "old_path": str,  # For renames only (optional via JSON schema)
            "diff": str,  # Optional diff content
        },
    )
    async def agent_log_file_operation(args: dict[str, Any]) -> dict[str, Any]:
        """Log a file operation performed during the task.

        This helps track all file changes made by the agent for
        review and potential rollback.
        """
        session_id = current_session_id_getter()
        if not session_id:
            return {
                "content": [{"type": "text", "text": "Error: No active session"}],
                "is_error": True,
            }

        file_path = args.get("file_path", "")
        operation = args.get("operation", "modify").lower()
        old_path = args.get("old_path")
        diff = args.get("diff")

        # Map operation to FileChangeType
        operation_map = {
            "create": FileChangeType.CREATE,
            "modify": FileChangeType.MODIFY,
            "delete": FileChangeType.DELETE,
            "rename": FileChangeType.RENAME,
        }

        change_type = operation_map.get(operation, FileChangeType.MODIFY)

        session = session_manager.get_session(session_id)
        if session:
            file_change = FileChange(
                file_path=file_path,
                change_type=change_type,
                old_path=old_path,
                diff=diff,
            )
            session.add_file_change(file_change)
            session_manager.update_session(session)
            logger.debug(f"Agent {session_id[:8]} logged file op: {operation} {file_path}")
            return {"content": [{"type": "text", "text": f"Logged {operation}: {file_path}"}]}
        return {
            "content": [{"type": "text", "text": f"Session not found: {session_id}"}],
            "is_error": True,
        }

    # Create the SDK MCP server with our tools
    server = create_sdk_mcp_server(
        name="hotl-agent-tools",
        version="1.0.0",
        tools=[agent_report_progress, agent_request_intervention, agent_log_file_operation],
    )

    return server


# =============================================================================
# SDK INTEGRATION CLASS
# =============================================================================


class SDKClaudeIntegration:
    """
    Manages Claude Code agent integration via the official claude-agent-sdk.

    This class provides in-process integration with Claude Code, replacing
    subprocess spawning with native SDK calls for better performance and
    tighter integration.

    Features:
    - Bidirectional, interactive conversations via ClaudeSDKClient
    - Custom in-process MCP tools for HOTL operations
    - Hook support for intercepting tool use
    - Session resumption support
    - Async-first design with sync fallbacks
    """

    def __init__(
        self,
        config: SDKAgentConfig | None = None,
        session_manager: AgentSessionManager | None = None,
        db: Any | None = None,
    ):
        """Initialize SDK integration.

        Args:
            config: Agent configuration
            session_manager: Optional session manager (created if not provided)
            db: Optional StateDB for persistence

        Raises:
            RuntimeError: If claude-agent-sdk is not installed
        """
        if not _SDK_AVAILABLE:
            raise RuntimeError(
                f"claude-agent-sdk is not installed. Install with: pip install claude-agent-sdk\n"
                f"Import error: {_SDK_IMPORT_ERROR}"
            )

        self.config = config or SDKAgentConfig()
        self.session_manager = session_manager or AgentSessionManager(db=db)
        self.db = db

        # Track active SDK clients
        self._active_clients: dict[str, ClaudeSDKClient] = {}
        self._current_session_id: str | None = None

        # Create MCP tools server
        self._mcp_server = create_hotl_mcp_tools(
            session_manager=self.session_manager,
            current_session_id_getter=lambda: self._current_session_id,
        )

        # Callbacks for agent events
        self._on_complete: Callable[[AgentSession], None] | None = None
        self._on_progress: Callable[[str, str], None] | None = None
        self._on_intervention: Callable[[AgentSession], None] | None = None
        self._on_tool_use: Callable[[str, str, dict], None] | None = None

        # Hook handlers
        self._pre_tool_hooks: list[Callable] = []
        self._post_tool_hooks: list[Callable] = []

    def set_callbacks(
        self,
        on_complete: Callable[[AgentSession], None] | None = None,
        on_progress: Callable[[str, str], None] | None = None,
        on_intervention: Callable[[AgentSession], None] | None = None,
        on_tool_use: Callable[[str, str, dict], None] | None = None,
    ) -> None:
        """Set callbacks for agent events.

        Args:
            on_complete: Called when agent completes (success or failure)
            on_progress: Called with (session_id, progress_message)
            on_intervention: Called when agent needs human help
            on_tool_use: Called with (session_id, tool_name, tool_input)
        """
        self._on_complete = on_complete
        self._on_progress = on_progress
        self._on_intervention = on_intervention
        self._on_tool_use = on_tool_use

    def add_pre_tool_hook(self, hook: Callable) -> None:
        """Add a PreToolUse hook.

        Args:
            hook: Async function with signature:
                  (input_data: dict, tool_use_id: str, context: dict) -> dict
        """
        self._pre_tool_hooks.append(hook)

    def add_post_tool_hook(self, hook: Callable) -> None:
        """Add a PostToolUse hook.

        Args:
            hook: Async function with signature:
                  (input_data: dict, tool_use_id: str, context: dict) -> dict
        """
        self._post_tool_hooks.append(hook)

    def _build_sdk_options(
        self,
        session: AgentSession,
        resume_from: str | None = None,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for a session.

        Args:
            session: Agent session
            resume_from: Optional session ID to resume from

        Returns:
            Configured ClaudeAgentOptions
        """
        # Build system prompt
        system_prompt = self._build_system_prompt(session)

        # Build hooks config
        hooks = self._build_hooks_config()

        # Build MCP servers config
        mcp_servers = {}
        if self._mcp_server:
            mcp_servers["hotl"] = self._mcp_server

        # Build allowed tools list including HOTL tools
        allowed_tools = list(self.config.allowed_tools)
        allowed_tools.extend(
            [
                "mcp__hotl__agent_report_progress",
                "mcp__hotl__agent_request_intervention",
                "mcp__hotl__agent_log_file_operation",
            ]
        )

        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            permission_mode=self.config.permission_mode.value,
            max_turns=self.config.max_turns,
            max_budget_usd=self.config.max_budget_usd,
            model=self.config.model,
            cwd=str(session.working_dir),
            env={
                **self.config.env,
                "HOTL_SESSION_ID": session.id,
                "HOTL_EXECUTION_ID": str(session.execution_id or ""),
            },
            resume=resume_from,
            enable_file_checkpointing=self.config.enable_file_checkpointing,
            mcp_servers=mcp_servers,
            hooks=hooks if hooks else None,
        )

    def _build_system_prompt(self, session: AgentSession) -> str:
        """Build system prompt for the agent.

        Args:
            session: Agent session

        Returns:
            Formatted system prompt
        """
        if self.config.system_prompt:
            base_prompt = self.config.system_prompt
        else:
            base_prompt = "You are a helpful AI assistant."

        hotl_instructions = f"""
You are operating in HOTL (Human Out of The Loop) mode as part of an automated workflow.

## Session Information
- Session ID: {session.id}
- Working Directory: {session.working_dir}
- Task: {session.task}

## Available HOTL Tools
Use these tools to communicate with the HOTL supervisor:

1. **agent_report_progress**: Report your progress on the task
   - Use periodically to update status
   - Helps with monitoring and logging

2. **agent_request_intervention**: Request human help when needed
   - Use when stuck on a problem you cannot solve
   - Use when you need approval for potentially destructive operations
   - Provide clear context about what help is needed

3. **agent_log_file_operation**: Log file changes you make
   - Call after creating, modifying, or deleting files
   - Helps track all changes for review

## Guidelines
1. Work autonomously to complete the task
2. Report progress periodically using agent_report_progress
3. If you encounter insurmountable problems, use agent_request_intervention
4. Track all file changes with agent_log_file_operation
5. Be thorough but efficient in your approach
"""

        # Add context if provided
        context_section = ""
        if session.context:
            context_section = "\n## Task Context\n"
            for key, value in session.context.items():
                if isinstance(value, dict | list):
                    context_section += f"**{key}**: {json.dumps(value, indent=2)}\n"
                else:
                    context_section += f"**{key}**: {value}\n"

        return f"{base_prompt}\n{hotl_instructions}{context_section}"

    def _build_hooks_config(self) -> dict | None:
        """Build hooks configuration for the SDK.

        Returns:
            Hooks configuration dict or None if no hooks registered
        """
        if not self._pre_tool_hooks and not self._post_tool_hooks:
            return None

        hooks = {}

        if self._pre_tool_hooks:
            hooks["PreToolUse"] = [HookMatcher(hooks=self._pre_tool_hooks)]

        if self._post_tool_hooks:
            hooks["PostToolUse"] = [HookMatcher(hooks=self._post_tool_hooks)]

        return hooks

    async def spawn_agent_async(
        self,
        task: str,
        working_dir: Path,
        context: dict[str, Any] | None = None,
        timeout: int | None = None,
        execution_id: int | None = None,
        resume_from: str | None = None,
    ) -> AgentSession:
        """Spawn a new Claude agent asynchronously.

        Args:
            task: Task description/prompt for the agent
            working_dir: Working directory for the agent
            context: Optional context information
            timeout: Optional timeout in seconds
            execution_id: Optional link to workflow execution
            resume_from: Optional session ID to resume from

        Returns:
            Created AgentSession
        """
        # Create session
        session = self.session_manager.create_session(
            task=task,
            working_dir=working_dir,
            context=context or {},
            execution_id=execution_id,
        )

        # Build SDK options
        options = self._build_sdk_options(session, resume_from)

        # Set current session for MCP tools
        self._current_session_id = session.id

        try:
            session.mark_started()
            self.session_manager.update_session(session)

            logger.info(f"Spawning SDK agent {session.id} for task: {task[:50]}...")

            # Create client and run
            async with ClaudeSDKClient(options=options) as client:
                self._active_clients[session.id] = client

                # Send the task prompt
                await client.query(task)

                # Process messages
                sdk_session_id: str | None = None
                async for message in client.receive_response():
                    await self._process_message(session, message)

                    # Extract SDK session ID from ResultMessage
                    if isinstance(message, ResultMessage):
                        sdk_session_id = getattr(message, "session_id", None)
                        if sdk_session_id:
                            session.context["sdk_session_id"] = sdk_session_id

                        # Check for errors
                        if message.is_error:
                            session.mark_failed(message.result or "Unknown error")
                        else:
                            session.mark_completed(message.result or "")

        except CLINotFoundError as e:
            logger.error(f"Claude Code CLI not found: {e}")
            session.mark_failed(f"Claude Code CLI not found: {e}")
        except ProcessError as e:
            logger.error(f"Process error for agent {session.id}: {e}")
            session.mark_failed(f"Process error: {e}")
        except CLIJSONDecodeError as e:
            logger.error(f"JSON decode error for agent {session.id}: {e}")
            session.mark_failed(f"JSON decode error: {e}")
        except ClaudeSDKError as e:
            logger.error(f"SDK error for agent {session.id}: {e}")
            session.mark_failed(f"SDK error: {e}")
        except TimeoutError:
            logger.warning(f"Agent {session.id} timed out")
            session.mark_failed(f"Agent timed out after {timeout or self.config.default_timeout}s")
        except Exception as e:
            logger.error(f"Unexpected error for agent {session.id}: {e}")
            session.mark_failed(str(e))
        finally:
            self._active_clients.pop(session.id, None)
            self._current_session_id = None
            self.session_manager.update_session(session)

            # Call completion callback
            if self._on_complete:
                try:
                    self._on_complete(session)
                except Exception as e:
                    logger.warning(f"Complete callback error: {e}")

            # Check for intervention
            if session.status == AgentStatus.NEEDS_HUMAN and self._on_intervention:
                try:
                    self._on_intervention(session)
                except Exception as e:
                    logger.warning(f"Intervention callback error: {e}")

        return session

    async def _process_message(self, session: AgentSession, message: Any) -> None:
        """Process a message from the SDK client.

        Args:
            session: Agent session
            message: Message from SDK
        """
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    # Add text to progress
                    session.add_progress(f"[RESPONSE] {block.text[:200]}")
                    if self._on_progress:
                        try:
                            self._on_progress(session.id, block.text)
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")

                elif isinstance(block, ToolUseBlock):
                    # Log tool use
                    tool_name = block.name
                    tool_input = block.input
                    session.add_progress(f"[TOOL] {tool_name}")

                    if self._on_tool_use:
                        try:
                            self._on_tool_use(session.id, tool_name, tool_input)
                        except Exception as e:
                            logger.warning(f"Tool use callback error: {e}")

                    # Parse for file changes from built-in tools
                    self._detect_file_changes(session, tool_name, tool_input)

                elif isinstance(block, ToolResultBlock):
                    # Check for errors
                    if block.is_error:
                        session.add_progress(f"[TOOL_ERROR] {block.content}")

    def _detect_file_changes(
        self,
        session: AgentSession,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Detect and record file changes from built-in tools.

        Args:
            session: Agent session
            tool_name: Name of the tool
            tool_input: Tool input parameters
        """
        file_path = tool_input.get("file_path")

        if tool_name == "Write" and file_path:
            session.add_file_change(
                FileChange(
                    file_path=file_path,
                    change_type=FileChangeType.CREATE,
                )
            )
        elif tool_name == "Edit" and file_path:
            session.add_file_change(
                FileChange(
                    file_path=file_path,
                    change_type=FileChangeType.MODIFY,
                )
            )

    def spawn_agent(
        self,
        task: str,
        working_dir: Path,
        context: dict[str, Any] | None = None,
        timeout: int | None = None,
        execution_id: int | None = None,
        resume_from: str | None = None,
    ) -> AgentSession:
        """Spawn a new Claude agent synchronously.

        This is a synchronous wrapper around spawn_agent_async that runs
        the agent in a new event loop. For async code, use spawn_agent_async.

        Args:
            task: Task description/prompt for the agent
            working_dir: Working directory for the agent
            context: Optional context information
            timeout: Optional timeout in seconds
            execution_id: Optional link to workflow execution
            resume_from: Optional session ID to resume from

        Returns:
            Created AgentSession
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, create task
            future = asyncio.ensure_future(
                self.spawn_agent_async(
                    task=task,
                    working_dir=working_dir,
                    context=context,
                    timeout=timeout,
                    execution_id=execution_id,
                    resume_from=resume_from,
                )
            )
            return loop.run_until_complete(future)
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(
                self.spawn_agent_async(
                    task=task,
                    working_dir=working_dir,
                    context=context,
                    timeout=timeout,
                    execution_id=execution_id,
                    resume_from=resume_from,
                )
            )

    async def continue_conversation_async(
        self,
        session_id: str,
        message: str,
    ) -> AgentSession:
        """Continue a conversation with an existing session.

        Args:
            session_id: Session ID to continue
            message: Message to send

        Returns:
            Updated AgentSession

        Raises:
            ValueError: If session not found or not resumable
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        sdk_session_id = session.context.get("sdk_session_id")
        if not sdk_session_id:
            raise ValueError(f"Session {session_id} has no SDK session ID for resumption")

        # Resume with the SDK session ID
        return await self.spawn_agent_async(
            task=message,
            working_dir=session.working_dir,
            context=session.context,
            execution_id=session.execution_id,
            resume_from=sdk_session_id,
        )

    async def interrupt_agent_async(self, session_id: str) -> bool:
        """Interrupt a running agent.

        Args:
            session_id: Session ID to interrupt

        Returns:
            True if interrupted successfully
        """
        client = self._active_clients.get(session_id)
        if not client:
            return False

        try:
            await client.interrupt()
            logger.info(f"Interrupted agent {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error interrupting agent {session_id}: {e}")
            return False

    def poll_agent(self, session_id: str) -> AgentStatus:
        """Poll the status of an agent.

        Args:
            session_id: Session identifier

        Returns:
            Current agent status

        Raises:
            ValueError: If session not found
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        return session.status

    def get_session(self, session_id: str) -> AgentSession | None:
        """Get full session details.

        Args:
            session_id: Session identifier

        Returns:
            AgentSession if found
        """
        return self.session_manager.get_session(session_id)

    def send_feedback(self, session_id: str, feedback: str) -> None:
        """Send feedback to an agent session.

        Note: For SDK integration, this stores feedback in the session
        for logging purposes. Use continue_conversation_async for
        interactive feedback.

        Args:
            session_id: Session identifier
            feedback: Feedback message

        Raises:
            ValueError: If session not found
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.add_progress(f"[HUMAN_FEEDBACK] {feedback}")
        self.session_manager.update_session(session)
        logger.info(f"Feedback recorded for session {session_id}: {feedback[:50]}...")

    def terminate_agent(self, session_id: str, reason: str = "Manual termination") -> bool:
        """Terminate a running agent.

        Args:
            session_id: Session identifier
            reason: Reason for termination

        Returns:
            True if agent was terminated
        """
        # Try to interrupt via SDK
        client = self._active_clients.get(session_id)
        if client:
            try:
                asyncio.get_event_loop().run_until_complete(client.interrupt())
            except Exception as e:
                logger.warning(f"Could not interrupt client: {e}")

        # Update session status
        session = self.session_manager.get_session(session_id)
        if session:
            session.status = AgentStatus.CANCELLED
            session.error_message = reason
            session.completed_at = datetime.utcnow()
            self.session_manager.update_session(session)
            logger.info(f"Terminated agent {session_id}: {reason}")
            return True

        return False

    def terminate_all_agents(self, reason: str = "Shutdown") -> int:
        """Terminate all running agents.

        Args:
            reason: Reason for termination

        Returns:
            Number of agents terminated
        """
        count = 0
        for session_id in list(self._active_clients.keys()):
            if self.terminate_agent(session_id, reason):
                count += 1
        return count

    def get_active_agents(self) -> list[AgentSession]:
        """Get all currently active agent sessions."""
        return self.session_manager.get_active_sessions()

    def get_pending_interventions(self) -> list[AgentSession]:
        """Get sessions waiting for human intervention."""
        return self.session_manager.get_pending_interventions()

    def resolve_intervention(
        self,
        session_id: str,
        resolution: str,
        continue_agent: bool = False,
    ) -> None:
        """Resolve a pending intervention request.

        Args:
            session_id: Session identifier
            resolution: Resolution description
            continue_agent: Whether to spawn a new agent to continue

        Raises:
            ValueError: If session not found or doesn't need intervention
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.status != AgentStatus.NEEDS_HUMAN:
            raise ValueError(f"Session {session_id} does not need intervention")

        session.add_progress(f"[INTERVENTION_RESOLVED] {resolution}")

        if continue_agent:
            # Build continuation context
            new_context = session.context.copy()
            new_context["previous_session_id"] = session_id
            new_context["intervention_resolution"] = resolution
            new_context["previous_output"] = session.output[-5000:]

            f"""Continue the previous task after human intervention.

Previous task: {session.task}

Intervention reason: {session.intervention_reason}

Resolution provided: {resolution}

Previous work summary:
{session.output[-2000:]}

Continue from where the previous agent left off, incorporating the resolution.
"""
            # Note: This would need to be run in an async context
            # For now, just record the intent
            session.add_progress(f"[CONTINUATION_PENDING] Resolution: {resolution}")

        # Mark original session as completed
        session.status = AgentStatus.COMPLETED
        session.completed_at = datetime.utcnow()
        self.session_manager.update_session(session)

    def get_stats(self) -> dict[str, Any]:
        """Get integration statistics."""
        session_stats = self.session_manager.get_stats()
        session_stats["active_sdk_clients"] = len(self._active_clients)
        session_stats["sdk_available"] = _SDK_AVAILABLE
        return session_stats


# =============================================================================
# ASYNC WRAPPER FOR COMPATIBILITY
# =============================================================================


class AsyncSDKClaudeIntegration:
    """Async wrapper around SDKClaudeIntegration for compatibility."""

    def __init__(self, integration: SDKClaudeIntegration):
        """Initialize async wrapper.

        Args:
            integration: Sync integration to wrap
        """
        self._integration = integration

    async def spawn_agent(
        self,
        task: str,
        working_dir: Path,
        context: dict[str, Any] | None = None,
        timeout: int | None = None,
        execution_id: int | None = None,
        resume_from: str | None = None,
    ) -> AgentSession:
        """Async spawn_agent."""
        return await self._integration.spawn_agent_async(
            task=task,
            working_dir=working_dir,
            context=context,
            timeout=timeout,
            execution_id=execution_id,
            resume_from=resume_from,
        )

    async def continue_conversation(
        self,
        session_id: str,
        message: str,
    ) -> AgentSession:
        """Async continue_conversation."""
        return await self._integration.continue_conversation_async(session_id, message)

    async def interrupt_agent(self, session_id: str) -> bool:
        """Async interrupt_agent."""
        return await self._integration.interrupt_agent_async(session_id)

    async def poll_agent(self, session_id: str) -> AgentStatus:
        """Async poll_agent."""
        return self._integration.poll_agent(session_id)

    async def get_session(self, session_id: str) -> AgentSession | None:
        """Async get_session."""
        return self._integration.get_session(session_id)

    async def send_feedback(self, session_id: str, feedback: str) -> None:
        """Async send_feedback."""
        self._integration.send_feedback(session_id, feedback)

    async def terminate_agent(self, session_id: str, reason: str = "Manual termination") -> bool:
        """Async terminate_agent."""
        return self._integration.terminate_agent(session_id, reason)

    async def wait_for_completion(
        self,
        session_id: str,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> AgentSession:
        """Wait for an agent to complete.

        Args:
            session_id: Session to wait for
            poll_interval: Seconds between polls
            timeout: Optional timeout in seconds

        Returns:
            Completed session

        Raises:
            TimeoutError: If timeout exceeded
            ValueError: If session not found
        """
        import time

        start = time.time()

        while True:
            session = self._integration.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")

            if session.status in (
                AgentStatus.COMPLETED,
                AgentStatus.FAILED,
                AgentStatus.CANCELLED,
                AgentStatus.NEEDS_HUMAN,
            ):
                return session

            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Timeout waiting for session {session_id}")

            await asyncio.sleep(poll_interval)

    def get_stats(self) -> dict[str, Any]:
        """Get stats."""
        return self._integration.get_stats()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_claude_integration(
    config: SDKAgentConfig | None = None,
    session_manager: AgentSessionManager | None = None,
    db: Any | None = None,
    fallback_to_subprocess: bool = True,
) -> Any:
    """Factory function to create Claude integration.

    Creates SDK integration if available, otherwise falls back to
    subprocess-based integration if configured.

    Args:
        config: Agent configuration
        session_manager: Optional session manager
        db: Optional StateDB for persistence
        fallback_to_subprocess: If True, fall back to subprocess integration

    Returns:
        SDKClaudeIntegration or HOTLClaudeIntegration

    Raises:
        RuntimeError: If SDK not available and fallback disabled
    """
    if _SDK_AVAILABLE:
        return SDKClaudeIntegration(
            config=config,
            session_manager=session_manager,
            db=db,
        )
    elif fallback_to_subprocess:
        from harness.hotl.claude_integration import (
            ClaudeAgentConfig,
            HOTLClaudeIntegration,
        )

        # Convert SDKAgentConfig to ClaudeAgentConfig if needed
        if config:
            subprocess_config = ClaudeAgentConfig(
                default_timeout=config.default_timeout,
                model=config.model,
                skip_permissions=config.permission_mode == PermissionMode.BYPASS_PERMISSIONS,
                env_vars=config.env,
            )
        else:
            subprocess_config = None

        logger.warning("claude-agent-sdk not available, falling back to subprocess integration")
        return HOTLClaudeIntegration(
            config=subprocess_config,
            session_manager=session_manager,
            db=db,
        )
    else:
        raise RuntimeError(
            f"claude-agent-sdk is not installed and fallback disabled.\n"
            f"Install with: pip install claude-agent-sdk\n"
            f"Import error: {_SDK_IMPORT_ERROR}"
        )
