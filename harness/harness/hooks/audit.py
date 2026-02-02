"""Audit hook for logging all subagent activity.

This module provides comprehensive audit logging for agent operations,
enabling:

- Compliance tracking
- Security auditing
- Performance monitoring
- Debugging and troubleshooting
- Activity reconstruction

The AuditHook captures all significant agent events with full context.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from harness.hooks.base import (
    HookContext,
    HookPriority,
    HookResult,
    PostToolUseHook,
    PreToolUseHook,
    SubagentStartHook,
    SubagentStopHook,
)

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Level of detail for audit logging."""

    MINIMAL = "minimal"  # Start/stop only
    NORMAL = "normal"  # + tool use summary
    DETAILED = "detailed"  # + full tool input/output
    DEBUG = "debug"  # + internal state


@dataclass
class AuditEntry:
    """A single audit log entry.

    Attributes:
        event_type: Type of event (start, stop, tool_use, etc.)
        agent_id: ID of the agent
        session_id: Session ID
        timestamp: When the event occurred
        level: Audit level
        message: Human-readable message
        details: Detailed event information
        metadata: Additional metadata
        parent_agent_id: Parent agent for subagents
        duration_ms: Duration in milliseconds (for stop events)
    """

    event_type: str
    agent_id: str
    session_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: AuditLevel = AuditLevel.NORMAL
    message: str = ""
    details: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_agent_id: str | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "details": self.details,
            "metadata": self.metadata,
            "parent_agent_id": self.parent_agent_id,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            event_type=data["event_type"],
            agent_id=data["agent_id"],
            session_id=data.get("session_id"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.utcnow(),
            level=AuditLevel(data.get("level", "normal")),
            message=data.get("message", ""),
            details=data.get("details"),
            metadata=data.get("metadata", {}),
            parent_agent_id=data.get("parent_agent_id"),
            duration_ms=data.get("duration_ms"),
        )

    def to_log_line(self) -> str:
        """Format as a single log line."""
        parts = [
            self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"[{self.event_type.upper()}]",
            f"agent={self.agent_id[:8]}",
        ]

        if self.session_id:
            parts.append(f"session={self.session_id[:8]}")
        if self.parent_agent_id:
            parts.append(f"parent={self.parent_agent_id[:8]}")
        if self.duration_ms is not None:
            parts.append(f"duration={self.duration_ms}ms")

        parts.append(self.message)

        return " ".join(parts)


class AuditLogger:
    """Logger for audit entries.

    Supports multiple output destinations:
    - File logging (JSON lines or structured text)
    - Custom callback
    - In-memory storage
    """

    def __init__(
        self,
        log_file: Path | None = None,
        log_format: str = "jsonl",  # jsonl, text
        callback: Callable[[AuditEntry], None] | None = None,
        max_memory_entries: int = 10000,
        rotate_size_mb: int = 100,
    ):
        """Initialize audit logger.

        Args:
            log_file: Path for audit log file
            log_format: Output format (jsonl or text)
            callback: Optional callback for each entry
            max_memory_entries: Maximum entries to keep in memory
            rotate_size_mb: Rotate log when it exceeds this size
        """
        self.log_file = log_file
        self.log_format = log_format
        self.callback = callback
        self.max_memory_entries = max_memory_entries
        self.rotate_size_mb = rotate_size_mb

        self._entries: list[AuditEntry] = []
        self._file_handle: Any | None = None

        if log_file:
            self._open_log_file()

    def _open_log_file(self) -> None:
        """Open the log file for writing."""
        if not self.log_file:
            return

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(self.log_file, "a", buffering=1)  # Line buffered

    def _check_rotation(self) -> None:
        """Check if log file needs rotation."""
        if not self.log_file or not self._file_handle:
            return

        try:
            size_mb = self.log_file.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotate_size_mb:
                self._rotate_log()
        except OSError:
            pass

    def _rotate_log(self) -> None:
        """Rotate the log file."""
        if not self.log_file or not self._file_handle:
            return

        self._file_handle.close()

        # Rename old log
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{self.log_file.stem}.{timestamp}{self.log_file.suffix}"
        rotated_path = self.log_file.parent / rotated_name

        try:
            self.log_file.rename(rotated_path)
            logger.info(f"Rotated audit log to {rotated_path}")
        except OSError as e:
            logger.warning(f"Failed to rotate log: {e}")

        self._open_log_file()

    def log(self, entry: AuditEntry) -> None:
        """Log an audit entry.

        Args:
            entry: The entry to log
        """
        # Store in memory
        self._entries.append(entry)
        if len(self._entries) > self.max_memory_entries:
            self._entries = self._entries[-self.max_memory_entries :]

        # Write to file
        if self._file_handle:
            try:
                if self.log_format == "jsonl":
                    self._file_handle.write(json.dumps(entry.to_dict()) + "\n")
                else:
                    self._file_handle.write(entry.to_log_line() + "\n")
                self._check_rotation()
            except OSError as e:
                logger.warning(f"Failed to write audit log: {e}")

        # Call callback
        if self.callback:
            try:
                self.callback(entry)
            except Exception as e:
                logger.warning(f"Audit callback error: {e}")

    def get_entries(
        self,
        agent_id: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit entries with optional filtering.

        Args:
            agent_id: Filter by agent ID
            event_type: Filter by event type
            since: Only entries after this time
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        entries = self._entries

        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries[-limit:]

    def get_agent_timeline(self, agent_id: str) -> list[AuditEntry]:
        """Get chronological timeline for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of entries in chronological order
        """
        entries = [e for e in self._entries if e.agent_id == agent_id]
        return sorted(entries, key=lambda e: e.timestamp)

    def close(self) -> None:
        """Close the logger and file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


class AuditHook(PreToolUseHook, PostToolUseHook, SubagentStartHook, SubagentStopHook):
    """Comprehensive audit hook that logs all agent activity.

    This hook implements all hook interfaces to capture:
    - Subagent start events
    - Subagent stop events with results
    - Tool use (pre and post)

    Usage:
        logger = AuditLogger(log_file=Path("audit.jsonl"))
        hook = AuditHook(audit_logger=logger)
        manager.register(hook)
    """

    def __init__(
        self,
        name: str = "audit",
        audit_logger: AuditLogger | None = None,
        level: AuditLevel = AuditLevel.NORMAL,
        log_file: Path | None = None,
        include_tool_input: bool = True,
        include_tool_output: bool = False,
        redact_patterns: list[str] | None = None,
    ):
        """Initialize audit hook.

        Args:
            name: Hook name
            audit_logger: Logger instance (created if not provided)
            level: Audit detail level
            log_file: Path for audit log (if no logger provided)
            include_tool_input: Include tool input in logs
            include_tool_output: Include tool output in logs
            redact_patterns: Patterns to redact from logs
        """
        # Initialize as PreToolUseHook (first parent)
        super().__init__(name=name, priority=HookPriority.AUDIT)

        self.level = level
        self.include_tool_input = include_tool_input
        self.include_tool_output = include_tool_output
        self.redact_patterns = redact_patterns or [
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "auth",
            "credential",
        ]

        # Create logger if not provided
        if audit_logger:
            self.audit_logger = audit_logger
        else:
            self.audit_logger = AuditLogger(log_file=log_file)

        # Track agent start times for duration calculation
        self._start_times: dict[str, datetime] = {}

    def _redact_sensitive(self, data: Any) -> Any:
        """Redact sensitive information from data.

        Args:
            data: Data to redact

        Returns:
            Redacted data
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                key_lower = key.lower()
                if any(pattern in key_lower for pattern in self.redact_patterns):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = self._redact_sensitive(value)
            return result
        elif isinstance(data, list):
            return [self._redact_sensitive(item) for item in data]
        elif isinstance(data, str):
            # Could add more sophisticated string redaction
            return data
        else:
            return data

    async def on_subagent_start(
        self,
        agent_id: str,
        task: str,
        context: HookContext,
    ) -> tuple[HookResult, dict[str, Any] | None]:
        """Log subagent start event.

        Args:
            agent_id: ID of new subagent
            task: Task description
            context: Hook context

        Returns:
            (CONTINUE, None)
        """
        self._start_times[agent_id] = datetime.utcnow()

        details = {
            "task": task[:500] if self.level != AuditLevel.MINIMAL else None,
            "execution_id": context.execution_id,
        }

        if self.level in (AuditLevel.DETAILED, AuditLevel.DEBUG):
            details["full_task"] = task
            details["context_metadata"] = self._redact_sensitive(context.metadata)

        entry = AuditEntry(
            event_type="subagent_start",
            agent_id=agent_id,
            session_id=context.session_id,
            level=self.level,
            message=f"Subagent started: {task[:100]}",
            details=details,
            parent_agent_id=context.parent_agent_id,
        )

        self.audit_logger.log(entry)
        return HookResult.CONTINUE, None

    async def on_subagent_stop(
        self,
        agent_id: str,
        result: dict[str, Any],
        context: HookContext,
    ) -> None:
        """Log subagent stop event.

        Args:
            agent_id: ID of completed subagent
            result: Result dict
            context: Hook context
        """
        # Calculate duration
        start_time = self._start_times.pop(agent_id, None)
        duration_ms = None
        if start_time:
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        status = result.get("status", "unknown")
        details = {
            "status": status,
            "error": result.get("error_message") if status == "failed" else None,
        }

        if self.level != AuditLevel.MINIMAL:
            details["output_length"] = len(result.get("output", ""))
            details["file_changes_count"] = len(result.get("file_changes", []))

        if self.level in (AuditLevel.DETAILED, AuditLevel.DEBUG):
            details["result"] = self._redact_sensitive(result)

        entry = AuditEntry(
            event_type="subagent_stop",
            agent_id=agent_id,
            session_id=context.session_id,
            level=self.level,
            message=f"Subagent stopped with status: {status}",
            details=details,
            parent_agent_id=context.parent_agent_id,
            duration_ms=duration_ms,
        )

        self.audit_logger.log(entry)

    async def on_pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: HookContext,
    ) -> tuple[HookResult, dict[str, Any] | None]:
        """Log tool use event (pre-execution).

        Args:
            tool_name: Tool name
            tool_input: Tool input
            context: Hook context

        Returns:
            (CONTINUE, None)
        """
        if self.level == AuditLevel.MINIMAL:
            return HookResult.CONTINUE, None

        details = {"tool_name": tool_name}

        if self.include_tool_input:
            details["input"] = self._redact_sensitive(tool_input)

        entry = AuditEntry(
            event_type="tool_use",
            agent_id=context.agent_id,
            session_id=context.session_id,
            level=self.level,
            message=f"Tool invoked: {tool_name}",
            details=details,
        )

        self.audit_logger.log(entry)
        return HookResult.CONTINUE, None

    async def on_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
        context: HookContext,
    ) -> None:
        """Log tool completion event.

        Args:
            tool_name: Tool name
            tool_input: Tool input
            tool_output: Tool output
            context: Hook context
        """
        if self.level not in (AuditLevel.DETAILED, AuditLevel.DEBUG):
            return

        if not self.include_tool_output:
            return

        details = {
            "tool_name": tool_name,
            "success": not tool_output.get("is_error", False),
        }

        # Truncate large outputs
        output_str = str(tool_output)
        if len(output_str) > 5000:
            details["output_truncated"] = True
            details["output"] = self._redact_sensitive(
                {k: str(v)[:500] for k, v in tool_output.items()}
            )
        else:
            details["output"] = self._redact_sensitive(tool_output)

        entry = AuditEntry(
            event_type="tool_complete",
            agent_id=context.agent_id,
            session_id=context.session_id,
            level=self.level,
            message=f"Tool completed: {tool_name}",
            details=details,
        )

        self.audit_logger.log(entry)

    def get_audit_entries(
        self,
        agent_id: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit entries from the logger.

        Args:
            agent_id: Filter by agent ID
            event_type: Filter by event type
            since: Only entries after this time
            limit: Maximum entries

        Returns:
            List of audit entries
        """
        return self.audit_logger.get_entries(
            agent_id=agent_id,
            event_type=event_type,
            since=since,
            limit=limit,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get audit statistics.

        Returns:
            Dict with statistics
        """
        stats = super().get_stats()
        stats["level"] = self.level.value
        stats["entries_in_memory"] = len(self.audit_logger._entries)
        stats["active_agents"] = len(self._start_times)
        return stats
