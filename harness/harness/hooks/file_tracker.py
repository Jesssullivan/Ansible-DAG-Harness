"""File change tracker hook for monitoring agent file modifications.

This module provides hooks that track all file changes made by agents,
enabling:

- Audit trail of all modifications
- Rollback capability
- Change diff generation
- File operation statistics

The FileChangeTrackerHook intercepts Write, Edit, and other file
operations to maintain a complete record of changes.
"""

import hashlib
import json
import logging
import shutil
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
)

logger = logging.getLogger(__name__)


class FileChangeType(Enum):
    """Type of file change."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    PERMISSION = "permission"


@dataclass
class TrackedFile:
    """Information about a tracked file.

    Attributes:
        path: Absolute path to the file
        original_hash: SHA-256 hash before changes
        current_hash: Current SHA-256 hash
        original_content: Original content (if stored)
        exists: Whether file exists
        size: File size in bytes
        permissions: File permissions mode
    """

    path: str
    original_hash: str | None = None
    current_hash: str | None = None
    original_content: str | None = None
    exists: bool = True
    size: int = 0
    permissions: int = 0o644

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "original_hash": self.original_hash,
            "current_hash": self.current_hash,
            "has_original_content": self.original_content is not None,
            "exists": self.exists,
            "size": self.size,
            "permissions": oct(self.permissions),
        }


@dataclass
class FileChange:
    """Record of a file change.

    Attributes:
        path: Path to the changed file
        change_type: Type of change
        agent_id: ID of agent that made the change
        session_id: Session ID when change occurred
        timestamp: When the change occurred
        tool_name: Tool that made the change
        old_path: Original path (for renames)
        diff: Unified diff if available
        original_hash: Hash before change
        new_hash: Hash after change
        metadata: Additional metadata
    """

    path: str
    change_type: FileChangeType
    agent_id: str
    session_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tool_name: str | None = None
    old_path: str | None = None
    diff: str | None = None
    original_hash: str | None = None
    new_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "change_type": self.change_type.value,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "old_path": self.old_path,
            "diff": self.diff,
            "original_hash": self.original_hash,
            "new_hash": self.new_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileChange":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            change_type=FileChangeType(data["change_type"]),
            agent_id=data["agent_id"],
            session_id=data.get("session_id"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.utcnow(),
            tool_name=data.get("tool_name"),
            old_path=data.get("old_path"),
            diff=data.get("diff"),
            original_hash=data.get("original_hash"),
            new_hash=data.get("new_hash"),
            metadata=data.get("metadata", {}),
        )


class FileChangeTrackerHook(PreToolUseHook, PostToolUseHook):
    """Hook that tracks all file changes made by agents.

    This hook implements both PreToolUse (to capture original state)
    and PostToolUse (to record the change) interfaces.

    Features:
    - Tracks file hash before/after changes
    - Optionally stores original content for rollback
    - Generates unified diffs for text files
    - Supports backup and restore

    Usage:
        tracker = FileChangeTrackerHook(
            backup_dir=Path("/tmp/backups"),
            store_content=True,
        )
        manager.register(tracker)
    """

    # Tools that modify files
    FILE_TOOLS = {
        "Write": FileChangeType.CREATE,
        "Edit": FileChangeType.MODIFY,
        "Bash": None,  # Needs command analysis
        "mcp__filesystem__write_file": FileChangeType.CREATE,
        "mcp__filesystem__edit_file": FileChangeType.MODIFY,
        "mcp__filesystem__move_file": FileChangeType.RENAME,
        "mcp__filesystem__create_directory": FileChangeType.CREATE,
    }

    def __init__(
        self,
        name: str = "file_change_tracker",
        priority: HookPriority = HookPriority.HIGH,
        backup_dir: Path | None = None,
        store_content: bool = False,
        max_content_size: int = 1024 * 1024,  # 1MB
        generate_diffs: bool = True,
        log_file: Path | None = None,
    ):
        """Initialize the file change tracker.

        Args:
            name: Hook name
            priority: Hook priority
            backup_dir: Directory for file backups
            store_content: Whether to store file content in memory
            max_content_size: Max size for content storage
            generate_diffs: Whether to generate diffs
            log_file: Path to write change log
        """
        # Initialize as PreToolUseHook (first parent)
        super().__init__(name=name, priority=priority)

        self.backup_dir = backup_dir
        self.store_content = store_content
        self.max_content_size = max_content_size
        self.generate_diffs = generate_diffs
        self.log_file = log_file

        # State tracking
        self._tracked_files: dict[str, TrackedFile] = {}
        self._changes: list[FileChange] = []
        self._pending_operations: dict[str, dict[str, Any]] = {}

        # Create backup directory if specified
        if backup_dir:
            backup_dir.mkdir(parents=True, exist_ok=True)

    async def on_pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: HookContext,
    ) -> tuple[HookResult, dict[str, Any] | None]:
        """Capture file state before modification.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            context: Hook context

        Returns:
            (CONTINUE, None) - never modifies input
        """
        if tool_name not in self.FILE_TOOLS:
            return HookResult.CONTINUE, None

        # Extract file path from tool input
        file_path = self._extract_file_path(tool_name, tool_input)
        if not file_path:
            return HookResult.CONTINUE, None

        # Track original state
        await self._track_file(file_path, context)

        # Store pending operation for post-hook
        op_id = f"{context.agent_id}:{tool_name}:{file_path}"
        self._pending_operations[op_id] = {
            "file_path": file_path,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "context": context,
        }

        return HookResult.CONTINUE, None

    async def on_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
        context: HookContext,
    ) -> None:
        """Record file change after tool execution.

        Args:
            tool_name: Name of the tool that ran
            tool_input: Tool input parameters
            tool_output: Tool output
            context: Hook context
        """
        if tool_name not in self.FILE_TOOLS:
            return

        file_path = self._extract_file_path(tool_name, tool_input)
        if not file_path:
            return

        # Check if operation succeeded
        if tool_output.get("is_error"):
            return

        # Determine change type
        change_type = self._determine_change_type(tool_name, tool_input, file_path)

        # Record the change
        await self._record_change(
            file_path=file_path,
            change_type=change_type,
            tool_name=tool_name,
            tool_input=tool_input,
            context=context,
        )

        # Clean up pending operation
        op_id = f"{context.agent_id}:{tool_name}:{file_path}"
        self._pending_operations.pop(op_id, None)

    def _extract_file_path(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> str | None:
        """Extract file path from tool input.

        Args:
            tool_name: Tool name
            tool_input: Tool input

        Returns:
            File path if found
        """
        # Common parameter names
        for key in ["file_path", "path", "filename", "source", "destination"]:
            if key in tool_input:
                return str(tool_input[key])

        # For Bash, try to parse command
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            # Very basic parsing - would need more sophisticated analysis
            for pattern in ["touch ", "rm ", "mv ", "cp "]:
                if pattern in command:
                    parts = command.split(pattern)
                    if len(parts) > 1:
                        return parts[1].split()[0].strip("\"'")

        return None

    def _determine_change_type(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        file_path: str,
    ) -> FileChangeType:
        """Determine the type of file change.

        Args:
            tool_name: Tool name
            tool_input: Tool input
            file_path: Path to file

        Returns:
            FileChangeType
        """
        # Check if file existed before
        tracked = self._tracked_files.get(file_path)

        if tool_name in self.FILE_TOOLS:
            default_type = self.FILE_TOOLS[tool_name]
            if default_type:
                # Adjust CREATE to MODIFY if file existed
                if default_type == FileChangeType.CREATE:
                    if tracked and tracked.exists:
                        return FileChangeType.MODIFY
                return default_type

        # Default to MODIFY
        return FileChangeType.MODIFY

    async def _track_file(
        self,
        file_path: str,
        context: HookContext,
    ) -> None:
        """Track a file's original state.

        Args:
            file_path: Path to file
            context: Hook context
        """
        path = Path(file_path)

        if file_path in self._tracked_files:
            return  # Already tracked

        tracked = TrackedFile(path=file_path)

        if path.exists():
            tracked.exists = True
            tracked.size = path.stat().st_size
            tracked.permissions = path.stat().st_mode

            # Calculate hash
            tracked.original_hash = self._hash_file(path)

            # Store content if enabled and small enough
            if self.store_content and tracked.size <= self.max_content_size:
                try:
                    tracked.original_content = path.read_text()
                except (OSError, UnicodeDecodeError):
                    pass

            # Create backup if enabled
            if self.backup_dir:
                await self._create_backup(path, context)
        else:
            tracked.exists = False

        self._tracked_files[file_path] = tracked
        logger.debug(f"Tracking file: {file_path} (exists={tracked.exists})")

    async def _record_change(
        self,
        file_path: str,
        change_type: FileChangeType,
        tool_name: str,
        tool_input: dict[str, Any],
        context: HookContext,
    ) -> None:
        """Record a file change.

        Args:
            file_path: Path to changed file
            change_type: Type of change
            tool_name: Tool that made the change
            tool_input: Tool input
            context: Hook context
        """
        path = Path(file_path)
        tracked = self._tracked_files.get(file_path)

        # Calculate new hash
        new_hash = None
        if path.exists():
            new_hash = self._hash_file(path)

        # Generate diff if enabled
        diff = None
        if self.generate_diffs and tracked and tracked.original_content:
            try:
                new_content = path.read_text() if path.exists() else ""
                diff = self._generate_diff(tracked.original_content, new_content, file_path)
            except (OSError, UnicodeDecodeError):
                pass

        # Create change record
        change = FileChange(
            path=file_path,
            change_type=change_type,
            agent_id=context.agent_id,
            session_id=context.session_id,
            tool_name=tool_name,
            original_hash=tracked.original_hash if tracked else None,
            new_hash=new_hash,
            diff=diff,
            metadata={
                "tool_input_keys": list(tool_input.keys()),
            },
        )

        self._changes.append(change)
        logger.info(
            f"Recorded {change_type.value} change: {file_path} by agent {context.agent_id[:8]}"
        )

        # Write to log file if enabled
        if self.log_file:
            await self._write_to_log(change)

        # Update tracked file
        if tracked:
            tracked.current_hash = new_hash
            tracked.exists = path.exists()

    async def _create_backup(
        self,
        file_path: Path,
        context: HookContext,
    ) -> Path | None:
        """Create a backup of a file.

        Args:
            file_path: Path to file to backup
            context: Hook context

        Returns:
            Path to backup file
        """
        if not self.backup_dir or not file_path.exists():
            return None

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.{context.agent_id[:8]}"
        backup_path = self.backup_dir / backup_name

        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
            return backup_path
        except OSError as e:
            logger.warning(f"Failed to create backup: {e}")
            return None

    def _hash_file(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of hash
        """
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            return ""

    def _generate_diff(
        self,
        original: str,
        new: str,
        file_path: str,
    ) -> str:
        """Generate unified diff between original and new content.

        Args:
            original: Original content
            new: New content
            file_path: File path for diff header

        Returns:
            Unified diff string
        """
        import difflib

        original_lines = original.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )

        return "".join(diff)

    async def _write_to_log(self, change: FileChange) -> None:
        """Write change to log file.

        Args:
            change: Change to log
        """
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(change.to_dict()) + "\n")
        except OSError as e:
            logger.warning(f"Failed to write to change log: {e}")

    def get_changes(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        change_type: FileChangeType | None = None,
        limit: int = 100,
    ) -> list[FileChange]:
        """Get recorded file changes.

        Args:
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            change_type: Filter by change type
            limit: Maximum results

        Returns:
            List of file changes
        """
        changes = self._changes

        if agent_id:
            changes = [c for c in changes if c.agent_id == agent_id]
        if session_id:
            changes = [c for c in changes if c.session_id == session_id]
        if change_type:
            changes = [c for c in changes if c.change_type == change_type]

        return changes[-limit:]

    def get_tracked_files(self) -> dict[str, TrackedFile]:
        """Get all tracked files.

        Returns:
            Dict mapping path to TrackedFile
        """
        return self._tracked_files.copy()

    def get_changes_for_file(self, file_path: str) -> list[FileChange]:
        """Get all changes for a specific file.

        Args:
            file_path: Path to file

        Returns:
            List of changes for that file
        """
        return [c for c in self._changes if c.path == file_path]

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics.

        Returns:
            Dict with statistics
        """
        change_counts = {}
        for change in self._changes:
            change_type = change.change_type.value
            change_counts[change_type] = change_counts.get(change_type, 0) + 1

        return {
            "total_tracked_files": len(self._tracked_files),
            "total_changes": len(self._changes),
            "changes_by_type": change_counts,
            "pending_operations": len(self._pending_operations),
            "backup_enabled": self.backup_dir is not None,
            "store_content_enabled": self.store_content,
        }

    def restore_file(self, file_path: str) -> bool:
        """Restore a file to its original content.

        Args:
            file_path: Path to file to restore

        Returns:
            True if restored successfully
        """
        tracked = self._tracked_files.get(file_path)
        if not tracked or not tracked.original_content:
            logger.warning(f"No original content for {file_path}")
            return False

        try:
            path = Path(file_path)
            if tracked.exists:
                path.write_text(tracked.original_content)
            else:
                # File didn't exist originally, delete it
                if path.exists():
                    path.unlink()
            logger.info(f"Restored file: {file_path}")
            return True
        except OSError as e:
            logger.error(f"Failed to restore {file_path}: {e}")
            return False

    def clear(self) -> None:
        """Clear all tracking state."""
        self._tracked_files.clear()
        self._changes.clear()
        self._pending_operations.clear()
        logger.debug("Cleared file change tracker state")
