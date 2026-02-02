#!/usr/bin/env python3
"""File change tracker hook script.

This script tracks all file modifications made by agents and logs them
for audit and rollback purposes.

Usage:
    # As a standalone hook script
    python file-change-tracker.py --log-file /path/to/changes.jsonl

    # As a module
    from file_change_tracker import FileChangeTrackerScript
    tracker = FileChangeTrackerScript(log_file="/path/to/changes.jsonl")
    tracker.run()

Environment Variables:
    FILE_TRACKER_LOG: Path to change log file
    FILE_TRACKER_BACKUP_DIR: Directory for file backups
    FILE_TRACKER_STORE_CONTENT: Store original content (true/false)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.hooks.file_tracker import (
    FileChange,
    FileChangeTrackerHook,
    FileChangeType,
)
from harness.hooks.base import HookContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FileChangeTrackerScript:
    """Standalone file change tracker script.

    Can be used as a hook script or run standalone to track file changes.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        backup_dir: Optional[str] = None,
        store_content: bool = False,
    ):
        """Initialize the tracker script.

        Args:
            log_file: Path to JSON lines log file
            backup_dir: Directory for file backups
            store_content: Whether to store original file content
        """
        self.log_file = Path(log_file) if log_file else None
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.store_content = store_content

        # Create the hook
        self.hook = FileChangeTrackerHook(
            name="script_file_tracker",
            backup_dir=self.backup_dir,
            store_content=store_content,
            log_file=self.log_file,
        )

        logger.info(f"File change tracker initialized")
        if self.log_file:
            logger.info(f"  Log file: {self.log_file}")
        if self.backup_dir:
            logger.info(f"  Backup dir: {self.backup_dir}")

    def record_change(
        self,
        file_path: str,
        change_type: str,
        agent_id: str,
        session_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a file change.

        Args:
            file_path: Path to the changed file
            change_type: Type of change (create, modify, delete, rename)
            agent_id: ID of the agent making the change
            session_id: Optional session ID
            tool_name: Tool that made the change
            metadata: Additional metadata
        """
        try:
            ctype = FileChangeType(change_type.lower())
        except ValueError:
            ctype = FileChangeType.MODIFY

        change = FileChange(
            path=file_path,
            change_type=ctype,
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            metadata=metadata or {},
        )

        # Write to log
        if self.log_file:
            self._write_to_log(change)

        logger.info(f"Recorded {change_type} change: {file_path} by {agent_id[:8]}")

    def _write_to_log(self, change: FileChange) -> None:
        """Write change to log file.

        Args:
            change: FileChange to log
        """
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(change.to_dict()) + "\n")
        except OSError as e:
            logger.error(f"Failed to write to log: {e}")

    def get_changes(
        self,
        agent_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[dict]:
        """Get recorded changes from log file.

        Args:
            agent_id: Filter by agent ID
            since: Only changes after this time

        Returns:
            List of change dictionaries
        """
        if not self.log_file or not self.log_file.exists():
            return []

        changes = []
        try:
            with open(self.log_file) as f:
                for line in f:
                    try:
                        change = json.loads(line)
                        if agent_id and change.get("agent_id") != agent_id:
                            continue
                        if since:
                            change_time = datetime.fromisoformat(
                                change.get("timestamp", "")
                            )
                            if change_time < since:
                                continue
                        changes.append(change)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except OSError as e:
            logger.error(f"Failed to read log: {e}")

        return changes

    def get_stats(self) -> dict[str, Any]:
        """Get change statistics.

        Returns:
            Dict with statistics
        """
        changes = self.get_changes()

        stats = {
            "total_changes": len(changes),
            "by_type": {},
            "by_agent": {},
        }

        for change in changes:
            ctype = change.get("change_type", "unknown")
            agent = change.get("agent_id", "unknown")

            stats["by_type"][ctype] = stats["by_type"].get(ctype, 0) + 1
            stats["by_agent"][agent] = stats["by_agent"].get(agent, 0) + 1

        return stats

    def run(self) -> None:
        """Run the tracker in interactive mode.

        Reads change events from stdin in JSON format.
        """
        logger.info("File change tracker running. Reading events from stdin...")

        for line in sys.stdin:
            try:
                event = json.loads(line.strip())

                self.record_change(
                    file_path=event.get("file_path", ""),
                    change_type=event.get("change_type", "modify"),
                    agent_id=event.get("agent_id", "unknown"),
                    session_id=event.get("session_id"),
                    tool_name=event.get("tool_name"),
                    metadata=event.get("metadata"),
                )

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON input: {e}")
            except KeyboardInterrupt:
                logger.info("Shutting down file change tracker")
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")


def main():
    """Main entry point for the file change tracker script."""
    parser = argparse.ArgumentParser(
        description="Track file changes made by agents",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("FILE_TRACKER_LOG", "file_changes.jsonl"),
        help="Path to the change log file",
    )
    parser.add_argument(
        "--backup-dir",
        default=os.environ.get("FILE_TRACKER_BACKUP_DIR"),
        help="Directory for file backups",
    )
    parser.add_argument(
        "--store-content",
        action="store_true",
        default=os.environ.get("FILE_TRACKER_STORE_CONTENT", "").lower() == "true",
        help="Store original file content",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics and exit",
    )

    args = parser.parse_args()

    tracker = FileChangeTrackerScript(
        log_file=args.log_file,
        backup_dir=args.backup_dir,
        store_content=args.store_content,
    )

    if args.stats:
        stats = tracker.get_stats()
        print(json.dumps(stats, indent=2))
    else:
        tracker.run()


if __name__ == "__main__":
    main()
