#!/usr/bin/env python3
"""Subagent audit hook script.

This script logs subagent start/stop events with full metadata for
compliance and debugging purposes.

Usage:
    # As a standalone hook script
    python subagent-audit.py --log-file /path/to/audit.jsonl

    # As a module
    from subagent_audit import SubagentAuditScript
    auditor = SubagentAuditScript(log_file="/path/to/audit.jsonl")
    auditor.run()

Environment Variables:
    SUBAGENT_AUDIT_LOG: Path to audit log file
    SUBAGENT_AUDIT_LEVEL: Audit detail level (minimal, normal, detailed, debug)
    SUBAGENT_AUDIT_CALLBACK_URL: Webhook URL for audit events
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

from harness.hooks.audit import (
    AuditEntry,
    AuditHook,
    AuditLevel,
    AuditLogger,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SubagentAuditScript:
    """Standalone subagent audit script.

    Captures and logs all subagent lifecycle events for audit purposes.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        level: str = "normal",
        callback_url: Optional[str] = None,
    ):
        """Initialize the audit script.

        Args:
            log_file: Path to JSON lines audit log
            level: Audit level (minimal, normal, detailed, debug)
            callback_url: Webhook URL for audit events
        """
        self.log_file = Path(log_file) if log_file else None
        self.level = AuditLevel(level.lower())
        self.callback_url = callback_url

        # Create audit logger
        self.audit_logger = AuditLogger(
            log_file=self.log_file,
            log_format="jsonl",
            callback=self._webhook_callback if callback_url else None,
        )

        # Create the hook
        self.hook = AuditHook(
            name="script_subagent_audit",
            audit_logger=self.audit_logger,
            level=self.level,
        )

        # Track active agents
        self._active_agents: dict[str, datetime] = {}

        logger.info(f"Subagent audit initialized (level={level})")
        if self.log_file:
            logger.info(f"  Log file: {self.log_file}")
        if self.callback_url:
            logger.info(f"  Callback URL: {self.callback_url}")

    def _webhook_callback(self, entry: AuditEntry) -> None:
        """Send audit entry to webhook.

        Args:
            entry: Audit entry to send
        """
        if not self.callback_url:
            return

        try:
            import urllib.request

            data = json.dumps(entry.to_dict()).encode()
            req = urllib.request.Request(
                self.callback_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Webhook callback failed: {e}")

    def log_subagent_start(
        self,
        agent_id: str,
        task: str,
        parent_agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        execution_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log a subagent start event.

        Args:
            agent_id: ID of the new subagent
            task: Task description
            parent_agent_id: ID of parent agent if any
            session_id: Session ID
            execution_id: Workflow execution ID
            metadata: Additional metadata
        """
        self._active_agents[agent_id] = datetime.utcnow()

        details = {
            "task": task[:500] if self.level != AuditLevel.MINIMAL else None,
            "execution_id": execution_id,
        }

        if self.level in (AuditLevel.DETAILED, AuditLevel.DEBUG):
            details["full_task"] = task
            details["metadata"] = metadata

        entry = AuditEntry(
            event_type="subagent_start",
            agent_id=agent_id,
            session_id=session_id,
            level=self.level,
            message=f"Subagent started: {task[:100]}",
            details=details,
            parent_agent_id=parent_agent_id,
        )

        self.audit_logger.log(entry)
        logger.info(f"Logged subagent start: {agent_id[:8]}")

    def log_subagent_stop(
        self,
        agent_id: str,
        status: str,
        output: Optional[str] = None,
        error_message: Optional[str] = None,
        file_changes: Optional[list] = None,
        session_id: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log a subagent stop event.

        Args:
            agent_id: ID of the completed subagent
            status: Final status (completed, failed, cancelled, etc.)
            output: Agent output
            error_message: Error message if failed
            file_changes: List of file changes made
            session_id: Session ID
            parent_agent_id: Parent agent ID
            metadata: Additional metadata
        """
        # Calculate duration
        start_time = self._active_agents.pop(agent_id, None)
        duration_ms = None
        if start_time:
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        details = {
            "status": status,
            "error": error_message if status == "failed" else None,
        }

        if self.level != AuditLevel.MINIMAL:
            details["output_length"] = len(output or "")
            details["file_changes_count"] = len(file_changes or [])

        if self.level in (AuditLevel.DETAILED, AuditLevel.DEBUG):
            details["output"] = (output or "")[-5000:]  # Truncate
            details["file_changes"] = file_changes
            details["metadata"] = metadata

        entry = AuditEntry(
            event_type="subagent_stop",
            agent_id=agent_id,
            session_id=session_id,
            level=self.level,
            message=f"Subagent stopped with status: {status}",
            details=details,
            parent_agent_id=parent_agent_id,
            duration_ms=duration_ms,
        )

        self.audit_logger.log(entry)
        logger.info(
            f"Logged subagent stop: {agent_id[:8]} " f"(status={status}, duration={duration_ms}ms)"
        )

    def get_entries(
        self,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get audit entries.

        Args:
            agent_id: Filter by agent ID
            event_type: Filter by event type
            since: Only entries after this time
            limit: Maximum entries to return

        Returns:
            List of audit entry dictionaries
        """
        entries = self.audit_logger.get_entries(
            agent_id=agent_id,
            event_type=event_type,
            since=since,
            limit=limit,
        )
        return [e.to_dict() for e in entries]

    def get_agent_timeline(self, agent_id: str) -> list[dict]:
        """Get chronological timeline for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of entries in chronological order
        """
        entries = self.audit_logger.get_agent_timeline(agent_id)
        return [e.to_dict() for e in entries]

    def get_stats(self) -> dict[str, Any]:
        """Get audit statistics.

        Returns:
            Dict with statistics
        """
        entries = self.audit_logger.get_entries(limit=10000)

        stats = {
            "total_entries": len(entries),
            "active_agents": len(self._active_agents),
            "by_event_type": {},
            "by_status": {},
            "avg_duration_ms": 0,
        }

        durations = []
        for entry in entries:
            etype = entry.event_type
            stats["by_event_type"][etype] = stats["by_event_type"].get(etype, 0) + 1

            if entry.event_type == "subagent_stop":
                status = entry.details.get("status", "unknown") if entry.details else "unknown"
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

                if entry.duration_ms:
                    durations.append(entry.duration_ms)

        if durations:
            stats["avg_duration_ms"] = sum(durations) / len(durations)
            stats["min_duration_ms"] = min(durations)
            stats["max_duration_ms"] = max(durations)

        return stats

    def run(self) -> None:
        """Run the auditor in interactive mode.

        Reads audit events from stdin in JSON format.
        """
        logger.info("Subagent auditor running. Reading events from stdin...")

        for line in sys.stdin:
            try:
                event = json.loads(line.strip())
                event_type = event.get("event_type", "").lower()

                if event_type == "start":
                    self.log_subagent_start(
                        agent_id=event.get("agent_id", "unknown"),
                        task=event.get("task", ""),
                        parent_agent_id=event.get("parent_agent_id"),
                        session_id=event.get("session_id"),
                        execution_id=event.get("execution_id"),
                        metadata=event.get("metadata"),
                    )
                elif event_type == "stop":
                    self.log_subagent_stop(
                        agent_id=event.get("agent_id", "unknown"),
                        status=event.get("status", "unknown"),
                        output=event.get("output"),
                        error_message=event.get("error_message"),
                        file_changes=event.get("file_changes"),
                        session_id=event.get("session_id"),
                        parent_agent_id=event.get("parent_agent_id"),
                        metadata=event.get("metadata"),
                    )
                else:
                    logger.warning(f"Unknown event type: {event_type}")

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON input: {e}")
            except KeyboardInterrupt:
                logger.info("Shutting down subagent auditor")
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")

        self.audit_logger.close()


def main():
    """Main entry point for the subagent audit script."""
    parser = argparse.ArgumentParser(
        description="Audit subagent start/stop events",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("SUBAGENT_AUDIT_LOG", "subagent_audit.jsonl"),
        help="Path to the audit log file",
    )
    parser.add_argument(
        "--level",
        choices=["minimal", "normal", "detailed", "debug"],
        default=os.environ.get("SUBAGENT_AUDIT_LEVEL", "normal"),
        help="Audit detail level",
    )
    parser.add_argument(
        "--callback-url",
        default=os.environ.get("SUBAGENT_AUDIT_CALLBACK_URL"),
        help="Webhook URL for audit events",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics and exit",
    )
    parser.add_argument(
        "--timeline",
        metavar="AGENT_ID",
        help="Print timeline for an agent and exit",
    )

    args = parser.parse_args()

    auditor = SubagentAuditScript(
        log_file=args.log_file,
        level=args.level,
        callback_url=args.callback_url,
    )

    if args.stats:
        stats = auditor.get_stats()
        print(json.dumps(stats, indent=2))
    elif args.timeline:
        timeline = auditor.get_agent_timeline(args.timeline)
        print(json.dumps(timeline, indent=2))
    else:
        auditor.run()


if __name__ == "__main__":
    main()
