#!/usr/bin/env python3
"""Claude Code PreToolUse hook for tool-level rate limiting.

This hook implements a token bucket algorithm to rate limit tool invocations,
preventing API overuse and ensuring sustainable operation.

Deployed by: harness init

Exit codes:
  0 - Allow the tool invocation
  2 - Block the tool invocation (rate limit exceeded)
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Configuration: per-tool rate limits
TOOL_LIMITS = {
    # Expensive operations - spawn subagents or make network calls
    "Task": {"per_minute": 5, "per_hour": 50},
    "WebFetch": {"per_minute": 10, "per_hour": 100},
    "WebSearch": {"per_minute": 5, "per_hour": 50},

    # File operations - moderate limits
    "Edit": {"per_minute": 30, "per_hour": 500},
    "Write": {"per_minute": 20, "per_hour": 200},
    "Read": {"per_minute": 60, "per_hour": 1000},

    # Shell operations
    "Bash": {"per_minute": 40, "per_hour": 600},

    # Default for unspecified tools
    "default": {"per_minute": 60, "per_hour": 1000},
}

# Database path - use configured path or fallback to claude directory
DB_PATH = Path(
    os.environ.get(
        "HARNESS_RATE_LIMIT_DB",
        os.path.expanduser("~/.claude/rate_limits.db")
    )
)


def get_db() -> sqlite3.Connection:
    """Get database connection, creating schema if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_invocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            session_id TEXT,
            timestamp REAL NOT NULL,
            allowed BOOLEAN DEFAULT TRUE
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_tool_ts
        ON tool_invocations(tool_name, timestamp)
    """)
    conn.commit()
    return conn


def count_recent(conn: sqlite3.Connection, tool_name: str, seconds: int) -> int:
    """Count recent allowed invocations for a tool."""
    cutoff = time.time() - seconds
    result = conn.execute(
        """SELECT COUNT(*) as c FROM tool_invocations
           WHERE tool_name = ? AND timestamp > ? AND allowed = TRUE""",
        (tool_name, cutoff)
    ).fetchone()
    return result["c"] if result else 0


def get_limits(tool_name: str) -> dict:
    """Get rate limits for a tool, checking prefixes for MCP tools."""
    if tool_name in TOOL_LIMITS:
        return TOOL_LIMITS[tool_name]

    for prefix, limits in TOOL_LIMITS.items():
        if prefix.endswith("__") and tool_name.startswith(prefix):
            return limits

    return TOOL_LIMITS["default"]


def check_rate_limit(tool_name: str) -> tuple[bool, str]:
    """Check if a tool invocation is allowed."""
    limits = get_limits(tool_name)
    conn = get_db()

    try:
        minute_count = count_recent(conn, tool_name, 60)
        hour_count = count_recent(conn, tool_name, 3600)

        allowed = True
        reason = ""

        if minute_count >= limits["per_minute"]:
            allowed = False
            reason = f"Rate limit exceeded: {tool_name} ({minute_count}/{limits['per_minute']} per minute). Wait ~{60 - (time.time() % 60):.0f}s."
        elif hour_count >= limits["per_hour"]:
            allowed = False
            reason = f"Rate limit exceeded: {tool_name} ({hour_count}/{limits['per_hour']} per hour)."

        session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")
        conn.execute(
            """INSERT INTO tool_invocations (tool_name, session_id, timestamp, allowed)
               VALUES (?, ?, ?, ?)""",
            (tool_name, session_id, time.time(), allowed)
        )
        conn.commit()

        return allowed, reason

    finally:
        conn.close()


def cleanup_old_records(days: int = 7) -> int:
    """Remove records older than specified days."""
    conn = get_db()
    try:
        cutoff = time.time() - (days * 24 * 3600)
        cursor = conn.execute(
            "DELETE FROM tool_invocations WHERE timestamp < ?",
            (cutoff,)
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def get_usage_stats() -> dict:
    """Get current usage statistics."""
    conn = get_db()
    try:
        stats = {}
        now = time.time()

        rows = conn.execute("""
            SELECT tool_name, COUNT(*) as count
            FROM tool_invocations
            WHERE timestamp > ? AND allowed = TRUE
            GROUP BY tool_name
            ORDER BY count DESC
        """, (now - 3600,)).fetchall()

        for row in rows:
            limits = get_limits(row["tool_name"])
            stats[row["tool_name"]] = {
                "count_last_hour": row["count"],
                "limit_per_hour": limits["per_hour"],
                "usage_pct": (row["count"] / limits["per_hour"]) * 100
            }

        return stats
    finally:
        conn.close()


def main():
    """Main entry point for the hook."""
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "unknown")

    skip_tools = {"Read", "Glob", "Grep", "TaskList", "TaskGet"}
    if tool_name in skip_tools:
        sys.exit(0)

    allowed, reason = check_rate_limit(tool_name)

    if not allowed:
        print(json.dumps({
            "error": reason,
            "tool": tool_name,
            "suggestion": "Wait before retrying or use a different approach."
        }), file=sys.stderr)
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--stats":
            stats = get_usage_stats()
            print(json.dumps(stats, indent=2))
        elif sys.argv[1] == "--cleanup":
            deleted = cleanup_old_records()
            print(f"Deleted {deleted} old records")
        elif sys.argv[1] == "--test":
            tool = sys.argv[2] if len(sys.argv) > 2 else "Bash"
            allowed, reason = check_rate_limit(tool)
            print(f"Tool: {tool}")
            print(f"Allowed: {allowed}")
            if reason:
                print(f"Reason: {reason}")
    else:
        main()
