"""
FastMCP server for harness state management.

Exposes tools for:
- Querying role status and dependencies
- Managing workflow execution
- Accessing test results and worktree status
- Tool search for context reduction
"""

import json
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from harness.config import HarnessConfig
from harness.db.models import WorktreeStatus
from harness.db.state import StateDB

# ============================================================================
# TOOL CATEGORIES
# ============================================================================

# Tool categories for context reduction via search_tools
# Each category groups related tools to help agents find relevant tools faster
TOOL_CATEGORIES: dict[str, list[str]] = {
    "role_management": [
        "list_roles",
        "get_role_status",
        "get_dependencies",
        "get_reverse_dependencies",
        "get_deployment_order",
        "get_dependency_graph",
        "sync_roles_from_filesystem",
    ],
    "worktree": [
        "list_worktrees",
        "get_worktree",
        "sync_worktrees_from_git",
    ],
    "workflow": [
        "get_workflow_status",
        "hotl_status",
        "hotl_cancel_executions",
        "hotl_get_recent_executions",
        "hotl_get_health",
    ],
    "testing": [
        "get_test_history",
        "get_active_regressions",
    ],
    "credentials": [
        "get_credentials",
    ],
    "merge_train": [
        "get_merge_train_status",
    ],
    "agent": [
        "agent_report_progress",
        "agent_request_intervention",
        "agent_log_file_operation",
        "agent_get_session_context",
        "agent_list_sessions",
        "agent_get_file_changes",
    ],
    "search": [
        "search_tools",
        "list_tool_categories",
    ],
    "costs": [
        "track_token_usage",
        "get_session_costs",
        "get_cost_summary",
    ],
}

# Tool descriptions for search matching
TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_roles": "List all Ansible roles with their status, optionally filtered by wave",
    "get_role_status": "Get detailed status for a specific role including worktree, issue, MR, and test info",
    "get_dependencies": "Get dependencies for a role, optionally including transitive dependencies",
    "get_reverse_dependencies": "Get roles that depend on this role",
    "get_deployment_order": "Get topologically sorted deployment order for all roles",
    "get_dependency_graph": "Get full dependency graph for visualization",
    "list_worktrees": "List all git worktrees with optional status filter",
    "get_worktree": "Get worktree information for a specific role",
    "get_workflow_status": "Get status of a workflow execution by ID",
    "get_test_history": "Get recent test runs for a role",
    "get_active_regressions": "Get active test regressions, optionally filtered by role",
    "get_credentials": "Get credential requirements for a role",
    "sync_roles_from_filesystem": "Scan ansible/roles/ and sync to database",
    "sync_worktrees_from_git": "Scan git worktrees and sync status to database",
    "get_merge_train_status": "Get merge train queue status for a target branch",
    "track_token_usage": "Record token usage for cost tracking",
    "get_session_costs": "Get cost summary for a session",
    "get_cost_summary": "Get overall cost summary with model and session breakdown",
    "hotl_status": "Get current HOTL supervisor status and database statistics",
    "hotl_cancel_executions": "Cancel all running workflow executions",
    "hotl_get_recent_executions": "Get recent workflow executions",
    "hotl_get_health": "Get overall harness health status with metrics and warnings",
    "agent_report_progress": "Report progress from a running Claude Code subagent",
    "agent_request_intervention": "Request human intervention from a subagent",
    "agent_log_file_operation": "Log a file operation performed by a subagent",
    "agent_get_session_context": "Get the context and status for an agent session",
    "agent_list_sessions": "List agent sessions with optional status filter",
    "agent_get_file_changes": "Get all file changes for an agent session",
    "search_tools": "Search for available tools by query or category",
    "list_tool_categories": "List all available tool categories and their tools",
}


def _search_tools_impl(query: str, category: str | None = None) -> list[dict[str, Any]]:
    """
    Implementation of tool search logic.

    Args:
        query: Search query string (searches tool names and descriptions)
        category: Optional category filter

    Returns:
        List of matching tools with name, category, and description
    """
    results: list[dict[str, Any]] = []
    query_lower = query.lower() if query else ""

    # Build a reverse mapping of tool -> category
    tool_to_category: dict[str, str] = {}
    for cat, tools in TOOL_CATEGORIES.items():
        for tool in tools:
            tool_to_category[tool] = cat

    # Filter by category if specified
    if category:
        if category not in TOOL_CATEGORIES:
            return []
        tools_to_search = TOOL_CATEGORIES[category]
    else:
        tools_to_search = list(TOOL_DESCRIPTIONS.keys())

    # Search tools
    for tool_name in tools_to_search:
        description = TOOL_DESCRIPTIONS.get(tool_name, "")
        tool_category = tool_to_category.get(tool_name, "uncategorized")

        # Match if query is empty (list all) or found in name/description
        if not query_lower or (
            query_lower in tool_name.lower()
            or query_lower in description.lower()
            or query_lower in tool_category.lower()
        ):
            results.append(
                {
                    "name": tool_name,
                    "category": tool_category,
                    "description": description,
                }
            )

    # Sort by relevance (exact name match first, then category match)
    def sort_key(item: dict) -> tuple:
        name_match = 0 if query_lower in item["name"].lower() else 1
        cat_match = 0 if query_lower in item["category"].lower() else 1
        return (name_match, cat_match, item["name"])

    results.sort(key=sort_key)

    return results


def create_mcp_server(db_path: str | None = None) -> FastMCP:
    """Create and configure the MCP server."""
    mcp = FastMCP("dag-harness")

    # Load config and use configured db_path if not provided
    config = HarnessConfig.load()
    if db_path is None:
        db_path = config.db_path

    db = StateDB(db_path)
    repo_root = Path(config.repo_root)

    # =========================================================================
    # TOOL SEARCH (for context reduction)
    # =========================================================================

    @mcp.tool()
    async def search_tools(query: str = "", category: str | None = None) -> list[dict[str, Any]]:
        """
        Search for available tools by query or category.

        This tool helps reduce context by finding relevant tools based on
        what you're trying to accomplish. Use it when you're unsure which
        tool to use or want to discover available functionality.

        Args:
            query: Search query string (searches tool names and descriptions).
                   Leave empty to list all tools or filter by category only.
            category: Optional category filter. Available categories:
                      - role_management: Tools for managing Ansible roles
                      - worktree: Git worktree operations
                      - workflow: Workflow execution and status
                      - testing: Test results and regressions
                      - credentials: Credential management
                      - merge_train: GitLab merge train status
                      - agent: Subagent communication tools
                      - search: Tool discovery

        Returns:
            List of matching tools with name, category, and description

        Examples:
            - search_tools(query="role") - Find tools related to roles
            - search_tools(category="testing") - List all testing tools
            - search_tools(query="status") - Find status-related tools
            - search_tools() - List all available tools
        """
        return _search_tools_impl(query, category)

    @mcp.tool()
    async def list_tool_categories() -> dict[str, list[str]]:
        """
        List all available tool categories and their tools.

        Returns a mapping of category names to the list of tools in each category.
        Use this to understand the organization of available tools.

        Returns:
            Dictionary mapping category names to tool lists
        """
        return TOOL_CATEGORIES

    # =========================================================================
    # ROLE TOOLS
    # =========================================================================

    @mcp.tool()
    async def list_roles(wave: int | None = None) -> list[dict[str, Any]]:
        """
        List all Ansible roles with their status.

        Args:
            wave: Optional wave filter (0-4)

        Returns:
            List of role status dictionaries
        """
        statuses = db.list_role_statuses()
        if wave is not None:
            statuses = [s for s in statuses if s.wave == wave]
        return [s.model_dump() for s in statuses]

    @mcp.tool()
    async def get_role_status(role_name: str) -> dict[str, Any] | None:
        """
        Get detailed status for a specific role.

        Args:
            role_name: Name of the Ansible role

        Returns:
            Role status including worktree, issue, MR, and test info
        """
        status = db.get_role_status(role_name)
        return status.model_dump() if status else None

    @mcp.tool()
    async def get_dependencies(role_name: str, transitive: bool = False) -> list[dict[str, Any]]:
        """
        Get dependencies for a role.

        Args:
            role_name: Name of the role
            transitive: If True, include all transitive dependencies

        Returns:
            List of dependencies with depth information
        """
        deps = db.get_dependencies(role_name, transitive=transitive)
        return [{"name": name, "depth": depth} for name, depth in deps]

    @mcp.tool()
    async def get_reverse_dependencies(
        role_name: str, transitive: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get roles that depend on this role.

        Args:
            role_name: Name of the role
            transitive: If True, include all transitive reverse dependencies

        Returns:
            List of dependent roles with depth information
        """
        deps = db.get_reverse_dependencies(role_name, transitive=transitive)
        return [{"name": name, "depth": depth} for name, depth in deps]

    @mcp.tool()
    async def get_deployment_order() -> list[str]:
        """
        Get topologically sorted deployment order.

        Returns roles in order such that dependencies come before dependents.
        """
        return db.get_deployment_order()

    @mcp.tool()
    async def get_dependency_graph() -> list[dict[str, Any]]:
        """
        Get full dependency graph for visualization.

        Returns:
            List of edge dictionaries with from_role, to_role, etc.
        """
        return db.get_dependency_graph()

    # =========================================================================
    # WORKTREE TOOLS
    # =========================================================================

    @mcp.tool()
    async def list_worktrees(status: str | None = None) -> list[dict[str, Any]]:
        """
        List all git worktrees.

        Args:
            status: Optional filter (active, stale, dirty, merged, pruned)

        Returns:
            List of worktree dictionaries
        """
        wt_status = WorktreeStatus(status) if status else None
        worktrees = db.list_worktrees(status=wt_status)
        return [w.model_dump() for w in worktrees]

    @mcp.tool()
    async def get_worktree(role_name: str) -> dict[str, Any] | None:
        """
        Get worktree for a role.

        Args:
            role_name: Name of the role

        Returns:
            Worktree info or None
        """
        wt = db.get_worktree(role_name)
        return wt.model_dump() if wt else None

    # =========================================================================
    # WORKFLOW TOOLS
    # =========================================================================

    @mcp.tool()
    async def get_workflow_status(execution_id: int) -> dict[str, Any] | None:
        """
        Get status of a workflow execution.

        Args:
            execution_id: ID of the execution

        Returns:
            Execution status with checkpoint data
        """
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT we.*, r.name as role_name, wd.name as workflow_name
                FROM workflow_executions we
                JOIN roles r ON we.role_id = r.id
                JOIN workflow_definitions wd ON we.workflow_id = wd.id
                WHERE we.id = ?
                """,
                (execution_id,),
            ).fetchone()
            if not row:
                return None

            # Get node executions
            nodes = conn.execute(
                """
                SELECT * FROM node_executions
                WHERE execution_id = ?
                ORDER BY started_at
                """,
                (execution_id,),
            ).fetchall()

            return {
                "id": row["id"],
                "role_name": row["role_name"],
                "workflow_name": row["workflow_name"],
                "status": row["status"],
                "current_node": row["current_node"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "error_message": row["error_message"],
                "nodes": [dict(n) for n in nodes],
            }

    # =========================================================================
    # TEST TOOLS
    # =========================================================================

    @mcp.tool()
    async def get_test_history(role_name: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent test runs for a role.

        Args:
            role_name: Name of the role
            limit: Maximum number of results

        Returns:
            List of test run dictionaries
        """
        runs = db.get_recent_test_runs(role_name, limit=limit)
        return [r.model_dump() for r in runs]

    @mcp.tool()
    async def get_active_regressions(role_name: str | None = None) -> list[dict[str, Any]]:
        """
        Get active test regressions.

        Args:
            role_name: Optional role name to filter by

        Returns:
            List of active regression dictionaries
        """
        regressions = db.get_active_regressions(role_name)
        return [r.model_dump() for r in regressions]

    # =========================================================================
    # CREDENTIAL TOOLS
    # =========================================================================

    @mcp.tool()
    async def get_credentials(role_name: str) -> list[dict[str, Any]]:
        """
        Get credential requirements for a role.

        Args:
            role_name: Name of the role

        Returns:
            List of credential dictionaries with entry names and purposes
        """
        creds = db.get_credentials(role_name)
        return [c.model_dump() for c in creds]

    # =========================================================================
    # SYNC TOOLS
    # =========================================================================

    @mcp.tool()
    async def sync_roles_from_filesystem() -> dict[str, int]:
        """
        Scan ansible/roles/ and sync to database.

        Returns:
            Count of roles added/updated
        """
        import yaml

        from harness.db.models import Role

        roles_path = repo_root / "ansible" / "roles"
        if not roles_path.exists():
            return {"error": f"ansible/roles not found at {roles_path}"}

        added = 0
        updated = 0

        for role_dir in roles_path.iterdir():
            if not role_dir.is_dir() or role_dir.name.startswith("_"):
                continue

            wave_num, _ = config.get_wave_for_role(role_dir.name)
            role = Role(
                name=role_dir.name,
                wave=wave_num,
                molecule_path=str(role_dir / "molecule")
                if (role_dir / "molecule").exists()
                else None,
                has_molecule_tests=(role_dir / "molecule").exists(),
            )

            # Try to read description from meta/main.yml
            meta_path = role_dir / "meta" / "main.yml"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = yaml.safe_load(f)
                        if meta and "galaxy_info" in meta:
                            role.description = meta["galaxy_info"].get("description", "")
                except Exception:
                    pass

            existing = db.get_role(role.name)
            db.upsert_role(role)
            if existing:
                updated += 1
            else:
                added += 1

        return {"added": added, "updated": updated}

    @mcp.tool()
    async def sync_worktrees_from_git() -> dict[str, int]:
        """
        Scan git worktrees and sync status to database.

        Returns:
            Count of worktrees synced
        """
        import subprocess

        from harness.db.models import Worktree

        result = subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
        )

        synced = 0
        for line in result.stdout.split("\n"):
            if line.startswith("worktree "):
                wt_path = line.split(" ", 1)[1]
                # Get branch
                branch_result = subprocess.run(
                    ["git", "-C", wt_path, "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                )
                branch = branch_result.stdout.strip()

                if not branch.startswith("sid/"):
                    continue

                role_name = branch.replace("sid/", "")
                role = db.get_role(role_name)
                if not role or not role.id:
                    continue

                # Get ahead/behind
                ahead = 0
                behind = 0
                try:
                    ahead_result = subprocess.run(
                        ["git", "-C", wt_path, "rev-list", "--count", "origin/main..HEAD"],
                        capture_output=True,
                        text=True,
                    )
                    ahead = int(ahead_result.stdout.strip() or 0)
                    behind_result = subprocess.run(
                        ["git", "-C", wt_path, "rev-list", "--count", "HEAD..origin/main"],
                        capture_output=True,
                        text=True,
                    )
                    behind = int(behind_result.stdout.strip() or 0)
                except Exception:
                    pass

                # Get uncommitted changes
                changes_result = subprocess.run(
                    ["git", "-C", wt_path, "status", "--porcelain"], capture_output=True, text=True
                )
                changes = (
                    len(changes_result.stdout.strip().split("\n"))
                    if changes_result.stdout.strip()
                    else 0
                )

                # Determine status
                if changes > 0:
                    status = WorktreeStatus.DIRTY
                elif behind > 10:
                    status = WorktreeStatus.STALE
                else:
                    status = WorktreeStatus.ACTIVE

                worktree = Worktree(
                    role_id=role.id,
                    path=wt_path,
                    branch=branch,
                    commits_ahead=ahead,
                    commits_behind=behind,
                    uncommitted_changes=changes,
                    status=status,
                )
                db.upsert_worktree(worktree)
                synced += 1

        return {"synced": synced}

    # =========================================================================
    # MERGE TRAIN TOOLS
    # =========================================================================

    @mcp.tool()
    async def get_merge_train_status(target_branch: str = "main") -> list[dict[str, Any]]:
        """
        Get merge train queue status.

        Args:
            target_branch: Target branch to check (default: main)

        Returns:
            List of merge train entries
        """
        entries = db.list_merge_train(target_branch=target_branch)
        return [e.model_dump() for e in entries]

    # =========================================================================
    # COST TRACKING TOOLS
    # =========================================================================

    @mcp.tool()
    async def track_token_usage(
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Record token usage for cost tracking.

        Used to track Claude API token usage per session for cost
        accounting and optimization.

        Args:
            session_id: Session identifier
            model: Model used (e.g., "claude-opus-4-5", "claude-sonnet-4", "claude-haiku-3-5")
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens generated
            context: Optional context metadata (role name, task type, etc.)

        Returns:
            Dict with record_id, cost, and session totals
        """

        from harness.costs.pricing import CLAUDE_PRICING, get_model_pricing

        # Get pricing for model
        pricing = get_model_pricing(model)
        if pricing is None:
            # Fallback to Sonnet pricing for unknown models
            pricing = CLAUDE_PRICING.get("claude-sonnet-4")

        cost = float(pricing.calculate_cost(input_tokens, output_tokens))

        # Record in database
        record_id = db.record_token_usage(
            session_id=session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            context=context,
        )

        # Get session totals
        session_costs = db.get_session_costs(session_id)

        return {
            "record_id": record_id,
            "cost": cost,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "session_total_cost": session_costs["total_cost"],
            "session_total_input": session_costs["total_input_tokens"],
            "session_total_output": session_costs["total_output_tokens"],
        }

    @mcp.tool()
    async def get_session_costs(session_id: str) -> dict[str, Any]:
        """
        Get cost summary for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with total cost, token counts, and breakdown by model
        """
        return db.get_session_costs(session_id)

    @mcp.tool()
    async def get_cost_summary(
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Get overall cost summary with model and session breakdown.

        Args:
            start_date: Start date (ISO format YYYY-MM-DD, inclusive)
            end_date: End date (ISO format YYYY-MM-DD, inclusive)

        Returns:
            Dict with totals, breakdown by model, and top sessions by cost
        """
        return db.get_cost_summary(start_date=start_date, end_date=end_date)

    # =========================================================================
    # HOTL TOOLS
    # =========================================================================

    @mcp.tool()
    async def hotl_status() -> dict[str, Any]:
        """
        Get current HOTL supervisor status and database statistics.

        Returns:
            Status dict with execution counts and regression info
        """
        stats = db.get_statistics()
        return {
            "pending_executions": stats.get("pending_executions", 0),
            "running_executions": stats.get("running_executions", 0),
            "active_regressions": stats.get("active_regressions", 0),
            "roles_count": stats.get("roles", 0),
            "test_runs": stats.get("test_runs", 0),
            "workflow_executions": stats.get("workflow_executions", 0),
        }

    @mcp.tool()
    async def hotl_cancel_executions() -> dict[str, Any]:
        """
        Cancel all running workflow executions.

        Use this to stop HOTL operations that are stuck or need to be aborted.

        Returns:
            Count of cancelled executions
        """
        with db.connection() as conn:
            count = conn.execute(
                "UPDATE workflow_executions SET status = 'cancelled' WHERE status = 'running'"
            ).rowcount
        return {"cancelled": count}

    @mcp.tool()
    async def hotl_get_recent_executions(limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent workflow executions.

        Args:
            limit: Maximum number of executions to return

        Returns:
            List of recent execution records
        """
        with db.connection() as conn:
            rows = conn.execute(
                """
                SELECT we.id, we.status, we.current_node, we.started_at, we.completed_at,
                       we.error_message, r.name as role_name, wd.name as workflow_name
                FROM workflow_executions we
                JOIN roles r ON we.role_id = r.id
                JOIN workflow_definitions wd ON we.workflow_id = wd.id
                ORDER BY we.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    @mcp.tool()
    async def hotl_get_health() -> dict[str, Any]:
        """
        Get overall harness health status.

        Returns aggregated metrics for monitoring HOTL autonomous operation.

        Returns:
            Health status with metrics and warnings
        """
        stats = db.get_statistics()
        data_validation = db.validate_data_integrity()
        dep_validation = db.validate_dependencies()

        warnings = []

        if stats.get("active_regressions", 0) > 0:
            warnings.append(f"{stats['active_regressions']} active test regressions")

        if not data_validation.get("valid"):
            warnings.append("Data integrity issues detected")

        if not dep_validation.get("valid"):
            cycles = dep_validation.get("cycles", [])
            if cycles:
                warnings.append(f"{len(cycles)} dependency cycle(s) detected")

        return {
            "healthy": len(warnings) == 0,
            "warnings": warnings,
            "metrics": {
                "roles": stats.get("roles", 0),
                "active_regressions": stats.get("active_regressions", 0),
                "pending_executions": stats.get("pending_executions", 0),
                "running_executions": stats.get("running_executions", 0),
            },
            "data_integrity_valid": data_validation.get("valid", False),
            "dependency_graph_valid": dep_validation.get("valid", False),
        }

    # =========================================================================
    # AGENT FEEDBACK TOOLS (for Claude Code subagents in HOTL mode)
    # =========================================================================

    @mcp.tool()
    async def agent_report_progress(session_id: str, progress: str) -> dict[str, Any]:
        """
        Report progress from a running Claude Code subagent.

        Used by subagents to communicate their current status and progress
        to the HOTL supervisor.

        Args:
            session_id: The agent session ID (provided in environment)
            progress: Progress message describing current status

        Returns:
            Acknowledgment with timestamp
        """
        with db.connection() as conn:
            # Get existing progress
            row = conn.execute(
                "SELECT progress_json FROM agent_sessions WHERE id = ?", (session_id,)
            ).fetchone()

            if not row:
                return {"success": False, "error": f"Session not found: {session_id}"}

            # Append new progress
            existing = json.loads(row["progress_json"]) if row["progress_json"] else []
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            existing.append(f"[{timestamp}] {progress}")

            conn.execute(
                "UPDATE agent_sessions SET progress_json = ? WHERE id = ?",
                (json.dumps(existing), session_id),
            )

        return {
            "success": True,
            "timestamp": timestamp,
            "session_id": session_id,
        }

    @mcp.tool()
    async def agent_request_intervention(session_id: str, reason: str) -> dict[str, Any]:
        """
        Request human intervention from a running Claude Code subagent.

        Used when the agent encounters a situation it cannot handle autonomously
        and needs human guidance or approval.

        Args:
            session_id: The agent session ID
            reason: Detailed explanation of why intervention is needed

        Returns:
            Acknowledgment that intervention has been requested
        """
        with db.connection() as conn:
            # Update session status
            result = conn.execute(
                """
                UPDATE agent_sessions
                SET status = 'needs_human', intervention_reason = ?
                WHERE id = ?
                RETURNING id
                """,
                (reason, session_id),
            ).fetchone()

            if not result:
                return {"success": False, "error": f"Session not found: {session_id}"}

            # Log to audit
            conn.execute(
                """
                INSERT INTO audit_log (entity_type, entity_id, action, new_value, actor)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "agent_session",
                    0,
                    "intervention_requested",
                    json.dumps({"session_id": session_id, "reason": reason}),
                    "agent",
                ),
            )

        return {
            "success": True,
            "session_id": session_id,
            "status": "needs_human",
            "message": "Intervention requested. The HOTL supervisor has been notified.",
        }

    @mcp.tool()
    async def agent_log_file_operation(
        session_id: str,
        file_path: str,
        operation: str,
        diff: str | None = None,
        old_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Log a file operation performed by a Claude Code subagent.

        Used to track all file changes made during autonomous operation
        for review and potential rollback.

        Args:
            session_id: The agent session ID
            file_path: Path to the file that was modified
            operation: Type of operation (create, modify, delete, rename)
            diff: Optional git-style diff showing changes
            old_path: For renames, the original file path

        Returns:
            Acknowledgment with file change ID
        """
        valid_operations = ("create", "modify", "delete", "rename")
        if operation not in valid_operations:
            return {
                "success": False,
                "error": f"Invalid operation. Must be one of: {valid_operations}",
            }

        with db.connection() as conn:
            # Verify session exists
            session = conn.execute(
                "SELECT id FROM agent_sessions WHERE id = ?", (session_id,)
            ).fetchone()

            if not session:
                return {"success": False, "error": f"Session not found: {session_id}"}

            # Insert file change record
            cursor = conn.execute(
                """
                INSERT INTO agent_file_changes (session_id, file_path, change_type, diff, old_path)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, file_path, change_type) DO UPDATE SET
                    diff = excluded.diff,
                    old_path = excluded.old_path
                RETURNING id
                """,
                (session_id, file_path, operation, diff, old_path),
            )
            change_id = cursor.fetchone()[0]

        return {
            "success": True,
            "change_id": change_id,
            "session_id": session_id,
            "file_path": file_path,
            "operation": operation,
        }

    @mcp.tool()
    async def agent_get_session_context(session_id: str) -> dict[str, Any]:
        """
        Get the context and status for an agent session.

        Allows a subagent to retrieve its own context and check status.

        Args:
            session_id: The agent session ID

        Returns:
            Session context and current status
        """
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT id, task, status, context_json, working_dir,
                       execution_id, created_at, started_at
                FROM agent_sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()

            if not row:
                return {"success": False, "error": f"Session not found: {session_id}"}

            # Get file changes count
            changes = conn.execute(
                "SELECT COUNT(*) as count FROM agent_file_changes WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        context = json.loads(row["context_json"]) if row["context_json"] else {}

        return {
            "success": True,
            "session_id": row["id"],
            "task": row["task"],
            "status": row["status"],
            "context": context,
            "working_dir": row["working_dir"],
            "execution_id": row["execution_id"],
            "file_changes_count": changes["count"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
        }

    @mcp.tool()
    async def agent_list_sessions(
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        List agent sessions with optional status filter.

        Args:
            status: Optional status filter (pending, running, completed, failed, needs_human, cancelled)
            limit: Maximum number of sessions to return

        Returns:
            List of agent session summaries
        """
        with db.connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM v_agent_sessions
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM v_agent_sessions
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            return [dict(row) for row in rows]

    @mcp.tool()
    async def agent_get_file_changes(session_id: str) -> list[dict[str, Any]]:
        """
        Get all file changes for an agent session.

        Args:
            session_id: The agent session ID

        Returns:
            List of file change records
        """
        with db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, file_path, change_type, diff, old_path, created_at
                FROM agent_file_changes
                WHERE session_id = ?
                ORDER BY created_at
                """,
                (session_id,),
            ).fetchall()

            return [dict(row) for row in rows]

    return mcp


# Entry point for running the MCP server
if __name__ == "__main__":
    mcp = create_mcp_server()
    mcp.run()
