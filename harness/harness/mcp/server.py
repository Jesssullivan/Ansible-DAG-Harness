"""
FastMCP server for harness state management.

Exposes tools for:
- Querying role status and dependencies
- Managing workflow execution
- Accessing test results and worktree status
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

from fastmcp import FastMCP, Context

from harness.db.state import StateDB
from harness.db.models import WorkflowStatus, WorktreeStatus, TestType
from harness.config import HarnessConfig


def create_mcp_server(db_path: Optional[str] = None) -> FastMCP:
    """Create and configure the MCP server."""
    mcp = FastMCP("dag-harness")

    # Load config and use configured db_path if not provided
    config = HarnessConfig.load()
    if db_path is None:
        db_path = config.db_path

    db = StateDB(db_path)
    repo_root = Path(config.repo_root)

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
    async def get_reverse_dependencies(role_name: str, transitive: bool = False) -> list[dict[str, Any]]:
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
                (execution_id,)
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
                (execution_id,)
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
                "nodes": [dict(n) for n in nodes]
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
                molecule_path=str(role_dir / "molecule") if (role_dir / "molecule").exists() else None,
                has_molecule_tests=(role_dir / "molecule").exists()
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
            text=True
        )

        synced = 0
        for line in result.stdout.split("\n"):
            if line.startswith("worktree "):
                wt_path = line.split(" ", 1)[1]
                # Get branch
                branch_result = subprocess.run(
                    ["git", "-C", wt_path, "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True
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
                        text=True
                    )
                    ahead = int(ahead_result.stdout.strip() or 0)
                    behind_result = subprocess.run(
                        ["git", "-C", wt_path, "rev-list", "--count", "HEAD..origin/main"],
                        capture_output=True,
                        text=True
                    )
                    behind = int(behind_result.stdout.strip() or 0)
                except Exception:
                    pass

                # Get uncommitted changes
                changes_result = subprocess.run(
                    ["git", "-C", wt_path, "status", "--porcelain"],
                    capture_output=True,
                    text=True
                )
                changes = len(changes_result.stdout.strip().split("\n")) if changes_result.stdout.strip() else 0

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
                    status=status
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
                (limit,)
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

    return mcp


# Entry point for running the MCP server
if __name__ == "__main__":
    mcp = create_mcp_server()
    mcp.run()
