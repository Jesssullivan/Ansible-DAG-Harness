"""Integration between A2A protocol and HOTL supervisor.

This module provides the bridge between the A2A protocol and the HOTL
supervisor, enabling agent handoff capabilities.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness.a2a.protocol import A2ACapability, A2AStatus
from harness.a2a.server import A2AServer

if TYPE_CHECKING:
    from harness.hotl.supervisor import HOTLSupervisor

logger = logging.getLogger(__name__)


class A2ASupervisorIntegration:
    """
    Integrates A2A protocol with HOTL supervisor.

    This class:
    - Registers supervisor capabilities with A2A server
    - Handles handoff requests from other agents
    - Provides capability handlers for supervisor operations

    Example:
        supervisor = HOTLSupervisor(db=db, config=config)
        a2a_server = A2AServer()

        integration = A2ASupervisorIntegration(supervisor, a2a_server)
        integration.setup()

        # Now handoff requests will be handled by the supervisor
    """

    def __init__(
        self,
        supervisor: "HOTLSupervisor",
        a2a_server: A2AServer,
    ):
        """Initialize the integration.

        Args:
            supervisor: HOTL supervisor instance
            a2a_server: A2A server instance
        """
        self.supervisor = supervisor
        self.a2a_server = a2a_server
        self._setup_complete = False

    def setup(self) -> None:
        """Set up the integration by registering handlers."""
        if self._setup_complete:
            return

        # Register capability handlers
        self._register_capabilities()

        # Register handoff handler
        self.a2a_server.register_handoff_handler(
            handler=self._handle_handoff,
            async_handler=False,  # Use sync handler for subprocess spawning
        )

        self._setup_complete = True
        logger.info("A2A supervisor integration setup complete")

    def _register_capabilities(self) -> None:
        """Register supervisor capabilities with A2A server."""

        # Ansible role analysis
        self.a2a_server.register_capability(
            A2ACapability.ANSIBLE_ROLES,
            handler=self._handle_ansible_roles,
            description="Analyze and manage Ansible roles",
        )

        # Dependency analysis
        self.a2a_server.register_capability(
            A2ACapability.DEPENDENCY_ANALYSIS,
            handler=self._handle_dependency_analysis,
            description="Analyze role dependencies",
        )

        # Workflow execution
        self.a2a_server.register_capability(
            A2ACapability.WORKFLOW_EXECUTION,
            handler=self._handle_workflow_execution,
            description="Execute HOTL workflows",
        )

        # Testing
        self.a2a_server.register_capability(
            A2ACapability.TESTING,
            handler=self._handle_testing,
            description="Run tests and analyze results",
        )

    def _handle_handoff(
        self,
        task: str,
        context: dict[str, Any],
        files_changed: list[str],
        reason: str,
        continuation_prompt: str,
    ) -> dict[str, Any]:
        """Handle a handoff request from another agent.

        Args:
            task: Task description
            context: Context from the handing-off agent
            files_changed: List of files changed by the previous agent
            reason: Reason for the handoff
            continuation_prompt: Prompt for continuing the work

        Returns:
            Dictionary with 'accepted', 'session_id', 'reason'
        """
        try:
            # Create a combined task from handoff data
            full_task = self._build_handoff_task(
                task=task,
                reason=reason,
                continuation_prompt=continuation_prompt,
                files_changed=files_changed,
            )

            # Spawn a new agent to handle the handoff
            session = self.supervisor.spawn_agent_manually(
                task=full_task,
                working_dir=Path(context.get("working_dir", self.supervisor.repo_root)),
                context={
                    **context,
                    "handoff": True,
                    "handoff_reason": reason,
                    "files_changed": files_changed,
                },
            )

            logger.info(f"Accepted handoff, created session {session.id}")

            return {
                "accepted": True,
                "session_id": session.id,
                "reason": f"Handoff accepted, session {session.id} created",
            }

        except Exception as e:
            logger.error(f"Handoff failed: {e}")
            return {
                "accepted": False,
                "session_id": None,
                "reason": str(e),
            }

    def _build_handoff_task(
        self,
        task: str,
        reason: str,
        continuation_prompt: str,
        files_changed: list[str],
    ) -> str:
        """Build a task prompt for the handoff.

        Args:
            task: Original task description
            reason: Reason for handoff
            continuation_prompt: Continuation instructions
            files_changed: List of changed files

        Returns:
            Combined task prompt
        """
        parts = [
            "## Agent Handoff Task",
            "",
            "You are receiving a handoff from another agent. Please continue their work.",
            "",
            f"### Original Task\n{task}",
            "",
        ]

        if reason:
            parts.append(f"### Handoff Reason\n{reason}\n")

        if continuation_prompt:
            parts.append(f"### Continuation Instructions\n{continuation_prompt}\n")

        if files_changed:
            parts.append("### Files Changed by Previous Agent")
            for f in files_changed[:20]:  # Limit to 20 files
                parts.append(f"- {f}")
            if len(files_changed) > 20:
                parts.append(f"- ... and {len(files_changed) - 20} more files")
            parts.append("")

        parts.extend(
            [
                "### Instructions",
                "1. Review the context and files changed by the previous agent",
                "2. Continue the work as described in the task",
                "3. Report progress using agent_report_progress tool",
                "4. If you encounter blockers, use agent_request_intervention",
            ]
        )

        return "\n".join(parts)

    def _handle_ansible_roles(
        self,
        task: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle Ansible role analysis requests.

        Args:
            task: Task description
            context: Request context

        Returns:
            Result dictionary
        """
        try:
            # Get role information from database
            role_name = context.get("role_name")
            if role_name:
                role = self.supervisor.db.get_role(role_name)
                if role:
                    return {
                        "status": A2AStatus.SUCCESS.value,
                        "role": {
                            "name": role.name,
                            "wave": role.wave,
                            "has_molecule_tests": role.has_molecule_tests,
                            "description": role.description,
                        },
                    }
                else:
                    return {
                        "status": A2AStatus.FAILURE.value,
                        "error": f"Role '{role_name}' not found",
                    }

            # List all roles if no specific role requested
            roles = self.supervisor.db.list_roles()
            return {
                "status": A2AStatus.SUCCESS.value,
                "roles": [
                    {
                        "name": r.name,
                        "wave": r.wave,
                        "has_molecule_tests": r.has_molecule_tests,
                    }
                    for r in roles
                ],
            }

        except Exception as e:
            logger.error(f"Ansible roles handler failed: {e}")
            return {
                "status": A2AStatus.FAILURE.value,
                "error": str(e),
            }

    def _handle_dependency_analysis(
        self,
        task: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle dependency analysis requests.

        Args:
            task: Task description
            context: Request context

        Returns:
            Result dictionary
        """
        try:
            role_name = context.get("role_name")
            if not role_name:
                # Return deployment order
                order = self.supervisor.db.get_deployment_order()
                return {
                    "status": A2AStatus.SUCCESS.value,
                    "deployment_order": order,
                }

            # Get dependencies for specific role
            transitive = context.get("transitive", False)
            deps = self.supervisor.db.get_dependencies(role_name, transitive=transitive)
            reverse_deps = self.supervisor.db.get_reverse_dependencies(
                role_name, transitive=transitive
            )

            return {
                "status": A2AStatus.SUCCESS.value,
                "role": role_name,
                "dependencies": [{"name": name, "depth": depth} for name, depth in deps],
                "reverse_dependencies": [
                    {"name": name, "depth": depth} for name, depth in reverse_deps
                ],
            }

        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return {
                "status": A2AStatus.FAILURE.value,
                "error": str(e),
            }

    def _handle_workflow_execution(
        self,
        task: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle workflow execution requests.

        Args:
            task: Task description
            context: Request context

        Returns:
            Result dictionary
        """
        try:
            # Get workflow status
            execution_id = context.get("execution_id")
            if execution_id:
                with self.supervisor.db.connection() as conn:
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

                    if row:
                        return {
                            "status": A2AStatus.SUCCESS.value,
                            "execution": dict(row),
                        }
                    else:
                        return {
                            "status": A2AStatus.FAILURE.value,
                            "error": f"Execution {execution_id} not found",
                        }

            # Return recent executions
            with self.supervisor.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT we.id, we.status, r.name as role_name, wd.name as workflow_name
                    FROM workflow_executions we
                    JOIN roles r ON we.role_id = r.id
                    JOIN workflow_definitions wd ON we.workflow_id = wd.id
                    ORDER BY we.created_at DESC
                    LIMIT 10
                    """
                ).fetchall()

            return {
                "status": A2AStatus.SUCCESS.value,
                "executions": [dict(row) for row in rows],
            }

        except Exception as e:
            logger.error(f"Workflow execution handler failed: {e}")
            return {
                "status": A2AStatus.FAILURE.value,
                "error": str(e),
            }

    def _handle_testing(
        self,
        task: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle testing requests.

        Args:
            task: Task description
            context: Request context

        Returns:
            Result dictionary
        """
        try:
            role_name = context.get("role_name")

            # Get active regressions
            regressions = self.supervisor.db.get_active_regressions(role_name)

            result: dict[str, Any] = {
                "status": A2AStatus.SUCCESS.value,
                "active_regressions": len(regressions),
            }

            if role_name:
                # Get test history for specific role
                test_runs = self.supervisor.db.get_recent_test_runs(role_name, limit=10)
                result["test_history"] = [r.model_dump() for r in test_runs]

            if regressions:
                result["regression_details"] = [
                    {"role": r.role_name, "test_name": r.test_name} for r in regressions[:10]
                ]

            return result

        except Exception as e:
            logger.error(f"Testing handler failed: {e}")
            return {
                "status": A2AStatus.FAILURE.value,
                "error": str(e),
            }


def create_a2a_integration(
    supervisor: "HOTLSupervisor",
    version: str = "0.2.0",
) -> tuple[A2AServer, A2ASupervisorIntegration]:
    """Create and set up A2A integration for a supervisor.

    Args:
        supervisor: HOTL supervisor instance
        version: Server version

    Returns:
        Tuple of (A2AServer, A2ASupervisorIntegration)
    """
    server = A2AServer(version=version)
    integration = A2ASupervisorIntegration(supervisor, server)
    integration.setup()
    return server, integration
