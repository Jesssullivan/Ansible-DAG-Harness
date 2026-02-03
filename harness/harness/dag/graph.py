"""
DAG-based workflow execution engine.

Implements LangGraph-style execution with:
- State accumulation across nodes
- Checkpointing for resumption
- Conditional edges and routing
- Retry logic with backoff
- Human-in-the-loop breakpoints
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.dag.nodes import (
    AnalyzeDependenciesNode,
    CheckDependenciesNode,
    ConditionalEdge,
    CreateCommitNode,
    CreateGitLabIssueNode,
    CreateMergeRequestNode,
    CreateWorktreeNode,
    Edge,
    Node,
    NodeContext,
    NodeDefinition,
    NodeResult,
    PushBranchNode,
    ReportSummaryNode,
    RouterEdge,
    RunMoleculeTestsNode,
    ValidateRoleNode,
    WarnReverseDepsNode,
)
from harness.db.models import NodeStatus, WorkflowStatus
from harness.db.state import StateDB

logger = logging.getLogger(__name__)


@dataclass
class ExecutionEvent:
    """Event emitted during workflow execution."""

    timestamp: datetime
    event_type: str
    node_name: str | None
    data: dict[str, Any]


class WorkflowGraph:
    """
    DAG-based workflow execution engine.

    Example usage:
        graph = WorkflowGraph(db)
        graph.add_node(ValidateRoleNode(), edges={NodeResult.SUCCESS: "analyze"})
        graph.add_node(AnalyzeDependenciesNode(), edges={NodeResult.SUCCESS: "check_deps"})
        ...
        graph.set_entry_point("validate_role")
        graph.set_terminal_nodes(["report_summary"])

        # Execute with repo config
        result = await graph.execute("common", repo_root=Path("/path/to/repo"))
    """

    def __init__(self, db: StateDB, name: str = "box_up_role"):
        self.db = db
        self.name = name
        self.nodes: dict[str, NodeDefinition] = {}
        self.entry_point: str | None = None
        self.terminal_nodes: set[str] = set()
        self.event_handlers: list[Callable[[ExecutionEvent], None]] = []

    def add_node(self, node: Node, edges: dict[NodeResult, Edge] | None = None) -> "WorkflowGraph":
        """Add a node to the graph."""
        self.nodes[node.name] = NodeDefinition(name=node.name, node=node, edges=edges or {})
        return self

    def set_entry_point(self, node_name: str) -> "WorkflowGraph":
        """Set the entry point node."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
        self.entry_point = node_name
        return self

    def set_terminal_nodes(self, node_names: list[str]) -> "WorkflowGraph":
        """Set terminal nodes (workflow ends when reaching these)."""
        for name in node_names:
            if name not in self.nodes:
                raise ValueError(f"Node '{name}' not found")
        self.terminal_nodes = set(node_names)
        return self

    def add_event_handler(self, handler: Callable[[ExecutionEvent], None]) -> "WorkflowGraph":
        """Add event handler for observability."""
        self.event_handlers.append(handler)
        return self

    def _emit_event(
        self, event_type: str, node_name: str | None = None, data: dict | None = None
    ) -> None:
        """Emit an event to all handlers."""
        event = ExecutionEvent(
            timestamp=datetime.utcnow(), event_type=event_type, node_name=node_name, data=data or {}
        )
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")

    def _resolve_next_node(
        self, node_def: NodeDefinition, result: NodeResult, ctx: NodeContext
    ) -> str | None:
        """Resolve the next node based on result and edges."""
        edge = node_def.edges.get(result)
        if edge is None:
            return None

        if isinstance(edge, str):
            return edge
        elif isinstance(edge, ConditionalEdge):
            return edge.evaluate(ctx)
        elif isinstance(edge, RouterEdge):
            return edge.evaluate(ctx)

        return None

    async def _execute_node(
        self, node_def: NodeDefinition, ctx: NodeContext, execution_id: int
    ) -> tuple[NodeResult, dict[str, Any]]:
        """Execute a single node with retry logic."""
        node = node_def.node
        retries_remaining = node.retries

        while True:
            # Update node status to running
            self.db.update_node_execution(
                execution_id, node.name, NodeStatus.RUNNING, input_data=dict(ctx.state)
            )
            self._emit_event("node_started", node.name, {"retries_remaining": retries_remaining})

            try:
                # Check if node can be skipped
                if node.can_skip(ctx):
                    self._emit_event("node_skipped", node.name)
                    return NodeResult.SKIP, {}

                # Execute with timeout
                result, updates = await asyncio.wait_for(
                    node.execute(ctx), timeout=node.timeout_seconds
                )

                self._emit_event(
                    "node_completed", node.name, {"result": result.value, "updates": updates}
                )

                return result, updates

            except TimeoutError:
                retries_remaining -= 1
                if retries_remaining <= 0:
                    self._emit_event("node_timeout", node.name)
                    return NodeResult.FAILURE, {"error": "Timeout exceeded all retries"}

                self._emit_event(
                    "node_retry",
                    node.name,
                    {"reason": "timeout", "retries_remaining": retries_remaining},
                )
                await asyncio.sleep(2 ** (node.retries - retries_remaining))  # Exponential backoff

            except Exception as e:
                logger.exception(f"Node {node.name} failed")
                retries_remaining -= 1
                if retries_remaining <= 0:
                    self._emit_event("node_failed", node.name, {"error": str(e)})
                    return NodeResult.FAILURE, {"error": str(e)}

                self._emit_event(
                    "node_retry",
                    node.name,
                    {"reason": str(e), "retries_remaining": retries_remaining},
                )
                await asyncio.sleep(2 ** (node.retries - retries_remaining))

    async def execute(
        self,
        role_name: str,
        resume_from: int | None = None,
        breakpoints: set[str] | None = None,
        repo_root: Path | None = None,
        repo_python: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute the workflow for a role.

        Args:
            role_name: Name of the role to process
            resume_from: Optional execution ID to resume from
            breakpoints: Optional set of node names to pause before
            repo_root: Path to the target repository root
            repo_python: Path to the Python interpreter for the target repo

        Returns:
            Final state after execution
        """
        if not self.entry_point:
            raise ValueError("Entry point not set")

        # Initialize or restore execution
        if resume_from:
            execution_id = resume_from
            checkpoint = self.db.get_checkpoint(execution_id)
            if checkpoint:
                ctx = NodeContext(
                    role_name=role_name,
                    execution_id=execution_id,
                    state=checkpoint.get("state", {}),
                    metadata=checkpoint.get("metadata", {}),
                    repo_root=repo_root,
                    repo_python=repo_python,
                )
                current_node = checkpoint.get("current_node")
            else:
                raise ValueError(f"No checkpoint found for execution {execution_id}")
        else:
            # Register workflow definition
            nodes_dict = [nd.to_dict() for nd in self.nodes.values()]
            edges_dict = [
                {"from": k, "to": v}
                for k, v in [(nd.name, list(nd.edges.keys())) for nd in self.nodes.values()]
            ]
            self.db.create_workflow_definition(
                self.name, f"Box up role workflow for {role_name}", nodes_dict, edges_dict
            )

            # Create new execution
            execution_id = self.db.create_execution(self.name, role_name)
            ctx = NodeContext(
                role_name=role_name,
                execution_id=execution_id,
                repo_root=repo_root,
                repo_python=repo_python,
            )
            current_node = self.entry_point

        self._emit_event(
            "workflow_started",
            data={"role": role_name, "execution_id": execution_id, "entry_point": current_node},
        )

        self.db.update_execution_status(
            execution_id, WorkflowStatus.RUNNING, current_node=current_node
        )

        completed_nodes: list[str] = []
        try:
            while current_node:
                # Check for breakpoint
                if breakpoints and current_node in breakpoints:
                    self._emit_event("breakpoint_hit", current_node)
                    self.db.update_execution_status(
                        execution_id, WorkflowStatus.PAUSED, current_node=current_node
                    )
                    self.db.checkpoint_execution(
                        execution_id,
                        {
                            "state": ctx.state,
                            "metadata": ctx.metadata,
                            "current_node": current_node,
                        },
                    )
                    return {
                        "status": "paused",
                        "execution_id": execution_id,
                        "paused_at": current_node,
                        "state": ctx.state,
                    }

                node_def = self.nodes.get(current_node)
                if not node_def:
                    raise ValueError(f"Node '{current_node}' not found")

                # Execute node
                result, updates = await self._execute_node(node_def, ctx, execution_id)

                # Update context state
                ctx.update(updates)

                # Update node execution record
                status = {
                    NodeResult.SUCCESS: NodeStatus.COMPLETED,
                    NodeResult.FAILURE: NodeStatus.FAILED,
                    NodeResult.SKIP: NodeStatus.SKIPPED,
                    NodeResult.RETRY: NodeStatus.PENDING,
                    NodeResult.HUMAN_NEEDED: NodeStatus.PENDING,
                }.get(result, NodeStatus.FAILED)

                self.db.update_node_execution(
                    execution_id,
                    current_node,
                    status,
                    output_data=updates,
                    error_message=updates.get("error"),
                )

                # Handle results
                if result == NodeResult.FAILURE:
                    self._emit_event(
                        "workflow_failed", current_node, {"error": updates.get("error")}
                    )
                    self.db.update_execution_status(
                        execution_id, WorkflowStatus.FAILED, error_message=updates.get("error")
                    )
                    return {
                        "status": "failed",
                        "execution_id": execution_id,
                        "failed_at": current_node,
                        "error": updates.get("error"),
                        "state": ctx.state,
                    }

                if result == NodeResult.HUMAN_NEEDED:
                    self._emit_event("human_needed", current_node, updates)
                    self.db.update_execution_status(
                        execution_id, WorkflowStatus.PAUSED, current_node=current_node
                    )
                    self.db.checkpoint_execution(
                        execution_id,
                        {
                            "state": ctx.state,
                            "metadata": ctx.metadata,
                            "current_node": current_node,
                            "human_input_needed": updates.get("human_input_needed", {}),
                        },
                    )
                    return {
                        "status": "human_needed",
                        "execution_id": execution_id,
                        "paused_at": current_node,
                        "human_input_needed": updates.get("human_input_needed"),
                        "state": ctx.state,
                    }

                completed_nodes.append(current_node)

                # Checkpoint after each successful node
                self.db.checkpoint_execution(
                    execution_id,
                    {
                        "state": ctx.state,
                        "metadata": ctx.metadata,
                        "completed_nodes": completed_nodes,
                    },
                )

                # Check if terminal
                if current_node in self.terminal_nodes:
                    break

                # Resolve next node
                current_node = self._resolve_next_node(node_def, result, ctx)
                if current_node:
                    self.db.update_execution_status(
                        execution_id, WorkflowStatus.RUNNING, current_node=current_node
                    )

            # Workflow completed
            self._emit_event(
                "workflow_completed",
                data={"execution_id": execution_id, "completed_nodes": completed_nodes},
            )
            self.db.update_execution_status(execution_id, WorkflowStatus.COMPLETED)

            return {
                "status": "completed",
                "execution_id": execution_id,
                "state": ctx.state,
                "summary": ctx.get("summary"),
            }

        except Exception as e:
            logger.exception("Workflow execution failed")
            self._emit_event("workflow_error", data={"error": str(e)})
            self.db.update_execution_status(
                execution_id, WorkflowStatus.FAILED, error_message=str(e)
            )
            return {
                "status": "error",
                "execution_id": execution_id,
                "error": str(e),
                "state": ctx.state,
            }

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph structure for storage/visualization."""
        return {
            "name": self.name,
            "entry_point": self.entry_point,
            "terminal_nodes": list(self.terminal_nodes),
            "nodes": {name: nd.to_dict() for name, nd in self.nodes.items()},
        }


def create_box_up_role_graph(db: StateDB) -> WorkflowGraph:
    """
    Create the standard box-up-role workflow graph.

    Flow:
        validate_role -> analyze_dependencies -> check_dependencies
            -> warn_reverse_deps -> create_worktree -> run_molecule_tests
            -> create_commit -> push_branch -> create_gitlab_issue
            -> create_merge_request -> report_summary

    Note: check_dependencies verifies UPSTREAM deps (roles this role needs).
          warn_reverse_deps provides INFO about downstream deps (does NOT block).
    """
    graph = WorkflowGraph(db, "box_up_role")

    # Add nodes with edges
    graph.add_node(
        ValidateRoleNode(),
        edges={NodeResult.SUCCESS: "analyze_dependencies", NodeResult.FAILURE: "report_summary"},
    )

    graph.add_node(
        AnalyzeDependenciesNode(),
        edges={NodeResult.SUCCESS: "check_dependencies", NodeResult.FAILURE: "report_summary"},
    )

    # CheckDependenciesNode checks UPSTREAM deps (roles this role needs)
    # Foundation roles (Wave 0) automatically pass
    graph.add_node(
        CheckDependenciesNode(),
        edges={
            NodeResult.SUCCESS: "warn_reverse_deps",
            NodeResult.FAILURE: "report_summary",  # Exit with blocking deps message
        },
    )

    # WarnReverseDepsNode provides INFO about downstream deps (does NOT block)
    graph.add_node(
        WarnReverseDepsNode(),
        edges={
            NodeResult.SUCCESS: "create_worktree",  # Always succeeds (just warns)
        },
    )

    graph.add_node(
        CreateWorktreeNode(),
        edges={NodeResult.SUCCESS: "run_molecule_tests", NodeResult.FAILURE: "report_summary"},
    )

    graph.add_node(
        RunMoleculeTestsNode(),
        edges={
            NodeResult.SUCCESS: "create_commit",
            NodeResult.SKIP: "create_commit",  # No tests = still commit
            NodeResult.FAILURE: "report_summary",  # Tests must pass
        },
    )

    graph.add_node(
        CreateCommitNode(),
        edges={
            NodeResult.SUCCESS: "push_branch",
            NodeResult.SKIP: "push_branch",  # No changes = still push
            NodeResult.FAILURE: "report_summary",
        },
    )

    graph.add_node(
        PushBranchNode(),
        edges={NodeResult.SUCCESS: "create_gitlab_issue", NodeResult.FAILURE: "report_summary"},
    )

    graph.add_node(
        CreateGitLabIssueNode(),
        edges={NodeResult.SUCCESS: "create_merge_request", NodeResult.FAILURE: "report_summary"},
    )

    graph.add_node(
        CreateMergeRequestNode(),
        edges={NodeResult.SUCCESS: "report_summary", NodeResult.FAILURE: "report_summary"},
    )

    graph.add_node(
        ReportSummaryNode(),
        edges={},  # Terminal node
    )

    graph.set_entry_point("validate_role")
    graph.set_terminal_nodes(["report_summary"])

    return graph
