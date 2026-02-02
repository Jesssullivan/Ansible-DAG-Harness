"""DAG-based workflow execution following LangGraph patterns.

This module provides both the original custom implementation and the
newer LangGraph-based implementation.

DEPRECATION NOTICE:
    The original implementation (WorkflowGraph, Node, NodeResult, NodeContext,
    create_box_up_role_graph) is deprecated and will be removed in version 0.3.0.
    Please migrate to the LangGraph implementation:

    - BoxUpRoleState: TypedDict state schema
    - LangGraphWorkflowRunner: Async workflow execution
    - create_langgraph_workflow: Graph factory function
    - create_initial_state: State initialization helper

Example migration:
    # Old (deprecated):
    from harness.dag import create_box_up_role_graph
    graph = create_box_up_role_graph(db)
    result = await graph.execute(role_name)

    # New (recommended):
    from harness.dag import LangGraphWorkflowRunner, create_initial_state
    runner = LangGraphWorkflowRunner(db_path)
    state = create_initial_state(role_name, db_path)
    result = await runner.execute(state)

SUBGRAPH COMPOSITION (Task #22):
    The workflow is now organized into modular subgraphs for better
    testability and maintainability:

    - Validation Subgraph: Role validation and dependency analysis
    - Testing Subgraph: Molecule, pytest, and deploy validation
    - GitLab Subgraph: Issue, MR, and merge train operations
    - Notification Subgraph: Summary and failure notifications

    Use the composed workflow for production:
        from harness.dag import create_composed_workflow
        workflow = create_composed_workflow(db_path)
        compiled = workflow.compile()
        result = await compiled.ainvoke(initial_state)

    Or use individual subgraphs for testing:
        from harness.dag import create_validation_subgraph
        validation = create_validation_subgraph()
        result = await validation.ainvoke(test_state)
"""

import warnings
from typing import Any

# Checkpointer factory with Postgres/SQLite support (Task #23)
from harness.dag.checkpointer import (
    CheckpointerContext,
    CheckpointerFactory,
    check_postgres_health,
    cleanup_old_checkpoints,
    cleanup_old_checkpoints_by_url,
    get_postgres_url,
    is_postgres_available,
    migrate_sqlite_to_postgres,
)

# LangGraph implementation (recommended)
from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    LangGraphWorkflowRunner,
    create_initial_state,
)
from harness.dag.langgraph_engine import (
    create_box_up_role_graph as create_langgraph_workflow,
)

# Subgraph composition (Task #22)
from harness.dag.subgraphs import (
    SubgraphWorkflowRunner,
    create_composed_workflow,
    create_gitlab_subgraph,
    create_notification_subgraph,
    create_testing_subgraph,
    create_validation_subgraph,
)

# Track which deprecated names have been warned about
_warned_deprecations: set[str] = set()

# Deprecated exports - these will trigger warnings on first access
_DEPRECATED_EXPORTS = {
    "WorkflowGraph": "harness.dag.graph",
    "Node": "harness.dag.nodes",
    "NodeResult": "harness.dag.nodes",
    "NodeContext": "harness.dag.nodes",
    "create_box_up_role_graph": "harness.dag.graph",
}


def __getattr__(name: str) -> Any:
    """Lazy import deprecated names with deprecation warnings.

    This allows the deprecated names to still work while warning users
    to migrate to the new LangGraph implementation.
    """
    if name in _DEPRECATED_EXPORTS:
        module_path = _DEPRECATED_EXPORTS[name]

        # Only warn once per name per session
        if name not in _warned_deprecations:
            warnings.warn(
                f"{name} is deprecated and will be removed in version 0.3.0. "
                f"Please migrate to the LangGraph implementation. "
                f"See harness.dag module docstring for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )
            _warned_deprecations.add(name)

        # Import and return the deprecated item
        if module_path == "harness.dag.graph":
            from harness.dag.graph import WorkflowGraph
            from harness.dag.graph import create_box_up_role_graph as _create_graph

            if name == "WorkflowGraph":
                return WorkflowGraph
            elif name == "create_box_up_role_graph":
                return _create_graph
        elif module_path == "harness.dag.nodes":
            from harness.dag.nodes import Node, NodeContext, NodeResult

            if name == "Node":
                return Node
            elif name == "NodeResult":
                return NodeResult
            elif name == "NodeContext":
                return NodeContext

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all exported names including deprecated ones."""
    return list(__all__)


__all__ = [
    # LangGraph implementation (recommended)
    "BoxUpRoleState",
    "LangGraphWorkflowRunner",
    "create_langgraph_workflow",
    "create_initial_state",
    # Subgraph composition (Task #22)
    "create_validation_subgraph",
    "create_testing_subgraph",
    "create_gitlab_subgraph",
    "create_notification_subgraph",
    "create_composed_workflow",
    "SubgraphWorkflowRunner",
    # Checkpointer factory (Task #23)
    "CheckpointerFactory",
    "CheckpointerContext",
    "check_postgres_health",
    "cleanup_old_checkpoints",
    "cleanup_old_checkpoints_by_url",
    "migrate_sqlite_to_postgres",
    "is_postgres_available",
    "get_postgres_url",
    # Deprecated exports (will warn on use)
    "WorkflowGraph",
    "Node",
    "NodeResult",
    "NodeContext",
    "create_box_up_role_graph",
]
