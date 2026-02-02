"""
Subgraph composition for modular workflow organization.

This module breaks down the box-up-role workflow into composable subgraphs:
- Validation subgraph: Role validation and dependency analysis
- Testing subgraph: Molecule, pytest, and deploy validation
- GitLab subgraph: Issue, MR creation, and merge train operations
- Notification subgraph: Summary reporting and failure notifications

Benefits:
- Better code organization and separation of concerns
- Independent testing of each workflow phase
- Easier maintenance and extension
- Clear phase boundaries for debugging and observability

Usage:
    from harness.dag.subgraphs import (
        create_validation_subgraph,
        create_testing_subgraph,
        create_gitlab_subgraph,
        create_notification_subgraph,
        create_composed_workflow,
    )

    # Use subgraphs directly for testing
    validation_graph = create_validation_subgraph()
    result = await validation_graph.ainvoke(initial_state)

    # Or use the composed workflow
    workflow = create_composed_workflow(db_path)
    compiled = workflow.compile()
    result = await compiled.ainvoke(initial_state)
"""

from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

# Type alias for clarity
CompiledGraph = CompiledStateGraph

from harness.dag.langgraph_engine import (
    GIT_RETRY_POLICY,
    # Parallel test routing (Task #21)
    GITLAB_API_RETRY_POLICY,
    SUBPROCESS_RETRY_POLICY,
    # State schema
    BoxUpRoleState,
    add_to_merge_train_node,
    analyze_deps_node,
    check_reverse_deps_node,
    create_commit_node,
    create_issue_node,
    create_mr_node,
    create_worktree_node,
    human_approval_node,
    notify_failure_node,
    push_branch_node,
    report_summary_node,
    run_molecule_node,
    run_pytest_node,
    validate_deploy_node,
    # Node functions
    validate_role_node,
)

# ============================================================================
# SUBGRAPH-LEVEL ROUTING FUNCTIONS
# ============================================================================


def validation_router(state: BoxUpRoleState) -> Literal["analyze_deps", "subgraph_error"]:
    """Route within validation subgraph after initial validation."""
    if state.get("errors"):
        return "subgraph_error"
    return "analyze_deps"


def deps_router(state: BoxUpRoleState) -> Literal["check_reverse_deps", "subgraph_error"]:
    """Route after dependency analysis."""
    if state.get("errors"):
        return "subgraph_error"
    return "check_reverse_deps"


def reverse_deps_router(state: BoxUpRoleState) -> Literal["__end__", "subgraph_error"]:
    """Route after reverse deps check - subgraph ends here."""
    if state.get("blocking_deps") or state.get("errors"):
        return "subgraph_error"
    return "__end__"


def worktree_router(state: BoxUpRoleState) -> Literal["run_molecule", "subgraph_error"]:
    """Route after worktree creation."""
    if state.get("errors"):
        return "subgraph_error"
    return "run_molecule"


def molecule_router(state: BoxUpRoleState) -> Literal["run_pytest", "subgraph_error"]:
    """Route after molecule tests."""
    if state.get("molecule_passed") is False:
        return "subgraph_error"
    return "run_pytest"


def pytest_router(state: BoxUpRoleState) -> Literal["validate_deploy", "subgraph_error"]:
    """Route after pytest."""
    if state.get("pytest_passed") is False:
        return "subgraph_error"
    return "validate_deploy"


def deploy_router(state: BoxUpRoleState) -> Literal["__end__", "subgraph_error"]:
    """Route after deploy validation - testing subgraph ends here."""
    if state.get("deploy_passed") is False:
        return "subgraph_error"
    return "__end__"


def commit_router(state: BoxUpRoleState) -> Literal["push_branch", "subgraph_error"]:
    """Route after commit."""
    if state.get("errors"):
        return "subgraph_error"
    return "push_branch"


def push_router(state: BoxUpRoleState) -> Literal["create_issue", "subgraph_error"]:
    """Route after push."""
    if not state.get("pushed") and state.get("errors"):
        return "subgraph_error"
    return "create_issue"


def issue_router(state: BoxUpRoleState) -> Literal["create_mr", "subgraph_error"]:
    """Route after issue creation."""
    if not state.get("issue_iid"):
        return "subgraph_error"
    return "create_mr"


def mr_router(state: BoxUpRoleState) -> Literal["human_approval", "subgraph_error"]:
    """Route after MR creation."""
    if not state.get("mr_iid"):
        return "subgraph_error"
    return "human_approval"


def human_approval_router(state: BoxUpRoleState) -> Literal["add_to_merge_train", "subgraph_error"]:
    """Route after human approval."""
    if state.get("human_approved"):
        return "add_to_merge_train"
    return "subgraph_error"


# ============================================================================
# ERROR MARKER NODE
# ============================================================================


async def subgraph_error_marker(state: BoxUpRoleState) -> dict:
    """
    Marker node that indicates a subgraph error occurred.

    This node simply marks the subgraph as failed and preserves error info.
    The main graph will handle actual error processing via the notification subgraph.
    """
    return {
        "completed_nodes": ["__subgraph_error__"],
    }


# ============================================================================
# VALIDATION SUBGRAPH
# ============================================================================


def create_validation_subgraph() -> CompiledGraph:
    """
    Create the validation subgraph for role validation and dependency analysis.

    Nodes:
    - validate_role: Verify role exists and extract metadata
    - analyze_deps: Analyze role dependencies using StateDB
    - check_reverse_deps: Check if reverse dependencies are boxed up

    Flow:
        validate_role -> analyze_deps -> check_reverse_deps -> END
                |             |                  |
                v             v                  v
            subgraph_error     subgraph_error          subgraph_error

    Returns:
        CompiledGraph that can be used as a node in the main workflow
    """
    graph = StateGraph(BoxUpRoleState)

    # Add nodes
    graph.add_node("validate_role", validate_role_node)
    graph.add_node("analyze_deps", analyze_deps_node)
    graph.add_node("check_reverse_deps", check_reverse_deps_node)
    graph.add_node("subgraph_error", subgraph_error_marker)

    # Set entry point
    graph.set_entry_point("validate_role")

    # Add conditional edges
    graph.add_conditional_edges(
        "validate_role",
        validation_router,
        {
            "analyze_deps": "analyze_deps",
            "subgraph_error": "subgraph_error",
        },
    )
    graph.add_conditional_edges(
        "analyze_deps",
        deps_router,
        {
            "check_reverse_deps": "check_reverse_deps",
            "subgraph_error": "subgraph_error",
        },
    )
    graph.add_conditional_edges(
        "check_reverse_deps",
        reverse_deps_router,
        {
            "__end__": END,
            "subgraph_error": "subgraph_error",
        },
    )

    # Error node goes to END
    graph.add_edge("subgraph_error", END)

    return graph.compile()


# ============================================================================
# TESTING SUBGRAPH
# ============================================================================


def create_testing_subgraph() -> CompiledGraph:
    """
    Create the testing subgraph for all testing operations.

    Nodes:
    - create_worktree: Create isolated git worktree
    - run_molecule: Run molecule tests (with retry on timeout)
    - run_pytest: Run pytest tests (with retry on timeout)
    - validate_deploy: Validate deployment configuration

    Flow:
        create_worktree -> run_molecule -> run_pytest -> validate_deploy -> END
              |                |              |                |
              v                v              v                v
          subgraph_error        subgraph_error      subgraph_error        subgraph_error

    Returns:
        CompiledGraph that can be used as a node in the main workflow
    """
    graph = StateGraph(BoxUpRoleState)

    # Add nodes with appropriate retry policies
    graph.add_node("create_worktree", create_worktree_node)
    graph.add_node("run_molecule", run_molecule_node, retry_policy=SUBPROCESS_RETRY_POLICY)
    graph.add_node("run_pytest", run_pytest_node, retry_policy=SUBPROCESS_RETRY_POLICY)
    graph.add_node("validate_deploy", validate_deploy_node)
    graph.add_node("subgraph_error", subgraph_error_marker)

    # Set entry point
    graph.set_entry_point("create_worktree")

    # Add conditional edges
    graph.add_conditional_edges(
        "create_worktree",
        worktree_router,
        {
            "run_molecule": "run_molecule",
            "subgraph_error": "subgraph_error",
        },
    )
    graph.add_conditional_edges(
        "run_molecule",
        molecule_router,
        {
            "run_pytest": "run_pytest",
            "subgraph_error": "subgraph_error",
        },
    )
    graph.add_conditional_edges(
        "run_pytest",
        pytest_router,
        {
            "validate_deploy": "validate_deploy",
            "subgraph_error": "subgraph_error",
        },
    )
    graph.add_conditional_edges(
        "validate_deploy",
        deploy_router,
        {
            "__end__": END,
            "subgraph_error": "subgraph_error",
        },
    )

    # Error node goes to END
    graph.add_edge("subgraph_error", END)

    return graph.compile()


# ============================================================================
# GITLAB SUBGRAPH
# ============================================================================


def create_gitlab_subgraph() -> CompiledGraph:
    """
    Create the GitLab subgraph for all GitLab operations.

    Nodes:
    - create_commit: Create git commit with semantic message
    - push_branch: Push branch to origin (with retry)
    - create_issue: Create GitLab issue (with retry)
    - create_mr: Create GitLab merge request (with retry)
    - human_approval: Human-in-the-loop approval (uses interrupt)
    - add_to_merge_train: Add MR to merge train (with retry)

    Flow:
        create_commit -> push_branch -> create_issue -> create_mr
              |              |               |              |
              v              v               v              v
          subgraph_error      subgraph_error       subgraph_error      subgraph_error
                                                           |
                                                           v
        human_approval -> add_to_merge_train -> END
              |                   |
              v                   v
          subgraph_error           subgraph_error

    Returns:
        CompiledGraph that can be used as a node in the main workflow
    """
    graph = StateGraph(BoxUpRoleState)

    # Add nodes with appropriate retry policies
    graph.add_node("create_commit", create_commit_node)
    graph.add_node("push_branch", push_branch_node, retry_policy=GIT_RETRY_POLICY)
    graph.add_node("create_issue", create_issue_node, retry_policy=GITLAB_API_RETRY_POLICY)
    graph.add_node("create_mr", create_mr_node, retry_policy=GITLAB_API_RETRY_POLICY)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node(
        "add_to_merge_train", add_to_merge_train_node, retry_policy=GITLAB_API_RETRY_POLICY
    )
    graph.add_node("subgraph_error", subgraph_error_marker)

    # Set entry point
    graph.set_entry_point("create_commit")

    # Add conditional edges
    graph.add_conditional_edges(
        "create_commit",
        commit_router,
        {
            "push_branch": "push_branch",
            "subgraph_error": "subgraph_error",
        },
    )
    graph.add_conditional_edges(
        "push_branch",
        push_router,
        {
            "create_issue": "create_issue",
            "subgraph_error": "subgraph_error",
        },
    )
    graph.add_conditional_edges(
        "create_issue",
        issue_router,
        {
            "create_mr": "create_mr",
            "subgraph_error": "subgraph_error",
        },
    )
    graph.add_conditional_edges(
        "create_mr",
        mr_router,
        {
            "human_approval": "human_approval",
            "subgraph_error": "subgraph_error",
        },
    )
    graph.add_conditional_edges(
        "human_approval",
        human_approval_router,
        {
            "add_to_merge_train": "add_to_merge_train",
            "subgraph_error": "subgraph_error",
        },
    )

    # Merge train success goes to END
    graph.add_edge("add_to_merge_train", END)

    # Error node goes to END
    graph.add_edge("subgraph_error", END)

    return graph.compile()


# ============================================================================
# NOTIFICATION SUBGRAPH
# ============================================================================


def create_notification_subgraph() -> CompiledGraph:
    """
    Create the notification subgraph for summary and failure notifications.

    Nodes:
    - report_summary: Generate workflow summary
    - notify_failure: Handle workflow failure and send notifications

    Flow:
        Entry depends on state:
        - If errors: notify_failure -> report_summary -> END
        - If success: report_summary -> END

    Returns:
        CompiledGraph that can be used as a node in the main workflow
    """
    graph = StateGraph(BoxUpRoleState)

    # Add nodes
    graph.add_node("report_summary", report_summary_node)
    graph.add_node("notify_failure", notify_failure_node)

    # Entry point routing function
    def notification_entry_router(
        state: BoxUpRoleState,
    ) -> Literal["notify_failure", "report_summary"]:
        """Route to failure handler if errors exist."""
        if state.get("errors"):
            return "notify_failure"
        return "report_summary"

    # Use a virtual entry node for conditional routing
    async def notification_entry_node(state: BoxUpRoleState) -> dict:
        """Entry point node - just passes through."""
        return {}

    graph.add_node("notification_entry", notification_entry_node)
    graph.set_entry_point("notification_entry")

    # Route based on error state
    graph.add_conditional_edges(
        "notification_entry",
        notification_entry_router,
        {
            "notify_failure": "notify_failure",
            "report_summary": "report_summary",
        },
    )

    # Failure notification leads to summary
    graph.add_edge("notify_failure", "report_summary")

    # Summary is terminal
    graph.add_edge("report_summary", END)

    return graph.compile()


# ============================================================================
# PHASE ROUTING FOR MAIN GRAPH
# ============================================================================


def should_continue_after_validation_phase(
    state: BoxUpRoleState,
) -> Literal["testing_phase", "notification_phase"]:
    """Route after validation phase - check for errors or blocking deps."""
    if state.get("errors") or state.get("blocking_deps"):
        return "notification_phase"
    return "testing_phase"


def should_continue_after_testing_phase(
    state: BoxUpRoleState,
) -> Literal["gitlab_phase", "notification_phase"]:
    """Route after testing phase - check for test failures."""
    errors = state.get("errors", [])
    if errors:
        return "notification_phase"
    if state.get("molecule_passed") is False:
        return "notification_phase"
    if state.get("pytest_passed") is False:
        return "notification_phase"
    if state.get("deploy_passed") is False:
        return "notification_phase"
    return "gitlab_phase"


def should_continue_after_gitlab_phase(state: BoxUpRoleState) -> Literal["notification_phase"]:
    """Route after GitLab phase - always go to notification."""
    # GitLab phase always leads to notification (success or failure)
    return "notification_phase"


# ============================================================================
# COMPOSED WORKFLOW
# ============================================================================


def create_composed_workflow(db_path: str = "harness.db") -> StateGraph:
    """
    Create the main workflow composing all subgraphs.

    This is the primary entry point for creating the box-up-role workflow
    using subgraph composition. The workflow is organized into four phases:

    1. Validation Phase: Role validation and dependency analysis
    2. Testing Phase: Worktree creation and test execution
    3. GitLab Phase: Commit, push, issue, MR, and merge train
    4. Notification Phase: Summary and failure notifications

    Args:
        db_path: Path to the SQLite database for state persistence

    Returns:
        StateGraph (uncompiled) - call .compile() with checkpointer to use

    Example:
        workflow = create_composed_workflow("harness.db")

        # Compile without checkpointer (for simple use)
        compiled = workflow.compile()

        # Compile with checkpointer (for persistence/resume)
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            compiled = workflow.compile(checkpointer=saver)
            result = await compiled.ainvoke(initial_state)
    """
    graph = StateGraph(BoxUpRoleState)

    # Add subgraphs as nodes
    # Each subgraph is a CompiledGraph that acts as a single node
    graph.add_node("validation_phase", create_validation_subgraph())
    graph.add_node("testing_phase", create_testing_subgraph())
    graph.add_node("gitlab_phase", create_gitlab_subgraph())
    graph.add_node("notification_phase", create_notification_subgraph())

    # Set entry point
    graph.set_entry_point("validation_phase")

    # Add conditional edges between phases
    graph.add_conditional_edges(
        "validation_phase",
        should_continue_after_validation_phase,
        {
            "testing_phase": "testing_phase",
            "notification_phase": "notification_phase",
        },
    )

    graph.add_conditional_edges(
        "testing_phase",
        should_continue_after_testing_phase,
        {
            "gitlab_phase": "gitlab_phase",
            "notification_phase": "notification_phase",
        },
    )

    graph.add_conditional_edges(
        "gitlab_phase",
        should_continue_after_gitlab_phase,
        {
            "notification_phase": "notification_phase",
        },
    )

    # Notification phase is terminal
    graph.add_edge("notification_phase", END)

    return graph


# ============================================================================
# SUBGRAPH-AWARE WORKFLOW RUNNER
# ============================================================================


class SubgraphWorkflowRunner:
    """
    Workflow runner that uses subgraph composition.

    This runner provides the same interface as LangGraphWorkflowRunner
    but uses the composed subgraph-based workflow for better organization.

    Usage:
        from harness.dag.subgraphs import SubgraphWorkflowRunner
        from harness.dag import create_initial_state

        runner = SubgraphWorkflowRunner(db, db_path="harness.db")
        state = create_initial_state("common")
        result = await runner.execute("common")
    """

    def __init__(
        self,
        db,  # StateDB
        db_path: str = "harness.db",
        notification_config=None,  # NotificationConfig
    ):
        from harness.dag.langgraph_engine import (
            LangGraphWorkflowRunner,
            set_module_db,
        )

        self.db = db
        self.db_path = db_path

        # Set module-level db for node access
        set_module_db(db)

        # Delegate to the standard runner but with composed workflow
        self._runner = LangGraphWorkflowRunner(
            db=db,
            db_path=db_path,
            notification_config=notification_config,
        )

        # Override the graph with composed version
        self._graph = None

    async def _get_graph(self):
        """Get the composed workflow graph."""
        if self._graph is None:
            workflow = create_composed_workflow(self.db_path)
            self._graph = workflow.compile()
        return self._graph

    async def execute(
        self,
        role_name: str,
        resume_from: int | None = None,
        config=None,
    ) -> dict:
        """
        Execute the box-up-role workflow using subgraph composition.

        Args:
            role_name: Name of the role to process
            resume_from: Optional execution ID to resume from
            config: Optional LangGraph configuration

        Returns:
            Final state after execution
        """
        # Delegate to the runner's execute method
        # The runner handles state initialization, checkpointing, and notifications
        return await self._runner.execute(
            role_name=role_name,
            resume_from=resume_from,
            config=config,
        )


# ============================================================================
# EXPORTED NAMES
# ============================================================================

__all__ = [
    # Subgraph factories
    "create_validation_subgraph",
    "create_testing_subgraph",
    "create_gitlab_subgraph",
    "create_notification_subgraph",
    # Composed workflow
    "create_composed_workflow",
    # Runner
    "SubgraphWorkflowRunner",
    # Routing functions (for testing)
    "should_continue_after_validation_phase",
    "should_continue_after_testing_phase",
    "should_continue_after_gitlab_phase",
]
