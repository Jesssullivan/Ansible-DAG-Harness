"""Pydantic models for database entities."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# Custom exceptions
class CyclicDependencyError(Exception):
    """Raised when a cyclic dependency is detected in the role graph."""

    def __init__(self, message: str, cycle_path: list[str] | None = None):
        super().__init__(message)
        self.cycle_path = cycle_path or []

    def __str__(self) -> str:
        if self.cycle_path:
            return f"{self.args[0]}"
        return super().__str__()


class DependencyType(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CREDENTIAL = "credential"


class WorktreeStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    DIRTY = "dirty"
    MERGED = "merged"
    PRUNED = "pruned"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TestType(str, Enum):
    MOLECULE = "molecule"
    PYTEST = "pytest"
    DEPLOY = "deploy"


class TestStatus(str, Enum):
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Role(BaseModel):
    """Ansible role with wave placement and metadata."""

    id: int | None = None
    name: str
    wave: int = Field(ge=0, le=4)
    wave_name: str | None = None
    description: str | None = None
    molecule_path: str | None = None
    has_molecule_tests: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RoleDependency(BaseModel):
    """Dependency relationship between roles."""

    id: int | None = None
    role_id: int
    depends_on_id: int
    dependency_type: DependencyType
    source_file: str | None = None
    created_at: datetime | None = None


class Credential(BaseModel):
    """Credential requirement for a role."""

    id: int | None = None
    role_id: int
    entry_name: str
    purpose: str | None = None
    is_base58: bool = False
    attribute: str | None = None
    source_file: str | None = None
    source_line: int | None = None
    created_at: datetime | None = None


class Worktree(BaseModel):
    """Git worktree for parallel development."""

    id: int | None = None
    role_id: int
    path: str
    branch: str
    base_commit: str | None = None
    current_commit: str | None = None
    commits_ahead: int = 0
    commits_behind: int = 0
    uncommitted_changes: int = 0
    status: WorktreeStatus = WorktreeStatus.ACTIVE
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Iteration(BaseModel):
    """GitLab iteration."""

    id: int
    title: str | None = None
    state: str = "opened"
    start_date: str | None = None
    due_date: str | None = None
    group_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Issue(BaseModel):
    """GitLab issue."""

    id: int
    iid: int
    role_id: int | None = None
    iteration_id: int | None = None
    title: str
    state: str = "opened"
    web_url: str | None = None
    labels: str | None = None
    assignee: str | None = None
    weight: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MergeRequest(BaseModel):
    """GitLab merge request."""

    id: int
    iid: int
    role_id: int | None = None
    issue_id: int | None = None
    source_branch: str
    target_branch: str = "main"
    title: str
    state: str = "opened"
    web_url: str | None = None
    merge_status: str | None = None
    squash_on_merge: bool = True
    remove_source_branch: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class WorkflowDefinition(BaseModel):
    """DAG workflow definition."""

    id: int | None = None
    name: str
    description: str | None = None
    nodes_json: str
    edges_json: str
    created_at: datetime | None = None


class WorkflowExecution(BaseModel):
    """Workflow execution instance."""

    id: int | None = None
    workflow_id: int
    role_id: int
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_node: str | None = None
    checkpoint_data: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class NodeExecution(BaseModel):
    """Individual node execution within a workflow."""

    id: int | None = None
    execution_id: int
    node_name: str
    status: NodeStatus = NodeStatus.PENDING
    input_data: str | None = None
    output_data: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = 0
    created_at: datetime | None = None


class TestRun(BaseModel):
    """Test execution record."""

    id: int | None = None
    role_id: int
    worktree_id: int | None = None
    execution_id: int | None = None
    test_type: TestType
    status: TestStatus
    duration_seconds: int | None = None
    log_path: str | None = None
    output_json: str | None = None
    commit_sha: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None


class TestCase(BaseModel):
    """Individual test case result."""

    id: int | None = None
    test_run_id: int
    name: str
    status: str
    duration_ms: int | None = None
    error_message: str | None = None
    failure_output: str | None = None
    created_at: datetime | None = None


class RoleStatusView(BaseModel):
    """Aggregated role status from v_role_status view."""

    id: int
    name: str
    wave: int
    wave_name: str | None = None
    worktree_status: str | None = None
    commits_ahead: int | None = None
    commits_behind: int | None = None
    issue_state: str | None = None
    issue_url: str | None = None
    mr_state: str | None = None
    mr_url: str | None = None
    passed_tests: int = 0
    failed_tests: int = 0


# ============================================================================
# SEE/ACP CONTEXT CONTROL MODELS
# ============================================================================


class CapabilityStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


class ExecutionContext(BaseModel):
    """MCP client session execution context."""

    id: int | None = None
    session_id: str
    user_id: str | None = None
    request_id: str | None = None
    capabilities: str | None = None  # JSON array
    metadata: str | None = None  # JSON object
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None


class ContextCapability(BaseModel):
    """Fine-grained capability grant for a context."""

    id: int | None = None
    context_id: int
    capability: str  # e.g., 'write:roles', 'execute:molecule'
    scope: str | None = None  # Optional scope restriction
    granted_by: str = "system"
    granted_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None


class ToolInvocation(BaseModel):
    """Tool invocation tracking for telemetry."""

    id: int | None = None
    context_id: int | None = None
    tool_name: str
    arguments: str | None = None  # JSON object
    result: str | None = None  # JSON object or error
    status: str = "pending"
    duration_ms: int | None = None
    blocked_reason: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


# ============================================================================
# TEST REGRESSION MODELS
# ============================================================================


class RegressionStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    FLAKY = "flaky"
    KNOWN_ISSUE = "known_issue"


class TestRegression(BaseModel):
    """Test regression tracking."""

    id: int | None = None
    role_id: int
    test_name: str
    test_type: TestType | None = None
    first_failure_run_id: int | None = None
    resolved_run_id: int | None = None
    failure_count: int = 1
    consecutive_failures: int = 1
    last_failure_at: datetime | None = None
    last_error_message: str | None = None
    status: RegressionStatus = RegressionStatus.ACTIVE
    notes: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MergeTrainStatus(str, Enum):
    QUEUED = "queued"
    MERGING = "merging"
    MERGED = "merged"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MergeTrainEntry(BaseModel):
    """Merge train queue entry."""

    id: int | None = None
    mr_id: int
    position: int | None = None
    target_branch: str = "main"
    status: MergeTrainStatus = MergeTrainStatus.QUEUED
    pipeline_id: int | None = None
    pipeline_status: str | None = None
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ActiveRegressionView(BaseModel):
    """Active regression from v_active_regressions view."""

    id: int
    role_name: str
    wave: int
    test_name: str
    test_type: str | None = None
    failure_count: int
    consecutive_failures: int
    last_failure_at: datetime | None = None
    last_error_message: str | None = None
    status: str
    notes: str | None = None


# ============================================================================
# AGENT SESSION MODELS (HOTL Claude Code Integration)
# ============================================================================


class AgentSessionStatus(str, Enum):
    """Status of a Claude Code agent session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"
    CANCELLED = "cancelled"


class AgentFileChangeType(str, Enum):
    """Type of file change made by an agent."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


class AgentSessionModel(BaseModel):
    """Database model for agent sessions."""

    id: str  # UUID string
    execution_id: int | None = None
    task: str
    status: AgentSessionStatus = AgentSessionStatus.PENDING
    output: str | None = None
    error_message: str | None = None
    intervention_reason: str | None = None
    context_json: str | None = None
    progress_json: str | None = None
    working_dir: str
    pid: int | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AgentFileChange(BaseModel):
    """Database model for agent file changes."""

    id: int | None = None
    session_id: str
    file_path: str
    change_type: AgentFileChangeType
    diff: str | None = None
    old_path: str | None = None
    created_at: datetime | None = None


class AgentSessionView(BaseModel):
    """View model for agent sessions with computed fields."""

    id: str
    execution_id: int | None = None
    task: str
    status: str
    working_dir: str
    intervention_reason: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    pid: int | None = None
    file_change_count: int = 0
    duration_seconds: float | None = None
