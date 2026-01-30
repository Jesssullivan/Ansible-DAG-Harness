# Architecture

This document describes the architecture of the DAG Harness.

## Overview

The harness is a DAG-based workflow orchestration system built on:

- **LangGraph** - State machine execution with checkpointing
- **SQLite** - Persistent state and history
- **FastMCP** - MCP client integration via MCP protocol
- **Typer/Rich** - CLI interface

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP client                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Skills    │  │    Hooks    │  │     MCP Server      │ │
│  │ /box-up-role│  │ PreToolUse  │  │    (dag-harness)    │ │
│  │ /hotl       │  │ PostToolUse │  │                     │ │
│  └─────────────┘  └─────────────┘  └──────────┬──────────┘ │
└───────────────────────────────────────────────┼─────────────┘
                                                │
┌───────────────────────────────────────────────┼─────────────┐
│                     Harness Core              │             │
│  ┌────────────────────────────────────────────▼───────────┐│
│  │                    CLI (Typer)                         ││
│  │  box-up-role │ status │ sync │ hotl │ bootstrap        ││
│  └────────────────────────────────────────────────────────┘│
│                            │                                │
│  ┌─────────────────────────┼─────────────────────────────┐ │
│  │              DAG Engine (LangGraph)                    │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │ │
│  │  │ validate │→│ analyze  │→│ worktree │→ ...        │ │
│  │  └──────────┘  └──────────┘  └──────────┘            │ │
│  │              SqliteSaver (checkpointing)              │ │
│  └───────────────────────────────────────────────────────┘ │
│                            │                                │
│  ┌─────────────────────────┼─────────────────────────────┐ │
│  │                   StateDB (SQLite)                     │ │
│  │  roles │ worktrees │ test_runs │ executions │ metrics │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### DAG Engine (LangGraph)

The workflow execution engine uses LangGraph for:

- **State management** - TypedDict-based state with Annotated reducers
- **Conditional routing** - Smart branching based on test results
- **Checkpointing** - SqliteSaver for resumable workflows
- **Async execution** - All nodes are async functions

```python
class BoxUpRoleState(TypedDict, total=False):
    role_name: str
    execution_id: Optional[int]
    worktree_path: str
    molecule_passed: bool
    mr_url: Optional[str]
    errors: Annotated[list[str], operator.add]
    # ...
```

Nodes are async functions that transform state:

```python
async def validate_role_node(state: BoxUpRoleState) -> BoxUpRoleState:
    # Validate role exists, extract metadata
    return {"role_path": str(role_path), "has_molecule_tests": True}
```

### State Database (SQLite)

Persistent storage for:

- **Roles** - Ansible role metadata
- **Worktrees** - Git worktree tracking
- **Test runs** - Test history and regression detection
- **Workflow executions** - Execution history
- **Golden metrics** - Performance and quality metrics

Key features:

- Regression detection for test failures
- Dependency graph tracking
- Checkpoint storage for resumption

### MCP Server (FastMCP)

Provides tools for MCP client integration:

```python
@mcp.tool()
def get_role_status(role_name: str) -> dict:
    """Get the status of a role."""
    ...

@mcp.tool()
def run_molecule_tests(role_name: str) -> dict:
    """Execute molecule tests for a role."""
    ...
```

20+ tools for role management, testing, GitLab integration, etc.

### Bootstrap System

Self-installing setup that:

1. Detects environment
2. Discovers credentials
3. Validates paths
4. Initializes database
5. Installs MCP client integration
6. Runs self-tests

### HOTL Supervisor

Human Out of The Loop autonomous operation:

- Worker pool for parallel execution
- Checkpoint/resume capability
- Discord/email notifications
- Configurable iteration limits

## Workflow: box-up-role

The primary workflow for packaging an Ansible role:

```
validate_role
    │
    ▼
analyze_deps ──────────────────────────┐
    │                                  │
    ▼                                  │
check_reverse_deps ───► [blocked] ────►│
    │                                  │
    ▼                                  │
create_worktree                        │
    │                                  │
    ▼                                  │
run_molecule ──────► [fail] ──────────►│
    │                                  │
    ▼                                  │
run_pytest ────────► [fail] ──────────►│
    │                                  │
    ▼                                  │
validate_deploy ───► [fail] ──────────►│
    │                                  ▼
    ▼                           notify_failure
create_commit                          │
    │                                  │
    ▼                                  │
push_branch                            │
    │                                  │
    ▼                                  │
create_issue                           │
    │                                  │
    ▼                                  │
create_mr                              │
    │                                  │
    ▼                                  │
add_to_merge_train                     │
    │                                  │
    ▼                                  │
report_summary ◄───────────────────────┘
    │
    ▼
   END
```

## Database Schema

Key tables:

```sql
CREATE TABLE roles (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    wave INTEGER DEFAULT 0,
    has_molecule_tests BOOLEAN,
    description TEXT
);

CREATE TABLE worktrees (
    id INTEGER PRIMARY KEY,
    role_id INTEGER REFERENCES roles(id),
    path TEXT NOT NULL,
    branch TEXT NOT NULL,
    status TEXT
);

CREATE TABLE test_runs (
    id INTEGER PRIMARY KEY,
    role_id INTEGER REFERENCES roles(id),
    test_name TEXT NOT NULL,
    test_type TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    error_message TEXT,
    run_at TEXT NOT NULL
);

CREATE TABLE workflow_executions (
    id INTEGER PRIMARY KEY,
    role_id INTEGER REFERENCES roles(id),
    status TEXT NOT NULL,
    current_node TEXT,
    started_at TEXT,
    completed_at TEXT,
    checkpoint BLOB
);
```

## Extension Points

### Custom Nodes

Add custom nodes to the workflow:

```python
from harness.dag.langgraph_engine import BoxUpRoleState

async def custom_node(state: BoxUpRoleState) -> BoxUpRoleState:
    # Custom logic
    return {"custom_field": "value"}
```

### MCP Tools

Add custom MCP tools:

```python
@mcp.tool()
def my_custom_tool(arg: str) -> dict:
    """Description of the tool."""
    return {"result": "value"}
```

### Hooks

Add custom hooks in `.claude/hooks/`:

- `PreToolUse` - Validate before tool execution
- `PostToolUse` - React after tool execution

## Configuration

Configuration via `harness.yml` or environment variables:

```yaml
gitlab:
  project_path: "group/project"
  assignee_ids: [123]

worktree:
  base_path: "../worktrees"
  branch_prefix: "sid/"

testing:
  molecule_timeout: 600
  pytest_timeout: 300

waves:
  0:
    name: "Foundation"
    roles: ["common"]
  1:
    name: "Infrastructure"
    roles: ["nginx", "docker"]
```
