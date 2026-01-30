# MCP Tools Reference

The DAG Harness MCP server provides tools for MCP client integration.

## Server Configuration

The MCP server is configured in `.claude/settings.json`:

```json
{
  "mcpServers": {
    "dag-harness": {
      "command": "uv",
      "args": ["run", "--directory", "./harness", "python", "-m", "harness.mcp.server"],
      "env": {
        "HARNESS_DB_PATH": "./harness/harness.db"
      }
    }
  }
}
```

## Available Tools

### Role Management

#### get_role_status

Get the status of a specific role.

```
get_role_status(role_name: str) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the Ansible role |

**Returns:** Role status including wave, worktree, issue, MR, and test information.

#### list_role_statuses

List statuses for all roles or filtered by wave.

```
list_role_statuses(wave: Optional[int] = None) -> list[dict]
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `wave` | int (optional) | Filter by wave number |

**Returns:** List of role status objects.

#### sync_roles_from_filesystem

Scan filesystem and update role database.

```
sync_roles_from_filesystem() -> dict
```

**Returns:** Count of added and updated roles.

### Dependency Analysis

#### get_role_dependencies

Get dependencies for a role.

```
get_role_dependencies(role_name: str, transitive: bool = False) -> list[str]
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `transitive` | bool | Include transitive dependencies |

**Returns:** List of dependency role names.

#### get_reverse_dependencies

Get roles that depend on a given role.

```
get_reverse_dependencies(role_name: str, transitive: bool = False) -> list[str]
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `transitive` | bool | Include transitive reverse dependencies |

**Returns:** List of dependent role names.

### Test Tracking

#### get_active_regressions

Find test regressions (previously passing tests that now fail).

```
get_active_regressions() -> list[dict]
```

**Returns:** List of regression information including role, test name, and failure details.

#### record_test_result

Record a test execution result.

```
record_test_result(
    role_name: str,
    test_name: str,
    test_type: str,
    passed: bool,
    error_message: Optional[str] = None
) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `test_name` | str | Name of the test |
| `test_type` | str | Type: "molecule", "pytest", "deploy" |
| `passed` | bool | Whether the test passed |
| `error_message` | str (optional) | Error message if failed |

**Returns:** Confirmation with regression detection status.

### Git Worktree Operations

#### create_worktree

Create a git worktree for a role.

```
create_worktree(role_name: str, branch_name: Optional[str] = None) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `branch_name` | str (optional) | Custom branch name |

**Returns:** Worktree path and branch information.

#### list_worktrees

List all git worktrees.

```
list_worktrees() -> list[dict]
```

**Returns:** List of worktree information including path, branch, and status.

#### remove_worktree

Remove a git worktree.

```
remove_worktree(role_name: str, force: bool = False) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `force` | bool | Force removal even if dirty |

**Returns:** Confirmation of removal.

### Test Execution

#### run_molecule_tests

Execute molecule tests for a role.

```
run_molecule_tests(
    role_name: str,
    scenario: str = "default",
    timeout: int = 600
) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `scenario` | str | Molecule scenario name |
| `timeout` | int | Timeout in seconds |

**Returns:** Test results including passed status and output.

#### run_pytest_tests

Execute pytest tests for a role.

```
run_pytest_tests(
    role_name: str,
    test_path: Optional[str] = None,
    timeout: int = 300
) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `test_path` | str (optional) | Specific test path |
| `timeout` | int | Timeout in seconds |

**Returns:** Test results including passed status and output.

### GitLab Integration

#### create_gitlab_issue

Create a GitLab issue for a role.

```
create_gitlab_issue(
    role_name: str,
    title: str,
    description: str,
    labels: Optional[list[str]] = None
) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `title` | str | Issue title |
| `description` | str | Issue description |
| `labels` | list[str] (optional) | Issue labels |

**Returns:** Issue URL and IID.

#### create_merge_request

Create a GitLab merge request.

```
create_merge_request(
    role_name: str,
    source_branch: str,
    title: str,
    description: str
) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `source_branch` | str | Source branch name |
| `title` | str | MR title |
| `description` | str | MR description |

**Returns:** MR URL and IID.

### Metrics

#### record_metric

Record a golden metric value.

```
record_metric(name: str, value: float, context: Optional[dict] = None) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `name` | str | Metric name |
| `value` | float | Metric value |
| `context` | dict (optional) | Additional context |

**Returns:** Metric status (ok, warning, critical).

#### get_metric_health

Get health status of all metrics.

```
get_metric_health() -> dict
```

**Returns:** Overall health and per-metric status.

### Workflow Execution

#### execute_workflow

Execute a workflow for a role.

```
execute_workflow(
    role_name: str,
    workflow: str = "box-up-role",
    breakpoints: Optional[list[str]] = None
) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `role_name` | str | Name of the role |
| `workflow` | str | Workflow name |
| `breakpoints` | list[str] (optional) | Nodes to pause before |

**Returns:** Execution result with status and summary.

#### resume_workflow

Resume a paused workflow.

```
resume_workflow(execution_id: int) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `execution_id` | int | Execution ID to resume |

**Returns:** Execution result.

### Database Operations

#### get_database_stats

Get database statistics.

```
get_database_stats() -> dict
```

**Returns:** Table counts and database size.

#### validate_database

Validate database schema and data integrity.

```
validate_database() -> dict
```

**Returns:** Validation results for schema, data, and dependencies.

## Error Handling

All tools return errors in a consistent format:

```json
{
  "error": true,
  "message": "Description of the error",
  "details": { ... }
}
```

## Rate Limiting

The MCP server includes rate limiting via the `universal-hook.py` hook.
Default limits prevent excessive API calls to GitLab and other services.
