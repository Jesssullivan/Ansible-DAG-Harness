# Vignette 1: Basic Workflow Execution

This vignette demonstrates executing the box-up-role workflow for a single Ansible role.

## Scenario

You need to package the `common` role for deployment, which includes:
- Creating a git worktree for isolated changes
- Running molecule tests to validate the role
- Creating a GitLab issue to track the work
- Creating a merge request with the changes

## Prerequisites

1. **Install harness**
   ```bash
   cd /path/to/project
   harness install run
   ```

2. **Configure GitLab authentication**
   ```bash
   # Option 1: Use glab CLI
   glab auth login

   # Option 2: Set environment variable
   export GITLAB_TOKEN=glpat-xxxxx
   ```

3. **Verify installation**
   ```bash
   harness install check
   ```

## Steps

### Step 1: Sync roles from filesystem

First, ensure the harness database knows about your roles:

```bash
harness sync --roles
```

### Step 2: Check role status

View the current status of the role:

```bash
harness status common
```

Expected output:
```
common (Wave 1: Foundation)
  Worktree: None
  Issue: None - N/A
  MR: None - N/A
  Tests: 0 passed, 0 failed
```

### Step 3: Execute the workflow

Run the box-up-role workflow:

```bash
# Via CLI
harness box-up-role common

# Or via MCP client skill
/box-up-role common
```

### Step 4: Monitor progress

The workflow outputs progress in real-time:

```
Starting box-up-role workflow for: common
  → create_worktree
  ✓ create_worktree
  → run_molecule_tests
  ✓ run_molecule_tests
  → create_gitlab_issue
  ✓ create_gitlab_issue
  → create_merge_request
  ✓ create_merge_request
  → generate_summary
  ✓ generate_summary

Workflow completed successfully!

  Issue URL: https://gitlab.com/project/issues/123
  MR URL: https://gitlab.com/project/merge_requests/456
  Worktree: .worktrees/sid-common
```

### Step 5: Verify results

Check the final status:

```bash
harness status common
```

Expected output:
```
common (Wave 1: Foundation)
  Worktree: active
    0 ahead, 0 behind
  Issue: opened - https://gitlab.com/project/issues/123
  MR: opened - https://gitlab.com/project/merge_requests/456
  Tests: 5 passed, 0 failed
```

## Dry Run Mode

To preview what the workflow would do without making changes:

```bash
harness box-up-role common --dry-run
```

## Breakpoints

To pause the workflow at specific nodes for inspection:

```bash
harness box-up-role common --breakpoints run_molecule_tests
```

This pauses before running tests. Resume with:

```bash
harness resume <execution-id>
```

## Troubleshooting

### "Role not found"
Ensure you're in the project root and the role exists:
```bash
ls ansible/roles/common
```

### "GitLab authentication missing"
Run `glab auth login` or set `GITLAB_TOKEN`.

### Molecule tests fail
Check test output and fix issues, then resume:
```bash
harness resume <execution-id>
```
