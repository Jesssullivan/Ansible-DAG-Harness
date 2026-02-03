---
name: box-up-role
description: Package an Ansible role into a GitLab iteration with worktree, issues, and MR
user-invocable: true
argument-hint: <role-name>
allowed-tools: Read, Grep, Glob, Bash, Task, Write, Edit
---

# Box Up Role Workflow

You are packaging the Ansible role `$ARGUMENTS` for GitLab iteration management.

## Harness Integration

The dag-harness LangGraph engine handles the full workflow. The `harness` CLI is the
primary entry point; script-based fallback is available when the harness package is
not installed.

---

## Step 1: Check Harness Availability

Run a version check to determine whether the harness CLI is available.

```bash
harness --version 2>/dev/null
```

**If harness is available** (exit 0): proceed to Step 2.

**If harness is NOT available** (exit non-zero): install it first:

```bash
pip install dag-harness
# or: uv tool install dag-harness
```

---

## Step 2: Pre-Flight Validation

Before running the workflow, verify basic prerequisites.

```bash
# Verify role exists
ls -d ansible/roles/$ARGUMENTS/ || { echo "ERROR: Role '$ARGUMENTS' not found"; exit 1; }

# Verify glab authentication
glab auth status

# Check SSH tunnels for tunneled targets
npm run tunnels:status
```

You can also use the harness to check role status and dependencies:

```bash
harness status $ARGUMENTS
harness deps $ARGUMENTS --transitive
harness deps $ARGUMENTS --reverse
```

---

## Step 3: Run the Workflow

Execute the full box-up-role DAG workflow. The harness orchestrates all steps:
analyze, validate dependencies, create worktree, run tests, commit, push, create
issue, create MR, and request human approval.

```bash
harness box-up-role $ARGUMENTS
```

**Dry run** (show what would happen without making changes):

```bash
harness box-up-role $ARGUMENTS --dry-run
```

**With breakpoints** (pause before specific nodes for inspection):

```bash
harness box-up-role $ARGUMENTS --breakpoints "create_commit,push_branch"
```

---

## Step 4: Parse Results

The workflow produces structured output. Watch for these result statuses:

### Completed Successfully

The workflow prints:
- `Issue URL: <url>` -- the GitLab issue created for this role
- `MR URL: <url>` -- the merge request with closing pattern
- `Worktree: <path>` -- the isolated worktree path

Record these URLs for the summary report.

### Paused at Human Approval

When the workflow reaches the `human_approval` node, it pauses with an
`interrupt()` and prints:

```
Workflow paused at: human_approval
Resume with: harness resume <execution-id>
```

The execution ID is a numeric identifier. Note it for the next step.

### Paused at Breakpoint

If you used `--breakpoints`, the workflow pauses before the named node:

```
Workflow paused at: <node-name>
Resume with: harness resume <execution-id>
```

### Failed

On failure, the workflow prints:
```
Workflow failed: <error-message>
Failed at node: <node-name>
```

Check the error, fix the issue, and re-run the workflow.

---

## Step 5: Handle Pauses (Human-in-the-Loop)

If the workflow paused at `human_approval`, review the current state and
ask the user whether to approve or reject.

**Show current state:**

```bash
harness status $ARGUMENTS
```

**Approve and continue to merge train:**

```bash
harness resume <execution-id> --approve
```

**Reject with a reason:**

```bash
harness resume <execution-id> --reject --reason "Tests need more coverage"
```

**Resume from a breakpoint (no approval needed):**

```bash
harness resume <execution-id>
```

You can also set additional breakpoints on resume:

```bash
harness resume <execution-id> --breakpoints "create_mr"
```

---

## Step 6: Report Summary

After the workflow completes (either directly or after resume), present the
user with a formatted summary:

```
==========================================
BOX UP ROLE COMPLETE: $ARGUMENTS
==========================================

Worktree Path: ../sid-$ARGUMENTS
Branch: sid/$ARGUMENTS
Wave: <wave_number> (<wave_name>)

GitLab Issue: <issue_url>
Merge Request: <mr_url>
Iteration: <iteration_name>

Test Commands:
  cd ../sid-$ARGUMENTS
  npm run molecule:role --role=$ARGUMENTS
  npm run deploy:<target> -- --tags <role-tags>

Next Steps:
1. Review MR in GitLab
2. Request code review
3. Merge when approved (auto-squash)
==========================================
```

---

## Harness CLI Reference

| Command | Purpose |
|---------|---------|
| `harness --version` | Check harness availability |
| `harness box-up-role <role>` | Execute full DAG workflow |
| `harness box-up-role <role> --dry-run` | Show what would happen |
| `harness box-up-role <role> --breakpoints <nodes>` | Pause at specific nodes |
| `harness resume <id>` | Resume paused workflow |
| `harness resume <id> --approve` | Approve at human_approval node |
| `harness resume <id> --reject --reason "..."` | Reject with reason |
| `harness status [role]` | Show role/workflow status |
| `harness deps <role> --transitive` | Show dependency tree |
| `harness deps <role> --reverse` | Show reverse dependencies |
| `harness list-roles --wave N` | List roles by wave |
| `harness sync --gitlab` | Sync state from GitLab |
| `harness graph` | Show workflow DAG |
| `harness check` | Run self-checks |

---

## Workflow DAG Nodes

The box-up-role graph executes these nodes in order:

1. **analyze_role** -- Extract dependencies, credentials, wave placement
2. **check_reverse_deps** -- Block if reverse dependencies are not boxed up
3. **create_worktree** -- Create role branch and worktree from origin/main
4. **run_tests** -- Execute molecule tests (blocks on failure)
5. **deploy_test** -- Deploy to test target (blocks on failure)
6. **create_commit** -- Create signed commit
7. **push_branch** -- Push to origin with tracking
8. **create_issue** -- Create GitLab issue with iteration assignment
9. **create_mr** -- Create merge request with closing pattern
10. **human_approval** -- Pause with `interrupt()` for human review
11. **merge_train** -- Add to merge train (on approval)

Visualize the graph:

```bash
harness graph
harness graph --format mermaid
harness graph --format json
```

---

## Error Handling

### Role Not Found
```
ERROR: Role '$ARGUMENTS' not found in ansible/roles/
Usage: /box-up-role <role-name>
```

### Reverse Dependency Blocking
```
ERROR: Cannot box up '$ARGUMENTS' yet.
The following roles depend on '$ARGUMENTS' and must be boxed up first.
```

### Molecule Test Failure
```
ERROR: Molecule tests FAILED
Fix the failing tests before proceeding.
```

### GitLab API Error
```
ERROR: GitLab API request failed.
Check: glab auth status, network connectivity, rate limits.
```

### Workflow Resume Failure
```
Resume failed: <error>
```
Check `harness status $ARGUMENTS` and the execution state in the database.

---

## Script-Based Fallback

When the harness CLI is not installed, use the legacy script-based workflow.
This requires the scripts in the `scripts/` directory.

### Analyze Dependencies

```bash
python scripts/analyze-role-deps.py $ARGUMENTS --json
```

### Create Worktree

```bash
scripts/create-role-worktree.sh $ARGUMENTS
scripts/create-role-worktree.sh $ARGUMENTS --force
```

### Run Tests

```bash
scripts/validate-role-tests.sh $ARGUMENTS
scripts/validate-role-tests.sh $ARGUMENTS --molecule-only
```

### Create GitLab Issue and MR

```bash
scripts/create-gitlab-issues.sh $ARGUMENTS
scripts/create-gitlab-mr.sh $ARGUMENTS --issue <issue_iid>
```

---

## Configuration

See `.claude/skills/box-up-role/config.yml` for:
- GitLab project settings
- Iteration cadence name
- Default assignee
- Label mappings
- Worktree base path
- Testing requirements

Templates for issues and MRs are in `.claude/skills/box-up-role/templates/`.
