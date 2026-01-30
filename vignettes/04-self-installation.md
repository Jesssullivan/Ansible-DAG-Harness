# Vignette 4: Self-Installation

This vignette demonstrates the self-installing capabilities of the DAG harness.

## Scenario

You've cloned the disposable-dag-langchain repository and want to set up the complete MCP client integration in a new project.

## What Gets Installed

The `harness install run` command sets up:

| Component | Location | Purpose |
|-----------|----------|---------|
| MCP Server | `.claude/settings.json` | Database queries, status |
| PreToolUse Hooks | `.claude/hooks/` | Validation, rate limiting |
| PostToolUse Hooks | `.claude/hooks/` | Notifications, audit logging |
| Skills | `.claude/skills/` | Slash commands |

## Quick Install

```bash
# One command installation
cd /path/to/your/project
harness install run
```

That's it! The harness is now integrated with MCP client.

## Step-by-Step Installation

### Step 1: Navigate to project root

```bash
cd /path/to/your/project
```

### Step 2: Verify prerequisites

```bash
# Check Python version
python3 --version  # Requires 3.11+

# Check uv is installed
uv --version
```

### Step 3: Install harness package

```bash
cd harness
uv sync
cd ..
```

### Step 4: Run installation

```bash
harness install run
```

Output:
```
Installation successful!

  OK directories
  OK mcp_server
  OK hook:validate-box-up-env.sh
  OK hook:notify-box-up-status.sh
  OK hook:rate-limiter-hook.py
  OK hook:universal-hook.py
  OK hook:audit-logger.py
  OK skill:box-up-role
  OK skill:hotl
  OK skill:observability
```

### Step 5: Verify installation

```bash
harness install check
```

Output:
```
Installation Status: COMPLETE

  OK directories
       /path/to/project/.claude
  OK mcp_server
       /path/to/project/.claude/settings.json
  OK hook:validate-box-up-env.sh
  OK hook:notify-box-up-status.sh
  OK hook:rate-limiter-hook.py
  OK hook:universal-hook.py
  OK hook:audit-logger.py
  OK skill:box-up-role
  OK skill:hotl
  OK skill:observability
```

## Installation Options

### Skip components

```bash
# Skip hooks (use existing hooks)
harness install run --skip-hooks

# Skip skills (use existing skills)
harness install run --skip-skills

# Skip MCP server config
harness install run --skip-mcp
```

### Force overwrite

```bash
# Overwrite existing files
harness install run --force
```

## Upgrading

When the harness is updated:

```bash
harness install upgrade
```

This:
1. Backs up custom settings
2. Re-installs hooks and skills
3. Restores custom permissions

## Uninstalling

```bash
# Remove harness integration
harness install uninstall

# Also remove data files
harness install uninstall --remove-data
```

## What Each Component Does

### MCP Server
Provides MCP client with tools:
- `get_role_status` - Query role deployment state
- `sync_roles_from_filesystem` - Refresh role data
- `get_active_regressions` - Find test failures
- 20+ other tools

### PreToolUse Hooks
Run before tool execution:
- **validate-box-up-env.sh** - Check GitLab auth, KeePass
- **rate-limiter-hook.py** - Prevent API overuse
- **universal-hook.py** - Route to sub-hooks

### PostToolUse Hooks
Run after tool execution:
- **notify-box-up-status.sh** - Discord/email on failure
- **audit-logger.py** - Record all tool calls

### Skills
Slash commands in MCP client:
- `/box-up-role <name>` - Package a role
- `/hotl start` - Autonomous operation
- `/observability debug-platform-health` - Diagnostics

## Customization

### Add custom permissions

Edit `.claude/settings.json`:
```json
{
  "permissions": {
    "allow": [
      "Bash(harness *)",
      "Bash(your-custom-command *)"
    ]
  }
}
```

### Disable rate limiting

In `.claude/settings.json`, modify hooks section:
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [...]
      }
      // Remove or comment out rate limiter hook
    ]
  }
}
```

### Add custom hook

Create `.claude/hooks/my-custom-hook.sh`:
```bash
#!/bin/bash
# My custom PreToolUse hook
exit 0  # Allow
```

Update `universal-hook.py` to include it.

## Troubleshooting

### "Permission denied" on hooks
```bash
chmod +x .claude/hooks/*.sh .claude/hooks/*.py
```

### MCP server not starting
```bash
# Test manually
uv run --directory ./harness python -m harness.mcp.server
```

### Skills not showing
Restart MCP client after installation.

### Partial installation
```bash
harness install run --force
```
