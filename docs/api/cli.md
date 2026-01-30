# CLI Reference

The `harness` command-line interface provides all commands for managing the DAG harness.

## Global Usage

```bash
harness [OPTIONS] COMMAND [ARGS]
```

## Core Commands

### bootstrap

Bootstrap the harness with interactive setup.

```bash
harness bootstrap [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--check-only, -c` | Only check current state, don't make changes |
| `--quick, -q` | Quick check (skip network tests) |

**Examples:**

```bash
harness bootstrap              # Full interactive setup
harness bootstrap --check-only # Verify current state
harness bootstrap --quick      # Quick prerequisite check
```

### box-up-role

Execute the box-up-role workflow for a role.

```bash
harness box-up-role ROLE_NAME [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `ROLE_NAME` | Name of the Ansible role |

**Options:**

| Option | Description |
|--------|-------------|
| `--breakpoints, -b TEXT` | Comma-separated node names to pause before |
| `--dry-run` | Show what would be done without making changes |

**Examples:**

```bash
harness box-up-role nginx
harness box-up-role nginx --dry-run
harness box-up-role nginx --breakpoints run_molecule,create_mr
```

### status

Show status of roles and their deployments.

```bash
harness status [ROLE_NAME]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `ROLE_NAME` | Role name (optional, shows all if not specified) |

### sync

Sync state from filesystem and GitLab.

```bash
harness sync [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--roles/--no-roles` | Sync roles from filesystem (default: yes) |
| `--worktrees/--no-worktrees` | Sync worktrees from git (default: yes) |
| `--gitlab` | Sync issues/MRs from GitLab |

### list-roles

List all Ansible roles.

```bash
harness list-roles [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--wave, -w INTEGER` | Filter by wave number |

### deps

Show dependencies for a role.

```bash
harness deps ROLE_NAME [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--reverse, -r` | Show reverse dependencies |
| `--transitive, -t` | Include transitive dependencies |

### worktrees

List all git worktrees.

```bash
harness worktrees [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

### resume

Resume a paused workflow execution.

```bash
harness resume EXECUTION_ID [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--breakpoints, -b TEXT` | Comma-separated node names to pause before |

### graph

Show the box-up-role workflow graph.

```bash
harness graph [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--format, -f TEXT` | Output format: text, json, mermaid |

### check

Run self-checks on the harness database.

```bash
harness check [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--schema/--no-schema` | Validate schema |
| `--data/--no-data` | Validate data integrity |
| `--graph/--no-graph` | Validate dependency graph |
| `--json` | Output as JSON |

### init

Initialize harness configuration.

```bash
harness init [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--config, -c TEXT` | Path to save config (default: harness.yml) |

### mcp-server

Start the MCP server for MCP client integration.

```bash
harness mcp-server
```

## Database Commands

### db stats

Show database statistics.

```bash
harness db stats [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

### db info

Show database file information.

```bash
harness db info
```

### db backup

Create a backup of the database.

```bash
harness db backup OUTPUT_PATH
```

### db reset

Reset database to initial state (DESTRUCTIVE).

```bash
harness db reset [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--yes, -y` | Confirm without prompting |

### db clear

Clear a specific table.

```bash
harness db clear TABLE_NAME [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--yes, -y` | Confirm without prompting |

### db vacuum

Vacuum database to reclaim space.

```bash
harness db vacuum
```

## Metrics Commands

### metrics status

Show current status of all golden metrics.

```bash
harness metrics status [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

### metrics record

Record a metric value.

```bash
harness metrics record NAME VALUE [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--context, -c TEXT` | JSON context metadata |

### metrics history

Show recent history for a metric.

```bash
harness metrics history NAME [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--hours, -h INTEGER` | Hours to look back (default: 24) |
| `--limit, -n INTEGER` | Maximum entries to show (default: 20) |

### metrics trend

Show trend analysis for a metric.

```bash
harness metrics trend NAME [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--hours, -h INTEGER` | Hours to analyze (default: 24) |

### metrics list

List all registered golden metrics.

```bash
harness metrics list
```

### metrics purge

Purge old metric records.

```bash
harness metrics purge [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--days, -d INTEGER` | Purge records older than N days (default: 30) |
| `--yes, -y` | Confirm without prompting |

## HOTL Commands

### hotl start

Start HOTL autonomous operation mode.

```bash
harness hotl start [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--max-iterations, -n INTEGER` | Maximum iterations (default: 100) |
| `--notify-interval, -i INTEGER` | Seconds between notifications (default: 300) |
| `--discord TEXT` | Discord webhook URL |
| `--email TEXT` | Email recipient |
| `--background, -b` | Run in background |

### hotl status

Show current HOTL status.

```bash
harness hotl status [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

### hotl stop

Request HOTL to stop gracefully.

```bash
harness hotl stop [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--force, -f` | Force stop (cancel running executions) |

### hotl resume

Resume HOTL from a checkpoint.

```bash
harness hotl resume THREAD_ID [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--max-iterations, -n INTEGER` | Maximum iterations |
| `--notify-interval, -i INTEGER` | Notification interval |

## Install Commands

### install run

Install harness into MCP client.

```bash
harness install run [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--force, -f` | Overwrite existing files |
| `--skip-hooks` | Don't install hook scripts |
| `--skip-skills` | Don't install skill definitions |
| `--skip-mcp` | Don't configure MCP server |

### install check

Verify MCP client installation status.

```bash
harness install check
```

### install uninstall

Remove harness from MCP client.

```bash
harness install uninstall [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--yes, -y` | Confirm without prompting |
| `--remove-data` | Also remove data files |

### install upgrade

Upgrade existing MCP client installation.

```bash
harness install upgrade
```
