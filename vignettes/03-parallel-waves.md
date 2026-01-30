# Vignette 3: Parallel Wave Execution

This vignette demonstrates processing multiple roles in parallel using the wave system.

## Scenario

You have roles organized into waves based on dependencies:

| Wave | Roles | Description |
|------|-------|-------------|
| 1 | common, windows_prerequisites | Foundation |
| 2 | iis-config, sql_server | Infrastructure |
| 3 | ems_webapp, ems_platform | Applications |
| 4 | grafana_alloy, monitoring | Observability |

You want to process all roles in a wave concurrently.

## Prerequisites

1. **Install harness**
   ```bash
   harness install run
   ```

2. **Sync roles with wave assignments**
   ```bash
   harness sync --roles
   ```

3. **Verify wave assignments**
   ```bash
   harness list-roles --wave 1
   ```

## Steps

### Step 1: View role dependencies

Check dependencies before parallel execution:

```bash
# Dependencies of a role
harness deps common

# What depends on this role
harness deps common --reverse
```

### Step 2: Execute Wave 1 (Foundation)

Process all Wave 1 roles in parallel:

```bash
# List wave 1 roles
harness list-roles --wave 1

# Execute each in parallel (in MCP client)
# Use multiple Task tool calls in parallel
```

**Via CLI:**
```bash
harness box-up-role common &
harness box-up-role windows_prerequisites &
wait
```

### Step 3: Verify Wave 1 completion

Before proceeding to Wave 2:

```bash
harness status common
harness status windows_prerequisites
```

All should show:
- Worktree: active
- Tests: passed
- MR: opened or merged

### Step 4: Execute Wave 2 (Infrastructure)

```bash
harness list-roles --wave 2

harness box-up-role iis-config &
harness box-up-role sql_server &
wait
```

### Step 5: Continue through waves

Repeat for waves 3 and 4.

## Automated Wave Execution

Use HOTL mode with wave filtering:

```bash
# Process only wave 1 roles autonomously
harness hotl start \
  --max-iterations 10 \
  --wave 1 \
  --notify-interval 300
```

## Parallel Execution in MCP client

When using MCP client, request parallel execution:

```
Process all Wave 1 roles in parallel using the box-up-role skill
```

MCP client will:
1. Identify Wave 1 roles
2. Launch parallel Task agents
3. Monitor progress concurrently
4. Report combined results

## Wave Dependencies

The wave system ensures:
- Wave N+1 roles can depend on Wave N
- Roles in the same wave are independent
- Parallel execution is safe within a wave

```
Wave 1: common, windows_prerequisites
        ↓
Wave 2: iis-config (depends: common, windows_prerequisites)
        sql_server (depends: common)
        ↓
Wave 3: ems_webapp (depends: iis-config)
        ems_platform (depends: iis-config, sql_server)
        ↓
Wave 4: grafana_alloy (depends: all above)
```

## Monitoring Parallel Execution

### Check all worktrees

```bash
harness worktrees
```

### View overall status

```bash
harness status
```

### Check metrics

```bash
harness metrics status
```

## Handling Failures

If a role in a wave fails:

1. **Other roles continue** - Parallel execution doesn't stop
2. **Dependent waves wait** - Can't proceed until fixed
3. **Resume failed role** - Fix and resume

```bash
# See what failed
harness status --wave 2

# Fix and resume
harness resume <execution-id>

# Or restart
harness box-up-role iis-config
```

## Example: Full Pipeline

```bash
#!/bin/bash
# Process all waves sequentially, roles in parallel

for wave in 1 2 3 4; do
  echo "Processing Wave $wave..."

  # Get roles in this wave
  roles=$(harness list-roles --wave $wave --json | jq -r '.[].name')

  # Execute in parallel
  for role in $roles; do
    harness box-up-role "$role" &
  done

  # Wait for all to complete
  wait

  echo "Wave $wave complete"
done

echo "All waves complete"
```

## Troubleshooting

### "Resource conflict" errors
Reduce parallelism or ensure roles are truly independent.

### Molecule tests conflict
Some tests share IIS resources. Use wave system to isolate.

### Git conflicts
Each role uses its own worktree, so conflicts are rare.
If they occur, resolve manually and resume.
