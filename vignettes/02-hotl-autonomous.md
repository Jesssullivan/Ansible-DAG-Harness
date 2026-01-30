# Vignette 2: HOTL Autonomous Operation

This vignette demonstrates using Human Out of The Loop (HOTL) mode for extended autonomous operation.

## Scenario

You need to process multiple roles overnight without manual intervention. HOTL will:
- Execute pending tasks from the queue
- Run tests and validate changes
- Send Discord notifications on progress
- Handle errors with automatic retries
- Checkpoint progress for recovery

## Prerequisites

1. **Install harness**
   ```bash
   harness install run
   ```

2. **Configure Discord notifications** (recommended)
   ```bash
   export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
   ```

3. **Populate task queue**
   ```bash
   # Sync roles
   harness sync --roles --gitlab

   # Check pending work
   harness list-roles --wave 1
   ```

## Steps

### Step 1: Start HOTL mode

Start autonomous operation with 2-hour runtime:

```bash
# Via CLI
harness hotl start --max-iterations 50 --notify-interval 600 --discord

# Via MCP client skill
/hotl start --max-iterations 50 --notify-interval 600
```

**Parameters explained:**
- `--max-iterations 50` - Process up to 50 tasks
- `--notify-interval 600` - Send status every 10 minutes
- `--discord` - Enable Discord notifications

### Step 2: Monitor via Discord

You'll receive notifications like:

```
[HOTL Status Update]
Phase: EXECUTING
Iteration: 12/50
Completed: 8 tasks
Failed: 1 task
Current: Running molecule tests for iis-config
Next check: 10 minutes
```

### Step 3: Check status (optional)

In another terminal:

```bash
harness hotl status
```

Or check metrics:

```bash
harness metrics status
```

### Step 4: Stop gracefully (if needed)

To stop before completion:

```bash
harness hotl stop
```

This completes the current task before stopping.

### Step 5: Resume after interruption

If HOTL was interrupted, resume from checkpoint:

```bash
harness hotl resume <thread-id>
```

The thread ID is shown in the initial startup output.

## Workflow Phases

HOTL cycles through these phases:

| Phase | Description |
|-------|-------------|
| IDLE | Waiting for work |
| RESEARCHING | Gathering codebase context |
| PLANNING | Updating task plans |
| GAP_ANALYZING | Finding incomplete work |
| EXECUTING | Running tasks |
| TESTING | Validating changes |
| NOTIFYING | Sending status updates |

## Example: Overnight Processing

```bash
# Start at 6 PM, process through night
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
harness hotl start \
  --max-iterations 200 \
  --notify-interval 1800 \
  --discord

# Check results next morning
harness metrics status
harness status
```

## Example: Quick Batch

```bash
# Process 10 tasks quickly
harness hotl start \
  --max-iterations 10 \
  --notify-interval 60
```

## Error Handling

HOTL automatically handles:

- **Transient failures** - Retries up to 3 times
- **Test failures** - Records failure, moves to next task
- **Network issues** - Waits and retries
- **Too many errors** - Stops after 5 consecutive failures

## Notifications

### Discord Format

```json
{
  "embeds": [{
    "title": "HOTL Status Update",
    "color": 3066993,
    "fields": [
      {"name": "Phase", "value": "EXECUTING", "inline": true},
      {"name": "Progress", "value": "12/50", "inline": true},
      {"name": "Success Rate", "value": "88%", "inline": true}
    ]
  }]
}
```

### Email Notifications

```bash
export HOTL_EMAIL_TO="you@example.com"
harness hotl start --email
```

## Troubleshooting

### "No pending tasks"
Ensure tasks are queued:
```bash
harness sync --roles --gitlab
harness list-roles
```

### Discord notifications not working
Test webhook directly:
```bash
curl -X POST "$DISCORD_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test message"}'
```

### HOTL stops unexpectedly
Check for error accumulation:
```bash
harness metrics history hotl_errors --hours 1
```
