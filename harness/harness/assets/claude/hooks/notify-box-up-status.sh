#!/bin/bash
# notify-box-up-status.sh - PostToolUse hook for box-up-role operations
#
# This Claude Code hook sends notifications after tool execution completes.
# It integrates with Discord and email for failure notifications.
#
# Deployed by: harness init
#
# Environment variables:
#   DISCORD_WEBHOOK_URL - Discord webhook for notifications
#   EMAIL_RECIPIENT - Email address for notifications
#   CLAUDE_TOOL_EXIT_CODE - Exit code from tool execution (provided by Claude Code)
#   CLAUDE_TOOL_NAME - Name of the tool that was executed
#   CLAUDE_TOOL_INPUT - Input that was passed to the tool

set -uo pipefail

# Get tool execution status from Claude Code
EXIT_CODE="${CLAUDE_TOOL_EXIT_CODE:-0}"
TOOL_NAME="${CLAUDE_TOOL_NAME:-unknown}"
TOOL_INPUT="${CLAUDE_TOOL_INPUT:-}"

# Only notify for box-up related commands
if [[ ! "$TOOL_INPUT" =~ (box-up|molecule|deploy|create-.*issue|create-.*mr) ]]; then
    exit 0
fi

# Only notify on failure
if [[ "$EXIT_CODE" -eq 0 ]]; then
    exit 0
fi

# Extract context from tool input
ROLE_NAME=""
if [[ "$TOOL_INPUT" =~ --role[=\ ]([a-z_]+) ]]; then
    ROLE_NAME="${BASH_REMATCH[1]}"
elif [[ "$TOOL_INPUT" =~ ansible/roles/([a-z_]+) ]]; then
    ROLE_NAME="${BASH_REMATCH[1]}"
fi

# Build notification message
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
MESSAGE="Box-up operation failed"
if [[ -n "$ROLE_NAME" ]]; then
    MESSAGE="Box-up failed for role: $ROLE_NAME"
fi

# Send Discord notification
if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
    DISCORD_PAYLOAD=$(cat <<EOF
{
  "embeds": [
    {
      "title": "Box Up Role Failed",
      "description": "$MESSAGE",
      "color": 16711680,
      "fields": [
        {"name": "Tool", "value": "$TOOL_NAME", "inline": true},
        {"name": "Exit Code", "value": "$EXIT_CODE", "inline": true},
        {"name": "Role", "value": "${ROLE_NAME:-N/A}", "inline": true}
      ],
      "timestamp": "$TIMESTAMP"
    }
  ]
}
EOF
)
    curl -s -X POST "$DISCORD_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "$DISCORD_PAYLOAD" &>/dev/null || true
fi

# Send email notification
if [[ -n "${EMAIL_RECIPIENT:-}" ]] && command -v mail &>/dev/null; then
    EMAIL_BODY="Box-up operation failed at $TIMESTAMP

Tool: $TOOL_NAME
Exit Code: $EXIT_CODE
Role: ${ROLE_NAME:-N/A}
Input: $TOOL_INPUT

Check the Claude Code session for details."

    echo "$EMAIL_BODY" | mail -s "Box Up Failed: ${ROLE_NAME:-unknown}" "$EMAIL_RECIPIENT" 2>/dev/null || true
fi

exit 0
