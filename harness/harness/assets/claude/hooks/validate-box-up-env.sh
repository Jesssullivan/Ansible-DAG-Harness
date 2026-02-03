#!/bin/bash
# validate-box-up-env.sh - PreToolUse hook for box-up-role operations
#
# This Claude Code hook validates the environment before tool execution.
# It checks for required credentials, glab authentication, and tunnel status.
#
# Deployed by: harness init
#
# Exit codes:
#   0 - Allow tool execution
#   2 - Block tool execution (missing requirements)

set -euo pipefail

# Only validate for box-up related commands
TOOL_INPUT="${CLAUDE_TOOL_INPUT:-}"
if [[ ! "$TOOL_INPUT" =~ (box-up|molecule|deploy|gitlab|glab|create-.*issue|create-.*mr) ]]; then
    exit 0
fi

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=()

# Check GitLab token
if [[ -z "${GITLAB_TOKEN:-}" ]] && [[ -z "${GITLAB_PRIVATE_TOKEN:-}" ]]; then
    # Check glab auth status
    if ! glab auth status &>/dev/null 2>&1; then
        ERRORS+=("GitLab authentication missing. Run: glab auth login")
    fi
fi

# Check KeePassXC database path (for credential lookups)
if [[ -z "${KEEPASS_DATABASE_PATH:-}" ]]; then
    # Search common locations for .kdbx files
    if ! find . -maxdepth 2 -name "*.kdbx" -print -quit 2>/dev/null | grep -q .; then
        ERRORS+=("KEEPASS_DATABASE_PATH not set and no .kdbx file found")
    fi
fi

# Check KeePassXC password (required for credential lookups)
if [[ -z "${KEEPASSXC_DB_PASSWORD:-}" ]]; then
    ERRORS+=("KEEPASSXC_DB_PASSWORD not set (required for credential lookups)")
fi

# For operations targeting specific hosts, check tunnel status
if [[ "$TOOL_INPUT" =~ (vmnode8[0-9][0-9]|8[0-9][0-9]) ]]; then
    # Check if any tunnels are running on expected ports
    if ! nc -zv localhost 15852 &>/dev/null 2>&1; then
        echo -e "${YELLOW}WARNING: SSH tunnels may not be running${NC}" >&2
        echo "For tunneled VMs, start tunnels with: npm run tunnels:start" >&2
    fi
fi

# Report errors if any
if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo -e "${RED}========================================${NC}" >&2
    echo -e "${RED}ENVIRONMENT VALIDATION FAILED${NC}" >&2
    echo -e "${RED}========================================${NC}" >&2
    for error in "${ERRORS[@]}"; do
        echo -e "${RED}  - $error${NC}" >&2
    done
    echo "" >&2
    echo "Fix the above issues before proceeding." >&2
    exit 2
fi

exit 0
