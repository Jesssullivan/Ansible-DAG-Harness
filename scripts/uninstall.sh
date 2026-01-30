#!/bin/bash
# uninstall.sh - Remove DAG Harness from MCP client
#
# Usage:
#   ./scripts/uninstall.sh [--remove-data]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REMOVE_DATA=false
if [[ "${1:-}" == "--remove-data" ]]; then
    REMOVE_DATA=true
fi

echo -e "${YELLOW}DAG Harness Uninstaller${NC}"
echo

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../harness/pyproject.toml" ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "./harness/pyproject.toml" ]]; then
    PROJECT_ROOT="$(pwd)"
else
    echo -e "${RED}Error: Cannot find project root${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

# Confirm
read -p "Remove harness from MCP client? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run harness uninstall
if $REMOVE_DATA; then
    uv run --directory ./harness harness install uninstall --yes --remove-data
else
    uv run --directory ./harness harness install uninstall --yes
fi

echo
echo -e "${GREEN}Uninstall complete.${NC}"
