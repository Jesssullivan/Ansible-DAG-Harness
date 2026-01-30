#!/bin/bash
# install.sh - One-command installer for DAG Harness
#
# Usage:
#   curl -sSL https://example.com/install.sh | bash
#   # or
#   ./scripts/install.sh
#
# This script:
# 1. Checks prerequisites (Python 3.11+, uv)
# 2. Installs the harness package
# 3. Runs harness install to configure MCP client

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DAG Harness Installer${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check Python version
echo -n "Checking Python... "
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [[ "$MAJOR" -ge 3 ]] && [[ "$MINOR" -ge 11 ]]; then
        echo -e "${GREEN}OK${NC} (Python $PYTHON_VERSION)"
    else
        echo -e "${RED}FAIL${NC}"
        echo -e "${RED}Python 3.11+ required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}FAIL${NC}"
    echo -e "${RED}Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi

# Check uv
echo -n "Checking uv... "
if command -v uv &>/dev/null; then
    UV_VERSION=$(uv --version | head -1)
    echo -e "${GREEN}OK${NC} ($UV_VERSION)"
else
    echo -e "${YELLOW}NOT FOUND${NC}"
    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

# Find project root (look for harness/pyproject.toml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../harness/pyproject.toml" ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "./harness/pyproject.toml" ]]; then
    PROJECT_ROOT="$(pwd)"
else
    echo -e "${RED}Error: Cannot find harness/pyproject.toml${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo

# Install harness package
echo -e "${BLUE}Installing harness package...${NC}"
cd "$PROJECT_ROOT/harness"
uv sync
cd "$PROJECT_ROOT"

# Run harness install
echo
echo -e "${BLUE}Installing MCP client integration...${NC}"
uv run --directory ./harness harness install run

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo "Next steps:"
echo "  1. Restart MCP client to load MCP server"
echo "  2. Try: /box-up-role --help"
echo "  3. Check status: harness install check"
echo
