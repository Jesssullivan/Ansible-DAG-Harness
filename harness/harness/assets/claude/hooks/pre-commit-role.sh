#!/bin/bash
# pre-commit-role.sh - Pre-commit hook for role branches
#
# This hook validates that molecule tests pass before allowing commits
# on role branches (sid/* pattern). Install as a git pre-commit hook.
#
# Deployed by: harness init
#
# Installation:
#   ln -sf ../../.claude/hooks/pre-commit-role.sh .git/hooks/pre-commit
#
# Skip validation:
#   git commit --no-verify

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

# Only run on sid/* branches (configurable via HARNESS_BRANCH_PREFIX)
BRANCH_PREFIX="${HARNESS_BRANCH_PREFIX:-sid/}"
if [[ ! "$CURRENT_BRANCH" =~ ^${BRANCH_PREFIX} ]]; then
    exit 0
fi

# Extract role name from branch
ROLE_NAME="${CURRENT_BRANCH#${BRANCH_PREFIX}}"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}PRE-COMMIT: Validating role $ROLE_NAME${NC}"
echo -e "${YELLOW}========================================${NC}"

# Find repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Check if role exists
ROLE_PATH="$REPO_ROOT/ansible/roles/$ROLE_NAME"
if [[ ! -d "$ROLE_PATH" ]]; then
    echo -e "${RED}Role not found: $ROLE_NAME${NC}"
    echo "Branch name suggests role '$ROLE_NAME' but it doesn't exist."
    echo "Either rename the branch or create the role."
    exit 1
fi

# Check if molecule directory exists
if [[ ! -d "$ROLE_PATH/molecule" ]]; then
    echo -e "${YELLOW}No molecule tests found for $ROLE_NAME${NC}"
    echo "Skipping molecule validation."
    echo ""
    exit 0
fi

# Run molecule test
echo ""
echo "Running molecule tests..."
if [[ -f "scripts/validate-role-tests.sh" ]]; then
    if scripts/validate-role-tests.sh "$ROLE_NAME" --molecule-only; then
        echo -e "${GREEN}Molecule tests passed${NC}"
    else
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}COMMIT BLOCKED: Molecule tests failed${NC}"
        echo -e "${RED}========================================${NC}"
        echo ""
        echo "Fix the failing tests before committing."
        echo "To skip validation: git commit --no-verify"
        exit 1
    fi
else
    # Fallback to direct molecule run
    cd "$ROLE_PATH"
    if molecule test; then
        echo -e "${GREEN}Molecule tests passed${NC}"
    else
        echo ""
        echo -e "${RED}COMMIT BLOCKED: Molecule tests failed${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PRE-COMMIT: Validation passed${NC}"
echo -e "${GREEN}========================================${NC}"
exit 0
