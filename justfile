# dag-harness monorepo justfile
# Orchestrates Python package, documentation, and release workflows
#
# Run `just` to see available recipes

set dotenv-load := true

# Default - show help
default:
    @just --list --unsorted

# =============================================================================
# DEVELOPMENT
# =============================================================================

# Install all dependencies and set up development environment
setup:
    cd harness && uv sync --all-extras
    git config core.hooksPath .githooks
    @echo "Development environment ready"

# Run all checks (lint, format, test)
check-all: lint test

# Delegate to harness justfile for Python-specific tasks
[no-cd]
harness *ARGS:
    just -f harness/justfile {{ARGS}}

# =============================================================================
# TESTING (delegated)
# =============================================================================

# Run tests
test *ARGS:
    just harness test {{ARGS}}

# Run tests with coverage
test-cov:
    just harness test-cov

# Run property-based tests
test-pbt:
    just harness test-pbt

# =============================================================================
# LINTING (delegated)
# =============================================================================

# Run linter
lint:
    just harness lint

# Auto-fix lint issues
lint-fix:
    just harness lint-fix

# Format code
format:
    just harness format

# =============================================================================
# DOCUMENTATION
# =============================================================================

# Build documentation
docs-build:
    mkdocs build --strict

# Serve documentation locally
docs-serve:
    mkdocs serve

# Deploy docs to GitHub Pages
docs-deploy:
    mkdocs gh-deploy --force

# =============================================================================
# BUILD & RELEASE
# =============================================================================

# Check version consistency
version-check:
    @grep '^__version__' harness/harness/__init__.py
    @grep '^version' harness/pyproject.toml || true
    @echo "Ensure versions match before release"

# Build wheel and sdist
build:
    cd harness && uv build

# Show current version
version:
    @grep '^__version__' harness/harness/__init__.py | cut -d'"' -f2

# Clean build artifacts
clean:
    rm -rf harness/dist/ harness/build/ harness/*.egg-info/
    rm -rf .coverage htmlcov/ .pytest_cache/ .ruff_cache/ .mypy_cache/
    rm -rf site/ public/

# =============================================================================
# GIT HOOKS
# =============================================================================

# Install git hooks
hooks-install:
    git config core.hooksPath .githooks
    chmod +x .githooks/*
    @echo "Git hooks installed"

# Check hooks status
hooks-status:
    @echo "Hooks path: $(git config core.hooksPath)"
    @ls -la .githooks/

# =============================================================================
# WORKTREES
# =============================================================================

# Create a role worktree
worktree-role ROLE:
    ./scripts/create-role-worktree.sh {{ROLE}}

# List all worktrees
worktree-list:
    git worktree list

# =============================================================================
# DATABASE
# =============================================================================

# Initialize database
db-init:
    just harness db-init

# Show database stats
db-stats:
    just harness db-stats

# =============================================================================
# MCP SERVER
# =============================================================================

# Start MCP server
mcp-serve:
    cd harness && uv run harness mcp-server

# =============================================================================
# HARNESS CLI
# =============================================================================

# Run harness CLI command
run *ARGS:
    cd harness && uv run harness {{ARGS}}

# Bootstrap check
bootstrap-check:
    cd harness && uv run harness bootstrap --check-only

# Sync roles from filesystem
sync:
    cd harness && uv run harness sync --roles
