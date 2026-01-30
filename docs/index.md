# DAG Harness Documentation

A self-installing DAG orchestration package built on LangGraph with complete MCP client integration.

## Overview

The DAG Harness provides automated workflow orchestration for Ansible role deployments, with features including:

- **LangGraph-based DAG execution** - Reliable, checkpointed workflow execution
- **MCP client integration** - MCP server, hooks, and skills for AI-assisted development
- **Self-bootstrapping** - Single-command setup from within MCP client
- **Human Out of The Loop (HOTL)** - Autonomous operation mode with notifications

## Quick Start

```bash
# Install the harness
cd harness && uv sync && cd ..

# Bootstrap (interactive setup)
harness bootstrap

# Verify installation
harness bootstrap --check-only
```

## Documentation

- [Getting Started](getting-started.md) - Installation and first steps
- [Architecture](architecture.md) - System design and components
- [CLI Reference](api/cli.md) - Command-line interface documentation
- [MCP Tools](api/mcp-tools.md) - MCP server tool reference

## Key Features

### Self-Bootstrapping

The harness can install and configure itself from within MCP client:

```bash
harness bootstrap              # Full interactive setup
harness bootstrap --check-only # Verify current state
```

The bootstrap wizard:

1. Detects your environment (Python, git, paths)
2. Discovers and validates credentials
3. Initializes the database
4. Installs MCP client integration
5. Runs self-tests to verify setup

### Workflow Execution

Execute box-up-role workflows for Ansible roles:

```bash
harness box-up-role <role-name>
harness status [role-name]
harness resume <execution-id>
```

### HOTL Mode

Run autonomous operations with human oversight:

```bash
harness hotl start --max-iterations 100
harness hotl status
harness hotl stop
```

## Project Structure

```
disposable-dag-langchain/
├── harness/                    # Core Python package
│   ├── harness/
│   │   ├── cli.py              # CLI entry point
│   │   ├── bootstrap/          # Self-bootstrapping system
│   │   ├── dag/                # LangGraph engine
│   │   ├── db/                 # SQLite state management
│   │   ├── hotl/               # Autonomous operation
│   │   └── mcp/                # MCP server
│   └── tests/
├── .claude/                    # MCP client integration
│   ├── settings.json           # MCP server + hooks config
│   ├── hooks/                  # PreToolUse/PostToolUse hooks
│   └── skills/                 # Skill definitions
└── docs/                       # Documentation
```

## License

MIT License - see [LICENSE](../LICENSE) for details.
