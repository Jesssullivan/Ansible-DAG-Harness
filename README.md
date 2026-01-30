# Disposable DAG LangChain

A standalone, self-installing DAG orchestration package built on LangGraph with MCP integration.

## Features

- **LangGraph-based DAG Engine** - Declarative workflow graphs with checkpointing
- **Self-Installing** - One command bootstrap setup
- **MCP Server** - 20+ tools for workflow management and status queries
- **HOTL Mode** - Human Out of The Loop autonomous operation
- **Hook System** - PreToolUse/PostToolUse hooks for validation and notifications

## Quick Start

```bash
# Install dependencies
cd harness && uv sync && cd ..

# Bootstrap the harness
harness bootstrap

# Verify installation
harness bootstrap --check-only
```

## Documentation

- **[docs/](./docs/)** - Full documentation
- **[vignettes/](./vignettes/)** - Step-by-step usage examples

## Directory Structure

```
disposable-dag-langchain/
├── harness/                    # Core Python package
│   ├── harness/
│   │   ├── cli.py              # CLI entry point
│   │   ├── bootstrap/          # Self-installation system
│   │   ├── dag/                # LangGraph workflow engine
│   │   ├── db/                 # SQLite state management
│   │   ├── hotl/               # Autonomous operation
│   │   └── mcp/                # MCP server
│   └── tests/
├── docs/                       # Documentation
├── vignettes/                  # Usage examples
└── scripts/                    # Utility scripts
```

## CLI Commands

```bash
# Bootstrap
harness bootstrap               # Interactive setup
harness bootstrap --check-only  # Verify setup

# Core operations
harness box-up-role <role>      # Execute workflow
harness status [role]           # Show status
harness sync                    # Sync from filesystem

# HOTL mode
harness hotl start              # Start autonomous mode
harness hotl status             # Check status
harness hotl stop               # Stop gracefully

# Database
harness db stats                # Show statistics
harness check                   # Run self-checks
```

## Requirements

- Python 3.11+
- [uv](https://astral.sh/uv/) package manager

## License

MIT
