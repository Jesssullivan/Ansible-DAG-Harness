# DAG Harness - Claude Code Project Instructions

## Project Overview

DAG Harness is a **LangGraph-based orchestration system** for Ansible role deployments.
Built on LangGraph 1.0.x, FastMCP, and SQLite with 40+ MCP tools for full Claude Code integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code / MCP Client                 │
├─────────────────────────────────────────────────────────────┤
│                        MCP Server                           │
│              (40+ tools across 8 categories)                │
├─────────────────────────────────────────────────────────────┤
│    CLI (Typer)    │   LangGraph DAG   │   HOTL Supervisor   │
├───────────────────┼───────────────────┼─────────────────────┤
│                      StateDB (SQLite)                       │
├─────────────────────────────────────────────────────────────┤
│   GitLab API   │   Git Worktrees   │   Molecule/Pytest     │
└─────────────────────────────────────────────────────────────┘
```

## Key Directories

```
dag-harness/
├── harness/                 # Python package root
│   ├── harness/
│   │   ├── cli.py          # Typer CLI entry point
│   │   ├── config.py       # Configuration (dataclass + Pydantic)
│   │   ├── dag/            # LangGraph workflow
│   │   │   ├── langgraph_engine.py   # Main DAG (14 nodes)
│   │   │   ├── langgraph_nodes.py    # Node implementations
│   │   │   ├── langgraph_state.py    # BoxUpRoleState TypedDict
│   │   │   └── checkpointer.py       # CheckpointerWithStateDB
│   │   ├── db/             # StateDB (SQLite)
│   │   │   ├── state.py    # StateDB class
│   │   │   └── schema.sql  # Database schema
│   │   ├── mcp/            # MCP server
│   │   │   └── server.py   # FastMCP implementation
│   │   └── hotl/           # Autonomous mode
│   ├── tests/              # 900+ tests (pytest + hypothesis)
│   ├── pyproject.toml      # Package config
│   └── justfile            # Development tasks
├── docs/                   # MkDocs documentation
├── scripts/                # Bootstrap and automation
└── .github/workflows/      # CI/CD
```

## Commands

### Development
```bash
cd harness
uv sync --all-extras           # Install dependencies
uv run pytest tests/ -v        # Run tests
uv run ruff check harness/     # Lint
uv run ruff format harness/    # Format
just test                      # Via justfile
just lint
```

### Workflow Operations
```bash
harness bootstrap              # Initial setup
harness sync --roles           # Sync from filesystem
harness status <role>          # Check role status
harness box-up-role <role>     # Execute workflow
harness resume <id> --approve  # Continue after HITL gate
harness hotl start             # Autonomous mode
```

## Git Workflow

- **Branch model**: Three-tier (main <- dev <- feature)
- **Commit style**: Conventional commits with scoped types
- **No AI attribution**: Do NOT include Co-Authored-By lines

### Commit Message Format
```
<type>(<scope>): <description>

[optional body]
```

Types: feat, fix, docs, style, refactor, test, chore

## Testing

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=harness --cov-report=term-missing

# Property-based tests only
uv run pytest -m pbt

# Unit tests only
uv run pytest -m unit

# Skip slow tests
uv run pytest -m "not slow"
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| langgraph >= 1.0.7 | Workflow engine with checkpointing |
| fastmcp >= 0.1.0 | MCP server implementation |
| typer >= 0.12 | CLI framework |
| pydantic >= 2.0 | Configuration validation |
| hypothesis | Property-based testing |

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| GITLAB_TOKEN | GitLab API authentication | Yes |
| HARNESS_DB_PATH | Database path override | No |
| HARNESS_CONFIG | Config file override | No |
| LANGCHAIN_TRACING_V2 | Enable LangSmith tracing | No |
| DISCORD_WEBHOOK_URL | Discord notifications | No |

## Skills Available

- `/box-up-role <role>` - Package Ansible role workflow
- `/hotl [start|stop|status]` - Autonomous batch processing
- `/observability <command>` - Platform debugging

## Important Notes

1. **Check status first**: Always `harness status <role>` before operations
2. **Sync before workflows**: `harness sync` ensures fresh state
3. **HITL gates**: Workflows pause at human_approval node
4. **Worktrees**: Each role gets isolated git worktree
5. **Cost tracking**: MCP tools track token usage automatically
