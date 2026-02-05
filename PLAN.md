# DAG Harness Developer Experience & CI Enhancement Plan

## Overview

This plan consolidates findings from 5 parallel Opus agent workstreams analyzing:
1. Git integration & repository state
2. Build systems (Justfile, Nix, Bazel)
3. CI/Linting parity (GitHub Actions ↔ GitLab CI)
4. Developer experience (.envrc, starship, MkDocs, Pages)
5. Claude Code configuration (CLAUDE.md, ignores, hooks)

## Current Repository State

| Item | Status |
|------|--------|
| Worktrees | Single (main) |
| Remotes | origin (GitHub) 1 behind, gitlab 10 behind, old-repo (stale) |
| Uncommitted | v0.6.1 tunnel preflight feature |
| Stashes | None |

## Implementation Phases

### Phase 1: Git Cleanup & Hooks (Tasks 1-3)
**Priority: High** - Foundation for all other work

1. **Sync remotes** - Remove stale `old-repo`, push to origin/gitlab
2. **Update .gitignore** - Add AI artifact patterns from jesssullivan
3. **Install git hooks** - pre-commit, commit-msg, prepare-commit-msg

### Phase 2: Build System Enhancement (Tasks 4-6)
**Priority: Medium** - Improves developer workflow

4. **Root justfile** - Monorepo orchestration
5. **Enhance harness/justfile** - Add molecule, MCP, docs recipes
6. **Nix flake** - Reproducible dev environment (optional)

### Phase 3: Linting & CI Parity (Tasks 7-11)
**Priority: High** - Quality gates

7. **Pre-commit config** - ruff, mypy, shellcheck, yamllint
8. **Mypy config** - Type checking setup
9. **GitHub Actions** - Matrix, caching, coverage, security
10. **GitLab CI** - Parity with GitHub
11. **GitHub Pages** - Documentation deployment

### Phase 4: Developer Experience (Tasks 12-13)
**Priority: Medium** - Onboarding improvement

12. **Create .envrc** - direnv integration
13. **Expand MkDocs** - Navigation, plugins

### Phase 5: Claude Code Setup (Tasks 14-16)
**Priority: High** - AI assistant configuration

14. **Create CLAUDE.md** - Project context
15. **Create .claudeignore** - Context optimization
16. **Update settings.json** - Disable Co-Authored-By

## Key Decisions

### Bazel: NOT RECOMMENDED
- Project complexity doesn't warrant Bazel overhead
- uv + hatchling handles Python packaging well
- Nix provides reproducibility if needed

### Nix Flake: RECOMMENDED (Optional)
- User has Nix experience
- Provides reproducible dev environment
- Pattern exists in other user projects

### CI Strategy
- GitHub Actions as primary
- GitLab CI mirrors for parity
- Pre-commit for local validation

## File Changes Summary

### New Files
- `/justfile` - Root monorepo orchestration
- `/flake.nix` - Nix dev environment
- `/.envrc` + `/.envrc.example` - direnv
- `/starship.toml` - Prompt config (optional)
- `/.pre-commit-config.yaml` - Local linting
- `/.yamllint.yml` - YAML linting rules
- `/.markdownlint.yml` - Markdown rules
- `/.github/workflows/docs.yml` - Pages deployment
- `/CLAUDE.md` - Claude Code context
- `/.claudeignore` - Context exclusions
- `/.claude/hooks/validate-commit-msg.sh` - Commit hook

### Modified Files
- `/.gitignore` - Add AI artifact patterns
- `/harness/justfile` - Add recipes
- `/harness/pyproject.toml` - Add mypy config
- `/.github/workflows/ci.yml` - Enhance CI
- `/.gitlab-ci.yml` - Add parity features
- `/mkdocs.yml` - Expand navigation
- `/.claude/settings.json` - Disable Co-Authored-By

## Verification Checklist

After implementation:
- [ ] `git remote -v` shows only origin and gitlab
- [ ] `git push origin main && git push gitlab main` succeeds
- [ ] `just` at root shows help
- [ ] `just test` runs tests
- [ ] `direnv allow` activates environment
- [ ] `pre-commit run --all-files` passes
- [ ] GitHub Actions CI passes
- [ ] GitLab CI passes
- [ ] GitHub Pages deploys
- [ ] Claude Code loads CLAUDE.md context
