# V2 Implementation Plan: 8-Week Parallel Agent Roadmap

**Created**: 2026-02-01
**Status**: Ready for Execution
**Structure**: 4 parallel agents per phase + gap analysis checkpoints

---

## Executive Summary

This plan addresses all gaps identified by the 4 deep analysis agents:
- **Claude 2026 Integration**: Migrate from subprocess to `claude-agent-sdk`
- **GitLab API Completeness**: Idempotency, labels, iterations, lifecycle
- **LangGraph 2026 Patterns**: Upgrade to 1.0.x, parallel execution, subgraphs
- **Manus/Agent IO**: Context engineering, A2A protocol, cost tracking

---

## Phase 1: Critical Fixes (Week 1-2)

### Agent 1.1: Claude Agent SDK Migration
**Goal**: Replace subprocess spawning with official SDK

**Tasks**:
1. Add `claude-agent-sdk` to pyproject.toml dependencies
2. Create `harness/hotl/claude_sdk_integration.py` with:
   - `SDKClaudeIntegration` class using `ClaudeSDKClient`
   - In-process MCP tools via `@tool` decorator
   - Hook support for `PreToolUse`/`PostToolUse`
   - Session resumption support
3. Migrate `AgentSession` to use SDK session IDs
4. Update `HOTLSupervisor` to use new SDK integration
5. Add unit tests for SDK integration
6. Deprecate old `claude_integration.py` (keep for fallback)

**Success Criteria**:
- [ ] SDK client can spawn agents
- [ ] Progress reporting works via in-process tools
- [ ] Session resume works
- [ ] All existing HOTL tests pass

---

### Agent 1.2: GitLab Idempotency
**Goal**: Prevent duplicate issues/MRs on re-run

**Tasks**:
1. Add `find_existing_issue(role_name)` to `gitlab/api.py`
2. Add `find_existing_mr(source_branch)` to `gitlab/api.py`
3. Add `get_or_create_issue()` wrapper (idempotent)
4. Add `get_or_create_mr()` wrapper (idempotent)
5. Add `remote_branch_exists(branch_name)` check
6. Update `create_issue_node` in langgraph_engine.py
7. Update `create_mr_node` in langgraph_engine.py
8. Add `issue_created` and `mr_created` flags to state
9. Add integration tests for idempotency

**Success Criteria**:
- [ ] Running `box-up-role common` twice doesn't create duplicates
- [ ] Existing issues/MRs are detected and reused
- [ ] Branch collision handled gracefully

---

### Agent 1.3: LangGraph 1.0.x Upgrade
**Goal**: Upgrade to stable LangGraph 1.0.x

**Tasks**:
1. Update `pyproject.toml`: `langgraph>=1.0.7`
2. Update checkpoint imports if changed
3. Add `RetryPolicy` to external API nodes:
   - `create_issue_node`
   - `create_mr_node`
   - `add_to_merge_train_node`
4. Add `RetryPolicy` to subprocess nodes:
   - `run_molecule_node`
   - `run_pytest_node`
5. Update any deprecated API calls
6. Run full test suite, fix breaking changes
7. Document migration notes

**Success Criteria**:
- [ ] All 272+ tests pass on LangGraph 1.0.x
- [ ] RetryPolicy configured for flaky operations
- [ ] No deprecation warnings

---

### Agent 1.4: Label Management
**Goal**: Auto-create labels with proper colors

**Tasks**:
1. Add `WAVE_LABEL_COLORS` constant to `gitlab/api.py`
2. Add `SCOPED_LABELS` constant for `priority::`, `status::`, `role::`
3. Add `ensure_label_exists(name, color, description)` method
4. Add `ensure_wave_labels()` method
5. Add `ensure_scoped_labels(scope, values)` method
6. Add `prepare_labels_for_role(role_name, wave)` method
7. Update `create_issue_node` to use `prepare_labels_for_role`
8. Add tests for label creation

**Success Criteria**:
- [ ] `wave-0` through `wave-4` labels created automatically
- [ ] Scoped labels work (`role::common`, `priority::high`)
- [ ] Labels have correct colors
- [ ] Idempotent (doesn't fail if label exists)

---

## Gap Analysis Checkpoint 1 (End of Week 2)

### Reality Checker Agent 1
**Scope**: Verify Phase 1 completion

**Checks**:
1. **SDK Integration**: Spawn test agent, verify output
2. **Idempotency**: Run `box-up-role test-role` twice, count issues/MRs
3. **LangGraph Version**: Verify `langgraph.__version__` >= 1.0.7
4. **Labels**: Check GitLab project for wave labels
5. **Test Suite**: All tests pass
6. **No Regressions**: Bootstrap still works

**Output**: Gap report with blockers for Phase 2

---

## Phase 2: High Priority Improvements (Week 3-4)

### Agent 2.1: Iteration & Issue Lifecycle
**Goal**: Fix iteration management, add lifecycle methods

**Tasks**:
1. Add `get_current_iteration_by_date()` using date range check
2. Add `ensure_iteration_exists()` with warning if none
3. Add `close_issue(issue_iid)` method
4. Add `reopen_issue(issue_iid)` method
5. Add `update_issue_on_failure(issue_iid, error)` method
6. Add `set_issue_due_date(issue_iid, due_date)` method
7. Add `add_time_estimate(issue_iid, duration)` method
8. Wire `update_issue_on_failure` into workflow error paths
9. Add tests for lifecycle methods

**Success Criteria**:
- [ ] Current iteration selected by date, not just "first"
- [ ] Issues updated with failure info when workflow fails
- [ ] Due dates can be set programmatically

---

### Agent 2.2: MR Lifecycle & Reviewers
**Goal**: Complete MR lifecycle management

**Tasks**:
1. Add `set_mr_reviewers(mr_iid, usernames)` method
2. Add `get_mr_pipeline_status(mr_iid)` method
3. Add `wait_for_mr_pipeline(mr_iid, timeout)` method
4. Add `add_mr_comment(mr_iid, body)` method
5. Add `mark_mr_ready(mr_iid)` to remove Draft prefix
6. Add `merge_immediately(mr_iid, skip_train)` method
7. Update `create_mr_node` to set reviewers from config
8. Add config option for default reviewers
9. Add tests for MR lifecycle

**Success Criteria**:
- [ ] MRs have reviewers assigned automatically
- [ ] Can check pipeline status before merge train
- [ ] Can add comments to MRs programmatically

---

### Agent 2.3: HITL Interrupt Pattern
**Goal**: Implement modern `interrupt()` pattern

**Tasks**:
1. Import `interrupt`, `Command` from `langgraph.types`
2. Create `human_approval_node` using `interrupt()`
3. Add `interrupt_before=["add_to_merge_train"]` to graph compile
4. Add `interrupt_after=["run_molecule"]` for test review
5. Create `resume_with_approval(thread_id, approved, reason)` helper
6. Update CLI to support `harness resume --approve`
7. Update CLI to support `harness resume --reject --reason "..."`
8. Add documentation for interrupt workflow
9. Add tests for interrupt/resume cycle

**Success Criteria**:
- [ ] Workflow pauses before merge train
- [ ] Can resume with approval via CLI
- [ ] Can reject with reason
- [ ] State preserved across interrupt

---

### Agent 2.4: Observability Integration
**Goal**: Add LangSmith and structured logging

**Tasks**:
1. Add `langsmith>=0.1.0` to pyproject.toml
2. Add environment variable support for `LANGCHAIN_TRACING_V2`
3. Create `AnonymizingCallback` to mask sensitive data
4. Add LangSmith tracer to graph execution
5. Add `astream_events` support for real-time updates
6. Create streaming callback for Discord notifications
7. Add MCP Tool Search pattern to reduce context
8. Add `search_tools(query, category)` MCP tool
9. Add tests for observability

**Success Criteria**:
- [ ] Traces visible in LangSmith when enabled
- [ ] Sensitive data masked (tokens, passwords)
- [ ] Real-time streaming to Discord works
- [ ] Tool search reduces context by 50%+

---

## Gap Analysis Checkpoint 2 (End of Week 4)

### Reality Checker Agent 2
**Scope**: Verify Phase 2 completion

**Checks**:
1. **Iterations**: Run with iteration, verify date-based selection
2. **Issue Lifecycle**: Trigger failure, verify issue updated
3. **MR Reviewers**: Create MR, verify reviewers assigned
4. **Interrupts**: Run workflow, verify pause at merge train
5. **LangSmith**: Check for traces (if API key configured)
6. **Tool Search**: Verify context reduction

**Output**: Gap report with blockers for Phase 3

---

## Phase 3: Parallel Execution & Subgraphs (Week 5-6)

### Agent 3.1: Parallel Test Execution
**Goal**: Run molecule and pytest in parallel

**Tasks**:
1. Import `Send` from `langgraph.constants`
2. Create `route_to_parallel_tests(state)` returning `list[Send]`
3. Add parallel edge: `create_worktree` → `[run_molecule, run_pytest]`
4. Add join edge: `[run_molecule, run_pytest]` → `validate_deploy`
5. Update state to handle parallel results
6. Add `test_results_reducer` for merging parallel outputs
7. Handle partial failures (one test fails, other succeeds)
8. Add performance benchmarks
9. Add tests for parallel execution

**Success Criteria**:
- [ ] Molecule and pytest run concurrently
- [ ] Both results merged correctly
- [ ] Partial failures handled gracefully
- [ ] 30%+ time reduction for test phase

---

### Agent 3.2: Subgraph Composition
**Goal**: Modular workflow via subgraphs

**Tasks**:
1. Create `create_validation_subgraph()` for validate/analyze nodes
2. Create `create_testing_subgraph()` for molecule/pytest/validate_deploy
3. Create `create_gitlab_subgraph()` for issue/MR/merge_train
4. Create `create_notification_subgraph()` for summary/failure
5. Refactor `create_box_up_role_graph()` to compose subgraphs
6. Add subgraph-level checkpointing
7. Add subgraph-level error handling
8. Update tests for new structure
9. Add documentation for subgraph architecture

**Success Criteria**:
- [ ] Workflow split into 4 logical subgraphs
- [ ] Each subgraph independently testable
- [ ] Checkpointing works at subgraph boundaries
- [ ] No behavior changes from user perspective

---

### Agent 3.3: PostgreSQL Checkpointer
**Goal**: Production-ready checkpointing

**Tasks**:
1. Add `langgraph-checkpoint-postgres>=2.0.0` to pyproject.toml
2. Create `PostgresCheckpointerFactory` class
3. Add `POSTGRES_URL` environment variable support
4. Add connection pooling configuration
5. Add checkpoint cleanup job (`cleanup_old_checkpoints`)
6. Add migration script from SQLite to Postgres
7. Add fallback to SQLite for development
8. Add health check for Postgres connection
9. Add tests with testcontainers-postgres

**Success Criteria**:
- [ ] Checkpoints stored in Postgres in production
- [ ] SQLite fallback for local development
- [ ] Old checkpoints cleaned up automatically
- [ ] Migration path from existing SQLite data

---

### Agent 3.4: Context Engineering (Manus Patterns)
**Goal**: Implement Manus-inspired context management

**Tasks**:
1. Create `ContextManager` class in `hotl/context.py`
2. Implement `should_compact(token_count, limit)` check
3. Implement `compact_context(state)` summarization
4. Implement `inject_diversity(prompt)` for template variation
5. Add `COMPACTION_THRESHOLD = 0.25` constant
6. Add token counting to agent sessions
7. Integrate context manager into supervisor
8. Add file-based memory for large observations
9. Add tests for context management

**Success Criteria**:
- [ ] Context compacted before hitting 25% limit
- [ ] Large observations stored as files, references kept
- [ ] Prompt diversity prevents pattern lock-in
- [ ] Token count tracked per session

---

## Gap Analysis Checkpoint 3 (End of Week 6)

### Reality Checker Agent 3
**Scope**: Verify Phase 3 completion

**Checks**:
1. **Parallel Tests**: Time comparison before/after
2. **Subgraphs**: Each subgraph runs independently
3. **Postgres**: Checkpoints in Postgres (if configured)
4. **Context**: Token count doesn't exceed threshold
5. **Performance**: Overall workflow time improved
6. **Stability**: No flaky tests introduced

**Output**: Gap report with blockers for Phase 4

---

## Phase 4: Agent Protocols & Advanced Features (Week 7-8)

### Agent 4.1: A2A Protocol Implementation
**Goal**: Agent-to-agent communication protocol

**Tasks**:
1. Create `.claude/agent-card.json` with capabilities
2. Create `AgentCard` dataclass for parsing
3. Create `A2AClient` for agent discovery
4. Create `A2AServer` for receiving agent requests
5. Add `/a2a/discover` endpoint
6. Add `/a2a/invoke` endpoint
7. Add agent handoff support in supervisor
8. Add tests for A2A protocol
9. Add documentation for multi-agent coordination

**Success Criteria**:
- [ ] Agent card published and discoverable
- [ ] Can invoke other A2A-compatible agents
- [ ] Handoffs work between supervisor and workers
- [ ] Protocol follows A2A spec

---

### Agent 4.2: Cost Tracking & Optimization
**Goal**: Per-session token accounting

**Tasks**:
1. Add `token_usage` table to database schema
2. Add `track_token_usage` MCP tool
3. Add pricing constants for Claude models
4. Add `get_session_cost(session_id)` method
5. Add `get_total_cost(date_range)` method
6. Add cost alerts (notify if over budget)
7. Add model tier routing (Opus for planning, Haiku for execution)
8. Add CLI command `harness costs --report`
9. Add tests for cost tracking

**Success Criteria**:
- [ ] Token usage tracked per session
- [ ] Cost calculated with model-specific pricing
- [ ] Alerts when approaching budget
- [ ] Report shows cost breakdown

---

### Agent 4.3: Enhanced Hooks & Skills
**Goal**: Modernize hooks and skills framework

**Tasks**:
1. Add agent hooks (spawn subagent for verification)
2. Add `PreToolUse` input modification support
3. Add `SubagentStop` hook for audit
4. Create `file-change-tracker.py` hook
5. Create `subagent-audit.py` hook
6. Update settings.json with new hook patterns
7. Add skill for autonomous testing
8. Add skill for dependency analysis
9. Add tests for hooks and skills

**Success Criteria**:
- [ ] Agent hooks can spawn verification subagents
- [ ] Input modification works for sandboxing
- [ ] All file changes tracked via hooks
- [ ] New skills documented and working

---

### Agent 4.4: E2B Sandbox Integration (Optional)
**Goal**: Isolated execution for destructive operations

**Tasks**:
1. Add `e2b` to optional dependencies
2. Create `SandboxedExecution` class
3. Add `run_in_sandbox(task, files)` method
4. Add sandbox template with required tools
5. Add file sync in/out of sandbox
6. Add timeout and resource limits
7. Add fallback to local execution if E2B unavailable
8. Add configuration for sandbox-required operations
9. Add tests with E2B mocks

**Success Criteria**:
- [ ] Destructive operations run in sandbox
- [ ] Files synced correctly
- [ ] Graceful fallback when E2B unavailable
- [ ] Resource limits enforced

---

## Final Gap Analysis (End of Week 8)

### Reality Checker Agent 4
**Scope**: Full system validation

**Checks**:
1. **End-to-End**: Run complete `box-up-role` workflow
2. **HOTL Autonomous**: Run HOTL with real gap
3. **Idempotency**: Multiple runs don't create duplicates
4. **Performance**: Measure total workflow time
5. **Cost**: Calculate cost per workflow run
6. **Observability**: Verify traces in LangSmith
7. **Reliability**: Run 10x, count failures
8. **Documentation**: All features documented

**Output**: Final readiness report

---

## Success Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Workflow Time | ~10 min | <5 min | Parallel tests, SDK |
| Duplicate Prevention | 0% | 100% | Idempotency tests |
| Test Coverage | 272 tests | 350+ tests | pytest --cov |
| Context Efficiency | 100% | <25% | Token tracking |
| Cost per Workflow | Unknown | Tracked | Cost reporting |
| Reliability | Unknown | >95% | 10x run test |
| LangGraph Version | 0.2.x | 1.0.7+ | Version check |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SDK breaking changes | Keep old integration as fallback |
| Postgres unavailable | SQLite fallback for dev |
| E2B costs | Make sandbox optional |
| LangSmith not configured | Graceful degradation |
| A2A protocol immaturity | Implement subset, iterate |

---

## Appendix: Task Dependencies

```
Phase 1 (Parallel):
  1.1 SDK Migration ─────┐
  1.2 Idempotency ───────┼──▶ Gap Check 1
  1.3 LangGraph 1.0 ─────┤
  1.4 Label Management ──┘

Phase 2 (Parallel, after Gap Check 1):
  2.1 Iteration/Issue ───┐
  2.2 MR Lifecycle ──────┼──▶ Gap Check 2
  2.3 HITL Interrupts ───┤
  2.4 Observability ─────┘

Phase 3 (Parallel, after Gap Check 2):
  3.1 Parallel Tests ────┐
  3.2 Subgraphs ─────────┼──▶ Gap Check 3
  3.3 Postgres ──────────┤
  3.4 Context Mgmt ──────┘

Phase 4 (Parallel, after Gap Check 3):
  4.1 A2A Protocol ──────┐
  4.2 Cost Tracking ─────┼──▶ Final Gap Check
  4.3 Hooks/Skills ──────┤
  4.4 E2B Sandbox ───────┘
```
