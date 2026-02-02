"""Tests for StateDB CRUD operations."""

import pytest
from hypothesis import given, settings

from harness.db.models import (
    Credential,
    Role,
    TestRun,
    TestStatus,
    TestType,
    Worktree,
    WorktreeStatus,
)
from harness.db.state import StateDB
from tests.strategies import (
    role_strategy,
    wave_strategy,
)


class TestRoleOperations:
    """Test role CRUD operations."""

    @pytest.mark.unit
    def test_upsert_role_creates_new(self, db: StateDB):
        """Upsert should create a new role."""
        role = Role(name="test_role", wave=1, has_molecule_tests=True)
        role_id = db.upsert_role(role)

        assert role_id > 0
        retrieved = db.get_role("test_role")
        assert retrieved is not None
        assert retrieved.name == "test_role"
        assert retrieved.wave == 1
        assert retrieved.has_molecule_tests is True

    @pytest.mark.unit
    def test_upsert_role_updates_existing(self, db: StateDB):
        """Upsert should update existing role."""
        role = Role(name="test_role", wave=1)
        db.upsert_role(role)

        updated = Role(name="test_role", wave=2, description="Updated description")
        db.upsert_role(updated)

        retrieved = db.get_role("test_role")
        assert retrieved.wave == 2
        assert retrieved.description == "Updated description"

    @pytest.mark.unit
    def test_get_role_not_found(self, db: StateDB):
        """Get role returns None for non-existent role."""
        result = db.get_role("nonexistent")
        assert result is None

    @pytest.mark.unit
    def test_get_role_by_id(self, db: StateDB):
        """Get role by ID."""
        role = Role(name="test_role", wave=1)
        role_id = db.upsert_role(role)

        retrieved = db.get_role_by_id(role_id)
        assert retrieved is not None
        assert retrieved.name == "test_role"

    @pytest.mark.unit
    def test_list_roles_empty(self, db: StateDB):
        """List roles on empty db returns empty list."""
        roles = db.list_roles()
        assert roles == []

    @pytest.mark.unit
    def test_list_roles_all(self, db_with_roles: StateDB):
        """List all roles."""
        roles = db_with_roles.list_roles()
        assert len(roles) == 5
        role_names = {r.name for r in roles}
        assert "common" in role_names
        assert "sql_server_2022" in role_names

    @pytest.mark.unit
    def test_list_roles_filter_by_wave(self, db_with_roles: StateDB):
        """Filter roles by wave number."""
        wave_2_roles = db_with_roles.list_roles(wave=2)
        assert len(wave_2_roles) == 3
        assert all(r.wave == 2 for r in wave_2_roles)

    @pytest.mark.unit
    def test_list_roles_ordered_by_wave(self, db_with_roles: StateDB):
        """Roles should be ordered by wave, then name."""
        roles = db_with_roles.list_roles()
        waves = [r.wave for r in roles]
        assert waves == sorted(waves)

    @pytest.mark.pbt
    @given(role=role_strategy())
    @settings(max_examples=50)
    def test_upsert_role_roundtrip_pbt(self, role: Role):
        """Property: upsert then get should return equivalent role."""
        # Create fresh db for each hypothesis example
        db = StateDB(":memory:")
        db.upsert_role(role)
        retrieved = db.get_role(role.name)

        assert retrieved is not None
        assert retrieved.name == role.name
        assert retrieved.wave == role.wave
        assert retrieved.has_molecule_tests == role.has_molecule_tests

    @pytest.mark.pbt
    @given(wave=wave_strategy)
    @settings(max_examples=20)
    def test_wave_constraint_pbt(self, wave: int):
        """Property: wave must be between 0 and 4."""
        # Create fresh db for each hypothesis example
        db = StateDB(":memory:")
        role = Role(name=f"role_wave_{wave}", wave=wave)
        db.upsert_role(role)
        retrieved = db.get_role(role.name)
        assert retrieved.wave == wave


class TestCredentialOperations:
    """Test credential operations."""

    @pytest.mark.unit
    def test_add_and_get_credentials(self, db_with_roles: StateDB):
        """Add and retrieve credentials for a role."""
        role = db_with_roles.get_role("common")
        cred = Credential(
            role_id=role.id, entry_name="ansible-self", purpose="WinRM auth", is_base58=True
        )
        db_with_roles.add_credential(cred)

        credentials = db_with_roles.get_credentials("common")
        assert len(credentials) == 1
        assert credentials[0].entry_name == "ansible-self"
        assert credentials[0].is_base58 is True

    @pytest.mark.unit
    def test_get_credentials_empty(self, db_with_roles: StateDB):
        """Get credentials for role with none returns empty list."""
        credentials = db_with_roles.get_credentials("ems_web_app")
        assert credentials == []

    @pytest.mark.unit
    def test_get_credentials_nonexistent_role(self, db: StateDB):
        """Get credentials for non-existent role returns empty list."""
        credentials = db.get_credentials("nonexistent")
        assert credentials == []

    @pytest.mark.unit
    def test_multiple_credentials(self, db_with_credentials: StateDB):
        """Role can have multiple credentials."""
        credentials = db_with_credentials.get_credentials("sql_server_2022")
        assert len(credentials) == 2
        entry_names = {c.entry_name for c in credentials}
        assert "dev-sql-sa" in entry_names
        assert "test-windows-admin" in entry_names

    @pytest.mark.unit
    def test_credential_upsert(self, db_with_roles: StateDB):
        """Credential upsert updates existing."""
        role = db_with_roles.get_role("common")
        cred1 = Credential(
            role_id=role.id, entry_name="test-cred", purpose="Initial purpose", is_base58=False
        )
        db_with_roles.add_credential(cred1)

        cred2 = Credential(
            role_id=role.id, entry_name="test-cred", purpose="Updated purpose", is_base58=True
        )
        db_with_roles.add_credential(cred2)

        credentials = db_with_roles.get_credentials("common")
        assert len(credentials) == 1
        assert credentials[0].purpose == "Updated purpose"
        assert credentials[0].is_base58 is True


class TestWorktreeOperations:
    """Test worktree operations."""

    @pytest.mark.unit
    def test_upsert_and_get_worktree(self, db_with_roles: StateDB):
        """Upsert and retrieve worktree."""
        role = db_with_roles.get_role("common")
        worktree = Worktree(
            role_id=role.id,
            path="../.worktrees/sid-common",
            branch="sid/common",
            status=WorktreeStatus.ACTIVE,
        )
        db_with_roles.upsert_worktree(worktree)

        retrieved = db_with_roles.get_worktree("common")
        assert retrieved is not None
        assert retrieved.branch == "sid/common"
        assert retrieved.status == WorktreeStatus.ACTIVE

    @pytest.mark.unit
    def test_get_worktree_not_found(self, db_with_roles: StateDB):
        """Get worktree for role without one returns None."""
        result = db_with_roles.get_worktree("ems_web_app")
        assert result is None

    @pytest.mark.unit
    def test_list_worktrees(self, db_with_worktrees: StateDB):
        """List all worktrees."""
        worktrees = db_with_worktrees.list_worktrees()
        assert len(worktrees) == 2

    @pytest.mark.unit
    def test_list_worktrees_by_status(self, db_with_worktrees: StateDB):
        """Filter worktrees by status."""
        active = db_with_worktrees.list_worktrees(status=WorktreeStatus.ACTIVE)
        dirty = db_with_worktrees.list_worktrees(status=WorktreeStatus.DIRTY)

        assert len(active) == 1
        assert len(dirty) == 1
        assert active[0].branch == "sid/common"
        assert dirty[0].branch == "sid/sql_server_2022"

    @pytest.mark.unit
    def test_worktree_commits_tracking(self, db_with_worktrees: StateDB):
        """Worktree tracks commits ahead/behind."""
        common_wt = db_with_worktrees.get_worktree("common")
        sql_wt = db_with_worktrees.get_worktree("sql_server_2022")

        assert common_wt.commits_ahead == 2
        assert common_wt.commits_behind == 0
        assert sql_wt.commits_behind == 5
        assert sql_wt.uncommitted_changes == 3


class TestRoleStatusView:
    """Test role status aggregation."""

    @pytest.mark.unit
    def test_get_role_status(self, db_with_worktrees: StateDB):
        """Get aggregated role status."""
        status = db_with_worktrees.get_role_status("common")
        assert status is not None
        assert status.name == "common"
        assert status.wave == 0
        assert status.worktree_status == "active"
        assert status.commits_ahead == 2

    @pytest.mark.unit
    def test_list_role_statuses(self, db_with_worktrees: StateDB):
        """List all role statuses."""
        statuses = db_with_worktrees.list_role_statuses()
        assert len(statuses) == 5

        # Find common and check its status
        common = next(s for s in statuses if s.name == "common")
        assert common.worktree_status == "active"


class TestExecutionContextOperations:
    """Test execution context (SEE/ACP) operations."""

    @pytest.mark.unit
    def test_create_context(self, db: StateDB):
        """Create a new execution context."""
        context_id = db.create_context(
            session_id="test-session-123",
            user_id="jsullivan2",
            capabilities=["read:roles", "read:worktrees"],
        )

        assert context_id > 0

        context = db.get_context("test-session-123")
        assert context is not None
        assert context.session_id == "test-session-123"
        assert context.user_id == "jsullivan2"

    @pytest.mark.unit
    def test_grant_capability(self, db: StateDB):
        """Grant capability to context."""
        context_id = db.create_context(session_id="test-session")
        db.grant_capability(context_id, "execute:workflow", scope="common")

        assert db.check_capability(context_id, "execute:workflow", scope="common")

    @pytest.mark.unit
    def test_revoke_capability(self, db: StateDB):
        """Revoke capability from context."""
        context_id = db.create_context(session_id="test-session")
        db.grant_capability(context_id, "execute:workflow")

        assert db.check_capability(context_id, "execute:workflow")

        db.revoke_capability(context_id, "execute:workflow")
        assert not db.check_capability(context_id, "execute:workflow")

    @pytest.mark.unit
    def test_check_capability_with_scope(self, db: StateDB):
        """Check capability respects scope."""
        context_id = db.create_context(session_id="test-session")
        db.grant_capability(context_id, "execute:workflow", scope="common")

        # Should have capability for common
        assert db.check_capability(context_id, "execute:workflow", scope="common")
        # Should not have capability for different role
        assert not db.check_capability(context_id, "execute:workflow", scope="sql_server_2022")

    @pytest.mark.unit
    def test_get_context_capabilities(self, db: StateDB):
        """Get all capabilities for a context."""
        context_id = db.create_context(session_id="test-session")
        db.grant_capability(context_id, "read:roles")
        db.grant_capability(context_id, "read:worktrees")
        db.grant_capability(context_id, "execute:sync")

        capabilities = db.get_context_capabilities(context_id)
        assert len(capabilities) == 3
        cap_names = {c.capability for c in capabilities}
        assert "read:roles" in cap_names
        assert "read:worktrees" in cap_names
        assert "execute:sync" in cap_names


class TestToolInvocationTracking:
    """Test tool invocation logging."""

    @pytest.mark.unit
    def test_log_tool_invocation(self, db: StateDB):
        """Log a tool invocation."""
        context_id = db.create_context(session_id="test-session")
        invocation_id = db.log_tool_invocation(
            context_id=context_id, tool_name="list_roles", arguments={"wave": 2}
        )

        assert invocation_id > 0

    @pytest.mark.unit
    def test_complete_tool_invocation(self, db: StateDB):
        """Complete a tool invocation."""
        context_id = db.create_context(session_id="test-session")
        invocation_id = db.log_tool_invocation(context_id=context_id, tool_name="list_roles")

        db.complete_tool_invocation(
            invocation_id,
            result={"roles": ["common", "sql_server_2022"]},
            status="completed",
            duration_ms=150,
        )

        # Verify completion (would need a get_invocation method to fully test)


class TestAuditLogging:
    """Test audit log functionality."""

    @pytest.mark.unit
    def test_log_audit(self, db: StateDB):
        """Log an audit entry."""
        db.log_audit(
            entity_type="role",
            entity_id=1,
            action="create",
            new_value={"name": "test_role", "wave": 1},
        )

        # Verify audit was logged (would need a get_audit_log method)

    @pytest.mark.unit
    def test_audit_with_old_value(self, db: StateDB):
        """Log audit with old and new values."""
        db.log_audit(
            entity_type="role",
            entity_id=1,
            action="update",
            old_value={"wave": 1},
            new_value={"wave": 2},
            actor="test_user",
        )


class TestTestRegressionTracking:
    """Test regression tracking operations."""

    @pytest.mark.unit
    def test_record_test_failure(self, db_with_roles: StateDB):
        """Record a test failure creates regression."""
        # Create a test run first
        role = db_with_roles.get_role("common")
        test_run = TestRun(role_id=role.id, test_type=TestType.MOLECULE, status=TestStatus.FAILED)
        run_id = db_with_roles.create_test_run(test_run)

        # Record failure
        regression_id = db_with_roles.record_test_failure(
            role_name="common",
            test_name="molecule:common",
            test_type=TestType.MOLECULE,
            test_run_id=run_id,
            error_message="Test failed",
        )

        assert regression_id > 0

    @pytest.mark.unit
    def test_record_test_success_resolves_regression(self, db_with_roles: StateDB):
        """Recording success after failures resolves regression."""
        role = db_with_roles.get_role("common")

        # Create failed run
        failed_run = TestRun(role_id=role.id, test_type=TestType.MOLECULE, status=TestStatus.FAILED)
        failed_run_id = db_with_roles.create_test_run(failed_run)

        # Record multiple failures
        for _ in range(3):
            db_with_roles.record_test_failure(
                role_name="common",
                test_name="molecule:common",
                test_type=TestType.MOLECULE,
                test_run_id=failed_run_id,
            )

        # Create passing run
        passed_run = TestRun(role_id=role.id, test_type=TestType.MOLECULE, status=TestStatus.PASSED)
        passed_run_id = db_with_roles.create_test_run(passed_run)

        # Record success
        db_with_roles.record_test_success(
            role_name="common",
            test_name="molecule:common",
            test_type=TestType.MOLECULE,
            test_run_id=passed_run_id,
        )

        # Regression should be resolved
        regression = db_with_roles.get_regression(
            role_name="common", test_name="molecule:common", test_type=TestType.MOLECULE
        )
        assert regression.status.value == "resolved"

    @pytest.mark.unit
    def test_get_active_regressions(self, db_with_roles: StateDB):
        """Get active regressions."""
        role = db_with_roles.get_role("common")
        test_run = TestRun(role_id=role.id, test_type=TestType.MOLECULE, status=TestStatus.FAILED)
        run_id = db_with_roles.create_test_run(test_run)

        db_with_roles.record_test_failure(
            role_name="common",
            test_name="molecule:common",
            test_type=TestType.MOLECULE,
            test_run_id=run_id,
        )

        regressions = db_with_roles.get_active_regressions()
        assert len(regressions) == 1
        assert regressions[0].role_name == "common"


class TestMergeTrainOperations:
    """Test merge train state tracking."""

    @pytest.mark.unit
    def test_add_to_merge_train(self, db_with_roles: StateDB):
        """Add MR to merge train."""
        # Would need an MR first - just test the method exists
        # This is more of an integration test
        pass

    @pytest.mark.unit
    def test_list_merge_train(self, db_with_roles: StateDB):
        """List merge train entries."""
        entries = db_with_roles.list_merge_train()
        assert entries == []  # Empty initially
