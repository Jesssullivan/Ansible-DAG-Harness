"""Tests for CLI commands."""

import json
import pytest
from typer.testing import CliRunner
from harness.cli import app
from harness.db.state import StateDB
from harness.db.models import Role


runner = CliRunner()


@pytest.fixture
def cli_env(temp_db_path):
    """Set up environment for CLI tests to use the temp database."""
    # Initialize the database
    db = StateDB(temp_db_path)
    # Set env var so CLI uses this database
    env = {"HARNESS_DB_PATH": str(temp_db_path)}
    return db, env


class TestCheckCommand:
    """Test the 'check' command."""

    @pytest.mark.unit
    def test_check_on_fresh_db(self, cli_env):
        """Check command should pass on fresh database."""
        db, env = cli_env

        result = runner.invoke(app, ["check"], env=env)
        assert result.exit_code == 0
        assert "All checks passed" in result.stdout or "check" in result.stdout.lower()

    @pytest.mark.unit
    def test_check_json_output(self, cli_env):
        """Check command with JSON output."""
        db, env = cli_env

        result = runner.invoke(app, ["check", "--json"], env=env)
        assert result.exit_code == 0
        # Should be valid JSON
        output = json.loads(result.stdout)
        assert "passed" in output
        assert "checks" in output


class TestDbCommands:
    """Test database management commands."""

    @pytest.mark.unit
    def test_db_stats(self, cli_env):
        """Stats command shows statistics."""
        db, env = cli_env
        # Add some data
        db.upsert_role(Role(name="test_role", wave=1))

        result = runner.invoke(app, ["db", "stats"], env=env)
        assert result.exit_code == 0
        assert "roles" in result.stdout

    @pytest.mark.unit
    def test_db_stats_json(self, cli_env):
        """Stats command with JSON output."""
        db, env = cli_env

        result = runner.invoke(app, ["db", "stats", "--json"], env=env)
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert "roles" in output

    @pytest.mark.unit
    def test_db_clear_protected_table(self, cli_env):
        """Clear command should reject protected tables."""
        db, env = cli_env

        result = runner.invoke(app, ["db", "clear", "roles", "--yes"], env=env)
        assert result.exit_code == 1
        assert "Cannot clear protected table" in result.stdout

    @pytest.mark.unit
    def test_db_clear_allowed_table(self, cli_env):
        """Clear command should work for allowed tables."""
        db, env = cli_env
        db.log_audit("test", 1, "test_action")

        result = runner.invoke(app, ["db", "clear", "audit_log", "--yes"], env=env)
        assert result.exit_code == 0
        assert "Cleared" in result.stdout

    @pytest.mark.unit
    def test_db_vacuum(self, cli_env):
        """Vacuum command should work."""
        db, env = cli_env

        result = runner.invoke(app, ["db", "vacuum"], env=env)
        assert result.exit_code == 0
        assert "Vacuum complete" in result.stdout


class TestMetricsCommands:
    """Test metrics commands."""

    @pytest.mark.unit
    def test_metrics_status(self, cli_env):
        """Metrics status command."""
        db, env = cli_env

        result = runner.invoke(app, ["metrics", "status"], env=env)
        assert result.exit_code == 0
        assert "Overall Health" in result.stdout

    @pytest.mark.unit
    def test_metrics_list(self, cli_env):
        """Metrics list command."""
        db, env = cli_env

        result = runner.invoke(app, ["metrics", "list"], env=env)
        assert result.exit_code == 0
        assert "workflow_completion_time" in result.stdout

    @pytest.mark.unit
    def test_metrics_record(self, cli_env):
        """Record a metric value."""
        db, env = cli_env

        result = runner.invoke(app, ["metrics", "record", "workflow_completion_time", "100.0"], env=env)
        assert result.exit_code == 0
        assert "workflow_completion_time" in result.stdout

    @pytest.mark.unit
    def test_metrics_record_unknown(self, cli_env):
        """Recording unknown metric should fail."""
        db, env = cli_env

        result = runner.invoke(app, ["metrics", "record", "unknown_metric", "100.0"], env=env)
        assert result.exit_code == 1
        assert "Unknown metric" in result.stdout

    @pytest.mark.unit
    def test_metrics_history_empty(self, cli_env):
        """History with no data."""
        db, env = cli_env

        result = runner.invoke(app, ["metrics", "history", "workflow_completion_time"], env=env)
        assert result.exit_code == 0
        assert "No data" in result.stdout

    @pytest.mark.unit
    def test_metrics_history_with_data(self, cli_env):
        """History with recorded data."""
        db, env = cli_env
        from harness.metrics.golden import GoldenMetricsTracker
        tracker = GoldenMetricsTracker(db)
        tracker.record("workflow_completion_time", 100.0)

        result = runner.invoke(app, ["metrics", "history", "workflow_completion_time"], env=env)
        assert result.exit_code == 0
        assert "100.0" in result.stdout or "100.000" in result.stdout


class TestListRolesCommand:
    """Test list-roles command."""

    @pytest.mark.unit
    def test_list_roles_empty(self, cli_env):
        """List roles on empty database."""
        db, env = cli_env

        result = runner.invoke(app, ["list-roles"], env=env)
        assert result.exit_code == 0
        assert "No roles found" in result.stdout

    @pytest.mark.unit
    def test_list_roles_with_data(self, cli_env):
        """List roles with data."""
        db, env = cli_env
        db.upsert_role(Role(name="common", wave=0, has_molecule_tests=True))
        db.upsert_role(Role(name="sql_server", wave=2, has_molecule_tests=False))

        result = runner.invoke(app, ["list-roles"], env=env)
        assert result.exit_code == 0
        assert "common" in result.stdout
        assert "sql_server" in result.stdout

    @pytest.mark.unit
    def test_list_roles_filter_wave(self, cli_env):
        """Filter roles by wave."""
        db, env = cli_env
        db.upsert_role(Role(name="common", wave=0, has_molecule_tests=True))
        db.upsert_role(Role(name="sql_server", wave=2, has_molecule_tests=False))

        result = runner.invoke(app, ["list-roles", "--wave", "2"], env=env)
        assert result.exit_code == 0
        assert "sql_server" in result.stdout


class TestDepsCommand:
    """Test deps command."""

    @pytest.mark.unit
    def test_deps_empty(self, cli_env):
        """Dependencies for role with none."""
        db, env = cli_env
        db.upsert_role(Role(name="common", wave=0))

        result = runner.invoke(app, ["deps", "common"], env=env)
        assert result.exit_code == 0
        assert "No" in result.stdout and "dependencies" in result.stdout.lower()


class TestStatusCommand:
    """Test status command."""

    @pytest.mark.unit
    def test_status_no_roles(self, cli_env):
        """Status with no roles."""
        db, env = cli_env

        result = runner.invoke(app, ["status"], env=env)
        assert result.exit_code == 0
        assert "No roles" in result.stdout

    @pytest.mark.unit
    def test_status_with_role(self, cli_env):
        """Status for specific role."""
        db, env = cli_env
        db.upsert_role(Role(name="common", wave=0))

        result = runner.invoke(app, ["status", "common"], env=env)
        assert result.exit_code == 0
        assert "common" in result.stdout

    @pytest.mark.unit
    def test_status_nonexistent_role(self, cli_env):
        """Status for non-existent role."""
        db, env = cli_env

        result = runner.invoke(app, ["status", "nonexistent"], env=env)
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()
