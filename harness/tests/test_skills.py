"""Tests for the skills framework.

Tests cover:
- Base skill classes
- SkillRegistry
- TestingSkill
- DependencySkill
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from harness.skills.base import (
    Skill,
    SkillAction,
    SkillContext,
    SkillRegistry,
    SkillResult,
    SkillStatus,
    create_simple_skill,
)
from harness.skills.dependencies import (
    DependencyInfo,
    DependencySkill,
    DependencyUpdate,
)
from harness.skills.testing import (
    TestFailure,
    TestingSkill,
    TestResult,
)

# ============================================================================
# BASE SKILL TESTS
# ============================================================================


class TestSkillContext:
    """Tests for SkillContext."""

    def test_create_context(self, tmp_path):
        """Test creating a skill context."""
        context = SkillContext(working_dir=tmp_path)
        assert context.working_dir == tmp_path
        assert context.timeout == 300
        assert context.env == {}

    def test_context_with_all_fields(self, tmp_path):
        """Test context with all fields."""
        context = SkillContext(
            working_dir=tmp_path,
            agent_id="agent-123",
            session_id="session-456",
            execution_id=789,
            timeout=60,
            env={"TEST": "value"},
            metadata={"custom": "data"},
        )
        assert context.agent_id == "agent-123"
        assert context.session_id == "session-456"
        assert context.execution_id == 789
        assert context.timeout == 60
        assert context.env == {"TEST": "value"}
        assert context.metadata == {"custom": "data"}


class TestSkillResult:
    """Tests for SkillResult."""

    def test_create_result(self):
        """Test creating a skill result."""
        result = SkillResult(
            status=SkillStatus.SUCCESS,
            message="Done",
        )
        assert result.status == SkillStatus.SUCCESS
        assert result.message == "Done"

    def test_success_factory(self):
        """Test success factory method."""
        result = SkillResult.success("All good", data={"count": 5})
        assert result.status == SkillStatus.SUCCESS
        assert result.message == "All good"
        assert result.data == {"count": 5}

    def test_failure_factory(self):
        """Test failure factory method."""
        result = SkillResult.failure("Failed", errors=["Error 1"])
        assert result.status == SkillStatus.FAILURE
        assert result.message == "Failed"
        assert result.errors == ["Error 1"]

    def test_error_factory(self):
        """Test error factory method."""
        exc = ValueError("Bad value")
        result = SkillResult.error("Error occurred", exception=exc)
        assert result.status == SkillStatus.ERROR
        assert "Bad value" in result.errors[0]

    def test_to_dict(self):
        """Test result serialization."""
        result = SkillResult(
            status=SkillStatus.SUCCESS,
            message="Done",
            data={"key": "value"},
            duration_ms=100,
        )
        data = result.to_dict()
        assert data["status"] == "success"
        assert data["message"] == "Done"
        assert data["data"] == {"key": "value"}
        assert data["duration_ms"] == 100


class TestSkillAction:
    """Tests for SkillAction."""

    def test_create_action(self):
        """Test creating a skill action."""

        async def handler(context, params):
            return SkillResult.success()

        action = SkillAction(
            name="test",
            description="Test action",
            handler=handler,
            parameters=["param1", "param2"],
            required_params=["param1"],
        )
        assert action.name == "test"
        assert action.description == "Test action"
        assert "param1" in action.parameters

    def test_validate_params_success(self):
        """Test parameter validation success."""
        action = SkillAction(
            name="test",
            description="",
            handler=lambda c, p: None,
            required_params=["required"],
        )
        missing = action.validate_params({"required": "value"})
        assert missing == []

    def test_validate_params_missing(self):
        """Test parameter validation with missing params."""
        action = SkillAction(
            name="test",
            description="",
            handler=lambda c, p: None,
            required_params=["required1", "required2"],
        )
        missing = action.validate_params({"required1": "value"})
        assert missing == ["required2"]


# ============================================================================
# CONCRETE SKILL FOR TESTING
# ============================================================================


class TestSkillImpl(Skill):
    """Test implementation of Skill."""

    @property
    def name(self) -> str:
        return "test_skill"

    @property
    def description(self) -> str:
        return "A test skill"

    def _register_actions(self) -> None:
        self.register_action(
            SkillAction(
                name="succeed",
                description="Always succeeds",
                handler=self._succeed,
            )
        )
        self.register_action(
            SkillAction(
                name="fail",
                description="Always fails",
                handler=self._fail,
            )
        )
        self.register_action(
            SkillAction(
                name="with_params",
                description="Requires params",
                handler=self._with_params,
                parameters=["required", "optional"],
                required_params=["required"],
            )
        )

    async def _succeed(self, context, params):
        return SkillResult.success("Success!")

    async def _fail(self, context, params):
        return SkillResult.failure("Failed!")

    async def _with_params(self, context, params):
        return SkillResult.success(f"Got: {params.get('required')}")


class TestSkill:
    """Tests for base Skill class."""

    @pytest.fixture
    def skill(self):
        """Create a test skill."""
        return TestSkillImpl()

    @pytest.fixture
    def context(self, tmp_path):
        """Create a skill context."""
        return SkillContext(working_dir=tmp_path)

    def test_skill_properties(self, skill):
        """Test skill properties."""
        assert skill.name == "test_skill"
        assert skill.description == "A test skill"
        assert skill.enabled is True

    def test_get_actions(self, skill):
        """Test getting action names."""
        actions = skill.get_actions()
        assert "succeed" in actions
        assert "fail" in actions
        assert "with_params" in actions

    def test_get_action(self, skill):
        """Test getting an action by name."""
        action = skill.get_action("succeed")
        assert action is not None
        assert action.name == "succeed"

    def test_get_action_not_found(self, skill):
        """Test getting non-existent action."""
        assert skill.get_action("nonexistent") is None

    @pytest.mark.asyncio
    async def test_execute_success(self, skill, context):
        """Test executing a successful action."""
        result = await skill.execute("succeed", context)
        assert result.status == SkillStatus.SUCCESS
        assert result.duration_ms >= 0  # May be 0 for very fast operations

    @pytest.mark.asyncio
    async def test_execute_failure(self, skill, context):
        """Test executing a failing action."""
        result = await skill.execute("fail", context)
        assert result.status == SkillStatus.FAILURE

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, skill, context):
        """Test executing unknown action."""
        result = await skill.execute("nonexistent", context)
        assert result.status == SkillStatus.ERROR
        assert "Unknown action" in result.message

    @pytest.mark.asyncio
    async def test_execute_missing_params(self, skill, context):
        """Test executing with missing required params."""
        result = await skill.execute("with_params", context, {})
        assert result.status == SkillStatus.ERROR
        assert "Missing required parameters" in result.message

    @pytest.mark.asyncio
    async def test_execute_with_params(self, skill, context):
        """Test executing with params."""
        result = await skill.execute("with_params", context, {"required": "value"})
        assert result.status == SkillStatus.SUCCESS
        assert "value" in result.message

    @pytest.mark.asyncio
    async def test_execute_disabled(self, skill, context):
        """Test executing when skill is disabled."""
        skill.enabled = False
        result = await skill.execute("succeed", context)
        assert result.status == SkillStatus.SKIPPED

    def test_get_stats(self, skill):
        """Test getting skill statistics."""
        stats = skill.get_stats()
        assert "executions" in stats
        assert "successes" in stats
        assert "failures" in stats

    @pytest.mark.asyncio
    async def test_stats_updated(self, skill, context):
        """Test that stats are updated after execution."""
        await skill.execute("succeed", context)
        await skill.execute("fail", context)

        stats = skill.get_stats()
        assert stats["executions"] == 2
        assert stats["successes"] == 1
        assert stats["failures"] == 1

    def test_get_info(self, skill):
        """Test getting skill info."""
        info = skill.get_info()
        assert info["name"] == "test_skill"
        assert info["description"] == "A test skill"
        assert len(info["actions"]) == 3


# ============================================================================
# SKILL REGISTRY TESTS
# ============================================================================


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a skill registry."""
        return SkillRegistry()

    @pytest.fixture
    def context(self, tmp_path):
        """Create a skill context."""
        return SkillContext(working_dir=tmp_path)

    def test_register_skill(self, registry):
        """Test registering a skill."""
        skill = TestSkillImpl()
        registry.register(skill)
        assert "test_skill" in registry.list_skills()

    def test_unregister_skill(self, registry):
        """Test unregistering a skill."""
        skill = TestSkillImpl()
        registry.register(skill)
        assert registry.unregister("test_skill") is True
        assert "test_skill" not in registry.list_skills()

    def test_unregister_nonexistent(self, registry):
        """Test unregistering non-existent skill."""
        assert registry.unregister("nonexistent") is False

    def test_get_skill(self, registry):
        """Test getting a skill by name."""
        skill = TestSkillImpl()
        registry.register(skill)
        retrieved = registry.get_skill("test_skill")
        assert retrieved is skill

    def test_get_skill_not_found(self, registry):
        """Test getting non-existent skill."""
        assert registry.get_skill("nonexistent") is None

    def test_list_skills(self, registry):
        """Test listing skills."""
        registry.register(TestSkillImpl())
        skills = registry.list_skills()
        assert "test_skill" in skills

    def test_get_all_actions(self, registry):
        """Test getting all actions."""
        registry.register(TestSkillImpl())
        actions = registry.get_all_actions()
        assert "test_skill" in actions
        assert "succeed" in actions["test_skill"]

    @pytest.mark.asyncio
    async def test_execute_skill(self, registry, context):
        """Test executing a skill through registry."""
        registry.register(TestSkillImpl())
        result = await registry.execute("test_skill", "succeed", context)
        assert result.status == SkillStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_unknown_skill(self, registry, context):
        """Test executing unknown skill."""
        result = await registry.execute("nonexistent", "action", context)
        assert result.status == SkillStatus.ERROR
        assert "Unknown skill" in result.message

    def test_get_info(self, registry):
        """Test getting registry info."""
        registry.register(TestSkillImpl())
        info = registry.get_info()
        assert info["total_skills"] == 1
        assert len(info["skills"]) == 1

    def test_get_stats(self, registry):
        """Test getting registry stats."""
        registry.register(TestSkillImpl())
        stats = registry.get_stats()
        assert "test_skill" in stats


class TestCreateSimpleSkill:
    """Tests for create_simple_skill helper."""

    @pytest.fixture
    def context(self, tmp_path):
        """Create a skill context."""
        return SkillContext(working_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_create_simple_skill(self, context):
        """Test creating a simple skill."""

        async def action1(ctx, params):
            return SkillResult.success("Action 1")

        async def action2(ctx, params):
            return SkillResult.success("Action 2")

        skill = create_simple_skill(
            name="simple",
            description="A simple skill",
            actions={"action1": action1, "action2": action2},
        )

        assert skill.name == "simple"
        assert "action1" in skill.get_actions()

        result = await skill.execute("action1", context)
        assert result.status == SkillStatus.SUCCESS


# ============================================================================
# TESTING SKILL TESTS
# ============================================================================


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_to_dict(self):
        """Test TestResult serialization."""
        result = TestResult(
            passed=10,
            failed=2,
            skipped=1,
            total=13,
            duration_s=5.5,
        )
        data = result.to_dict()
        assert data["passed"] == 10
        assert data["failed"] == 2
        assert data["total"] == 13

    def test_success_rate(self):
        """Test success rate calculation."""
        result = TestResult(passed=8, failed=2, total=10)
        assert result.success_rate == 80.0

    def test_success_rate_zero_total(self):
        """Test success rate with zero total."""
        result = TestResult()
        assert result.success_rate == 0.0


class TestTestFailure:
    """Tests for TestFailure dataclass."""

    def test_to_dict(self):
        """Test TestFailure serialization."""
        failure = TestFailure(
            test_name="test_example",
            test_file="tests/test_example.py",
            error_message="AssertionError",
            failure_type="assertion",
        )
        data = failure.to_dict()
        assert data["test_name"] == "test_example"
        assert data["failure_type"] == "assertion"


class TestTestingSkill:
    """Tests for TestingSkill."""

    @pytest.fixture
    def skill(self):
        """Create a testing skill."""
        return TestingSkill()

    @pytest.fixture
    def context(self, tmp_path):
        """Create a skill context."""
        return SkillContext(working_dir=tmp_path, timeout=60)

    def test_skill_properties(self, skill):
        """Test skill properties."""
        assert skill.name == "testing"
        assert "run_tests" in skill.get_actions()
        assert "analyze_failures" in skill.get_actions()

    @pytest.mark.asyncio
    async def test_run_tests_no_tests(self, skill, context, tmp_path):
        """Test running tests when no tests exist."""
        # Create empty test directory
        (tmp_path / "tests").mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="===== no tests ran =====",
                stderr="",
            )

            result = await skill.execute("run_tests", context, {"path": "tests"})

            # Should handle no tests gracefully
            assert result.status in (SkillStatus.SKIPPED, SkillStatus.SUCCESS)

    @pytest.mark.asyncio
    async def test_run_tests_success(self, skill, context):
        """Test running tests successfully."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="5 passed in 1.5s",
                stderr="",
            )

            result = await skill.execute("run_tests", context, {"path": "."})

            assert result.status == SkillStatus.SUCCESS
            assert result.data["passed"] == 5

    @pytest.mark.asyncio
    async def test_run_tests_failure(self, skill, context):
        """Test running tests with failures."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="3 passed, 2 failed in 2.0s",
                stderr="",
            )

            result = await skill.execute("run_tests", context, {"path": "."})

            assert result.status == SkillStatus.FAILURE
            assert result.data["passed"] == 3
            assert result.data["failed"] == 2

    @pytest.mark.asyncio
    async def test_analyze_failures(self, skill, context):
        """Test analyzing failures."""
        test_result = {
            "failures": [
                {
                    "test_name": "test_example",
                    "error_message": "AssertionError: expected 1, got 2",
                    "failure_type": "failure",
                },
                {
                    "test_name": "test_import",
                    "error_message": "ImportError: No module named 'missing'",
                    "failure_type": "error",
                },
            ]
        }

        result = await skill.execute(
            "analyze_failures",
            context,
            {"test_result": test_result, "include_suggestions": True},
        )

        assert result.status == SkillStatus.SUCCESS
        assert len(result.data["analysis"]) == 2
        assert len(result.data["suggestions"]) >= 1


# ============================================================================
# DEPENDENCY SKILL TESTS
# ============================================================================


class TestDependencyInfo:
    """Tests for DependencyInfo dataclass."""

    def test_to_dict(self):
        """Test DependencyInfo serialization."""
        dep = DependencyInfo(
            name="requests",
            version="2.28.0",
            required_version=">=2.0",
            source="pyproject.toml",
            is_direct=True,
        )
        data = dep.to_dict()
        assert data["name"] == "requests"
        assert data["version"] == "2.28.0"
        assert data["is_direct"] is True


class TestDependencyUpdate:
    """Tests for DependencyUpdate dataclass."""

    def test_to_dict(self):
        """Test DependencyUpdate serialization."""
        update = DependencyUpdate(
            name="requests",
            current_version="2.28.0",
            latest_version="2.31.0",
            update_type="minor",
            breaking=False,
        )
        data = update.to_dict()
        assert data["name"] == "requests"
        assert data["update_type"] == "minor"
        assert data["breaking"] is False


class TestDependencySkill:
    """Tests for DependencySkill."""

    @pytest.fixture
    def skill(self):
        """Create a dependency skill."""
        return DependencySkill()

    @pytest.fixture
    def context(self, tmp_path):
        """Create a skill context."""
        return SkillContext(working_dir=tmp_path, timeout=60)

    def test_skill_properties(self, skill):
        """Test skill properties."""
        assert skill.name == "dependencies"
        assert "analyze_deps" in skill.get_actions()
        assert "check_updates" in skill.get_actions()

    @pytest.mark.asyncio
    async def test_analyze_deps_pyproject(self, skill, context, tmp_path):
        """Test analyzing dependencies from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test"
dependencies = [
    "requests>=2.0",
    "click",
]

[project.optional-dependencies]
dev = ["pytest"]
""")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"name": "requests", "version": "2.28.0"},
                        {"name": "click", "version": "8.0.0"},
                    ]
                ),
            )

            result = await skill.execute("analyze_deps", context)

            assert result.status == SkillStatus.SUCCESS
            assert result.data["direct_count"] >= 2

    @pytest.mark.asyncio
    async def test_analyze_deps_requirements(self, skill, context, tmp_path):
        """Test analyzing dependencies from requirements.txt."""
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("""
requests>=2.0
click==8.0.0
# Comment
-e git+https://github.com/example/repo.git
""")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"name": "requests", "version": "2.28.0"},
                        {"name": "click", "version": "8.0.0"},
                    ]
                ),
            )

            result = await skill.execute("analyze_deps", context)

            assert result.status == SkillStatus.SUCCESS
            deps = result.data["dependencies"]
            dep_names = [d["name"] for d in deps]
            assert "requests" in dep_names
            assert "click" in dep_names

    @pytest.mark.asyncio
    async def test_check_updates(self, skill, context):
        """Test checking for updates."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "name": "requests",
                            "version": "2.28.0",
                            "latest_version": "2.31.0",
                        },
                        {
                            "name": "django",
                            "version": "3.2.0",
                            "latest_version": "4.2.0",
                        },
                    ]
                ),
            )

            result = await skill.execute("check_updates", context)

            assert result.status == SkillStatus.SUCCESS
            assert len(result.data["updates"]) == 2
            assert result.data["major_updates"] >= 1

    @pytest.mark.asyncio
    async def test_check_updates_all_current(self, skill, context):
        """Test when all dependencies are current."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="[]",
            )

            result = await skill.execute("check_updates", context)

            assert result.status == SkillStatus.SUCCESS
            assert result.data["updates"] == []

    @pytest.mark.asyncio
    async def test_find_unused(self, skill, context, tmp_path):
        """Test finding unused dependencies."""
        # Create requirements
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("requests\nunused_package\n")

        # Create source file that imports requests
        source = tmp_path / "main.py"
        source.write_text("import requests\n")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"name": "requests", "version": "2.28.0"},
                        {"name": "unused_package", "version": "1.0.0"},
                    ]
                ),
            )

            result = await skill.execute("find_unused", context, {"source_dir": "."})

            assert result.status == SkillStatus.SUCCESS
            assert "unused_package" in result.data["potentially_unused"]

    @pytest.mark.asyncio
    async def test_security_check_no_vulnerabilities(self, skill, context):
        """Test security check with no vulnerabilities."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="[]",
            )

            result = await skill.execute("security_check", context)

            assert result.status == SkillStatus.SUCCESS
            assert result.data["vulnerabilities"] == []

    @pytest.mark.asyncio
    async def test_security_check_with_vulnerabilities(self, skill, context):
        """Test security check finding vulnerabilities."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout=json.dumps(
                    [
                        {
                            "name": "vulnerable-pkg",
                            "version": "1.0.0",
                            "id": "CVE-2024-1234",
                            "description": "Security issue",
                        }
                    ]
                ),
            )

            result = await skill.execute("security_check", context)

            assert result.status == SkillStatus.FAILURE
            assert len(result.data["vulnerabilities"]) == 1

    @pytest.mark.asyncio
    async def test_analyze_ansible(self, skill, context, tmp_path):
        """Test analyzing Ansible role dependencies."""
        # Create role structure
        role_path = tmp_path / "my_role"
        meta_dir = role_path / "meta"
        meta_dir.mkdir(parents=True)

        meta_file = meta_dir / "main.yml"
        meta_file.write_text("""
dependencies:
  - role: common
  - role: nginx
    version: "2.0"
""")

        result = await skill.execute("analyze_ansible", context, {"role_path": "my_role"})

        assert result.status == SkillStatus.SUCCESS
        deps = result.data["dependencies"]
        assert len(deps) >= 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestSkillIntegration:
    """Integration tests for skills."""

    @pytest.fixture
    def registry(self):
        """Create a populated registry."""
        registry = SkillRegistry()
        registry.register(TestingSkill())
        registry.register(DependencySkill())
        return registry

    @pytest.fixture
    def context(self, tmp_path):
        """Create a skill context."""
        return SkillContext(working_dir=tmp_path)

    def test_registry_has_all_skills(self, registry):
        """Test registry has expected skills."""
        skills = registry.list_skills()
        assert "testing" in skills
        assert "dependencies" in skills

    def test_get_all_actions(self, registry):
        """Test getting all actions from registry."""
        actions = registry.get_all_actions()
        assert "testing" in actions
        assert "run_tests" in actions["testing"]
        assert "dependencies" in actions
        assert "analyze_deps" in actions["dependencies"]

    @pytest.mark.asyncio
    async def test_execute_multiple_skills(self, registry, context):
        """Test executing multiple skills through registry."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="5 passed in 1.0s",
                stderr="",
            )

            result1 = await registry.execute("testing", "run_tests", context)
            assert result1.status in (SkillStatus.SUCCESS, SkillStatus.SKIPPED)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="[]",
            )

            result2 = await registry.execute("dependencies", "check_updates", context)
            assert result2.status == SkillStatus.SUCCESS
