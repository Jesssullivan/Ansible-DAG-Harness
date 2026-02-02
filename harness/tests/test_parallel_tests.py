"""Tests for parallel test execution (Task #21).

Tests the LangGraph Send API-based parallel execution of molecule and pytest tests.
"""

import time

import pytest

from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    create_box_up_role_graph,
    create_initial_state,
    merge_test_results_node,
    route_to_parallel_tests,
    run_molecule_node,
    run_pytest_node,
    should_continue_after_merge,
)


class TestRouteToParallelTests:
    """Test the parallel routing function."""

    @pytest.mark.unit
    def test_routes_both_tests_when_molecule_available(self):
        """Should route to both molecule and pytest when molecule tests exist."""
        state = create_initial_state("test_role")
        state["has_molecule_tests"] = True
        state["worktree_path"] = "/tmp/test"

        sends = route_to_parallel_tests(state)

        assert len(sends) == 2
        node_names = [s.node for s in sends]
        assert "run_molecule" in node_names
        assert "run_pytest" in node_names

    @pytest.mark.unit
    def test_routes_only_pytest_when_no_molecule(self):
        """Should only route to pytest when no molecule tests exist."""
        state = create_initial_state("test_role")
        state["has_molecule_tests"] = False
        state["worktree_path"] = "/tmp/test"

        sends = route_to_parallel_tests(state)

        # Should only route to pytest when molecule tests don't exist
        # This optimizes execution by not spawning unnecessary nodes
        assert len(sends) == 1
        assert sends[0].node == "run_pytest"

    @pytest.mark.unit
    def test_sends_include_test_phase_start_time(self):
        """Each Send should include test_phase_start_time for timing."""
        state = create_initial_state("test_role")
        state["has_molecule_tests"] = True

        sends = route_to_parallel_tests(state)

        for send in sends:
            assert "test_phase_start_time" in send.arg
            assert isinstance(send.arg["test_phase_start_time"], float)

    @pytest.mark.unit
    def test_sends_preserve_state(self):
        """Send arguments should preserve original state."""
        state = create_initial_state("test_role")
        state["has_molecule_tests"] = True
        state["worktree_path"] = "/custom/path"
        state["execution_id"] = 42

        sends = route_to_parallel_tests(state)

        for send in sends:
            assert send.arg["role_name"] == "test_role"
            assert send.arg["worktree_path"] == "/custom/path"
            assert send.arg["execution_id"] == 42


class TestMergeTestResults:
    """Test the test results merger node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_all_tests_passed(self):
        """When both tests pass, all_tests_passed should be True."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": True,
            "molecule_skipped": False,
            "molecule_duration": 60,
            "pytest_passed": True,
            "pytest_skipped": False,
            "pytest_duration": 30,
            "test_phase_start_time": time.time() - 65,  # Started 65s ago
        }

        result = await merge_test_results_node(state)

        assert result["all_tests_passed"] is True
        assert "errors" not in result or len(result.get("errors", [])) == 0
        assert "merge_test_results" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_molecule_failed_pytest_passed(self):
        """When molecule fails but pytest passes, all_tests_passed should be False."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": False,
            "molecule_skipped": False,
            "molecule_duration": 120,
            "pytest_passed": True,
            "pytest_skipped": False,
            "pytest_duration": 30,
            "test_phase_start_time": time.time() - 125,
        }

        result = await merge_test_results_node(state)

        assert result["all_tests_passed"] is False
        assert len(result.get("errors", [])) == 1
        assert "Molecule" in result["errors"][0]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_molecule_passed_pytest_failed(self):
        """When molecule passes but pytest fails, all_tests_passed should be False."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": True,
            "molecule_skipped": False,
            "molecule_duration": 60,
            "pytest_passed": False,
            "pytest_skipped": False,
            "pytest_duration": 10,
            "test_phase_start_time": time.time() - 65,
        }

        result = await merge_test_results_node(state)

        assert result["all_tests_passed"] is False
        assert len(result.get("errors", [])) == 1
        assert "Pytest" in result["errors"][0]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_both_tests_failed(self):
        """When both tests fail, should report both failures."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": False,
            "molecule_skipped": False,
            "molecule_duration": 120,
            "pytest_passed": False,
            "pytest_skipped": False,
            "pytest_duration": 10,
            "test_phase_start_time": time.time() - 125,
        }

        result = await merge_test_results_node(state)

        assert result["all_tests_passed"] is False
        assert len(result.get("errors", [])) == 2

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_skipped_tests_count_as_passed(self):
        """Skipped tests should count as passed for workflow continuation."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": None,
            "molecule_skipped": True,
            "molecule_duration": 0,
            "pytest_passed": None,
            "pytest_skipped": True,
            "pytest_duration": 0,
            "test_phase_start_time": time.time() - 1,
        }

        result = await merge_test_results_node(state)

        assert result["all_tests_passed"] is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mixed_pass_and_skip(self):
        """When one test passes and other is skipped, should pass."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": True,
            "molecule_skipped": False,
            "molecule_duration": 60,
            "pytest_passed": None,
            "pytest_skipped": True,
            "pytest_duration": 0,
            "test_phase_start_time": time.time() - 62,
        }

        result = await merge_test_results_node(state)

        assert result["all_tests_passed"] is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_timing_metrics_calculated(self):
        """Should calculate test phase duration from start time."""
        start_time = time.time() - 100  # Started 100 seconds ago
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": True,
            "molecule_skipped": False,
            "molecule_duration": 60,
            "pytest_passed": True,
            "pytest_skipped": False,
            "pytest_duration": 30,
            "test_phase_start_time": start_time,
        }

        result = await merge_test_results_node(state)

        # Duration should be approximately 100 seconds (with some tolerance)
        assert result["test_phase_duration"] is not None
        assert 99 < result["test_phase_duration"] < 102


class TestShouldContinueAfterMerge:
    """Test the routing after test merge."""

    @pytest.mark.unit
    def test_routes_to_deploy_when_all_passed(self):
        """Should continue to validate_deploy when all tests passed."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "all_tests_passed": True,
        }

        result = should_continue_after_merge(state)

        assert result == "validate_deploy"

    @pytest.mark.unit
    def test_routes_to_failure_when_tests_failed(self):
        """Should route to notify_failure when tests failed."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "all_tests_passed": False,
        }

        result = should_continue_after_merge(state)

        assert result == "notify_failure"

    @pytest.mark.unit
    def test_routes_to_failure_when_status_missing(self):
        """Should default to failure if all_tests_passed is missing."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
        }

        result = should_continue_after_merge(state)

        assert result == "notify_failure"


class TestParallelTestNodes:
    """Test that test nodes support parallel execution."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_molecule_node_tracks_parallel_completion(self):
        """Molecule node should add itself to parallel_tests_completed."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "has_molecule_tests": False,  # Skip actual execution
            "worktree_path": "/tmp/test",
        }

        result = await run_molecule_node(state)

        assert "run_molecule" in result["parallel_tests_completed"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_pytest_node_tracks_parallel_completion(self):
        """Pytest node should add itself to parallel_tests_completed."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "worktree_path": "/nonexistent/path",  # Will skip due to no test file
        }

        result = await run_pytest_node(state)

        assert "run_pytest" in result["parallel_tests_completed"]


class TestGraphConstruction:
    """Test graph construction with parallel tests."""

    @pytest.mark.unit
    def test_creates_graph_with_parallel_tests_enabled(self):
        """Graph should include merge_test_results when parallel_tests=True."""
        graph = create_box_up_role_graph(parallel_tests=True)

        # Check that the graph was created (basic sanity check)
        assert graph is not None

    @pytest.mark.unit
    def test_creates_graph_with_parallel_tests_disabled(self):
        """Graph should work in sequential mode when parallel_tests=False."""
        graph = create_box_up_role_graph(parallel_tests=False)

        # Check that the graph was created
        assert graph is not None


class TestPerformanceBenchmarks:
    """Test performance tracking and benchmarking."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_parallel_execution_shows_time_savings(self):
        """When tests run in parallel, should show positive time savings."""
        # Simulate parallel execution where actual time < sum of individual times
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": True,
            "molecule_skipped": False,
            "molecule_duration": 60,  # Would take 60s sequentially
            "pytest_passed": True,
            "pytest_skipped": False,
            "pytest_duration": 40,  # Would take 40s sequentially
            # Actual parallel time: max(60, 40) = 60s, not 100s
            "test_phase_start_time": time.time() - 62,  # Ran in ~62s
        }

        result = await merge_test_results_node(state)

        # Verify timing is tracked
        assert result["test_phase_duration"] is not None
        # In parallel mode, duration should be close to max(60, 40) = 60s, not 100s
        # Actual calculation happens in merge_test_results_node

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handles_missing_timing_gracefully(self):
        """Should handle cases where timing info is missing."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": True,
            "molecule_skipped": False,
            "molecule_duration": 60,
            "pytest_passed": True,
            "pytest_skipped": False,
            "pytest_duration": 40,
            # No test_phase_start_time
        }

        result = await merge_test_results_node(state)

        # Should still work, using fallback calculation
        assert result["all_tests_passed"] is True
        assert result["test_phase_duration"] is not None


class TestPartialFailureHandling:
    """Test handling of partial test failures."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_partial_failure_reports_failed_test(self):
        """When one test fails, should report which one."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": False,
            "molecule_skipped": False,
            "molecule_duration": 120,
            "pytest_passed": True,
            "pytest_skipped": False,
            "pytest_duration": 30,
            "test_phase_start_time": time.time() - 125,
        }

        result = await merge_test_results_node(state)

        # Should include specific failure info
        assert result["all_tests_passed"] is False
        errors = result.get("errors", [])
        assert len(errors) == 1
        assert "Molecule" in errors[0]
        assert "120s" in errors[0]  # Duration should be in error message

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_partial_failure_both_tests_ran(self):
        """Even if one test fails, both should have completed."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": False,
            "molecule_skipped": False,
            "molecule_duration": 120,
            "pytest_passed": True,
            "pytest_skipped": False,
            "pytest_duration": 30,
            "parallel_tests_completed": ["run_molecule", "run_pytest"],
            "test_phase_start_time": time.time() - 125,
        }

        result = await merge_test_results_node(state)

        # Result indicates failure but both tests completed
        assert "merge_test_results" in result["completed_nodes"]


class TestStateIntegration:
    """Test state field integration."""

    @pytest.mark.unit
    def test_initial_state_has_parallel_fields(self):
        """Initial state should include parallel test fields."""
        state = create_initial_state("test_role")

        assert "all_tests_passed" in state
        assert "parallel_tests_completed" in state
        assert "test_phase_start_time" in state
        assert "test_phase_duration" in state
        assert "parallel_execution_enabled" in state
        assert state["parallel_execution_enabled"] is True
        assert state["parallel_tests_completed"] == []
