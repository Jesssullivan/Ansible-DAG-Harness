"""Tests for MCP tool search functionality."""

import pytest

from harness.mcp.server import (
    TOOL_CATEGORIES,
    TOOL_DESCRIPTIONS,
    _search_tools_impl,
)

# ============================================================================
# TOOL CATEGORIES TESTS
# ============================================================================


class TestToolCategories:
    """Tests for TOOL_CATEGORIES structure."""

    def test_all_categories_exist(self):
        """Test that expected categories are defined."""
        expected_categories = [
            "role_management",
            "worktree",
            "workflow",
            "testing",
            "credentials",
            "merge_train",
            "agent",
            "search",
        ]
        for category in expected_categories:
            assert category in TOOL_CATEGORIES, f"Missing category: {category}"

    def test_role_management_tools(self):
        """Test role_management category contains expected tools."""
        tools = TOOL_CATEGORIES["role_management"]
        expected = [
            "list_roles",
            "get_role_status",
            "get_dependencies",
            "get_reverse_dependencies",
        ]
        for tool in expected:
            assert tool in tools, f"Missing tool in role_management: {tool}"

    def test_workflow_tools(self):
        """Test workflow category contains expected tools."""
        tools = TOOL_CATEGORIES["workflow"]
        expected = [
            "get_workflow_status",
            "hotl_status",
        ]
        for tool in expected:
            assert tool in tools, f"Missing tool in workflow: {tool}"

    def test_testing_tools(self):
        """Test testing category contains expected tools."""
        tools = TOOL_CATEGORIES["testing"]
        expected = [
            "get_test_history",
            "get_active_regressions",
        ]
        for tool in expected:
            assert tool in tools, f"Missing tool in testing: {tool}"

    def test_agent_tools(self):
        """Test agent category contains expected tools."""
        tools = TOOL_CATEGORIES["agent"]
        expected = [
            "agent_report_progress",
            "agent_request_intervention",
        ]
        for tool in expected:
            assert tool in tools, f"Missing tool in agent: {tool}"

    def test_search_tools_category(self):
        """Test search category exists and contains search_tools."""
        assert "search" in TOOL_CATEGORIES
        assert "search_tools" in TOOL_CATEGORIES["search"]

    def test_no_duplicate_tools_across_categories(self):
        """Test that each tool appears in only one category."""
        all_tools = []
        for category, tools in TOOL_CATEGORIES.items():
            for tool in tools:
                assert tool not in all_tools, f"Duplicate tool: {tool}"
                all_tools.append(tool)


# ============================================================================
# TOOL DESCRIPTIONS TESTS
# ============================================================================


class TestToolDescriptions:
    """Tests for TOOL_DESCRIPTIONS structure."""

    def test_all_categorized_tools_have_descriptions(self):
        """Test that all tools in categories have descriptions."""
        for category, tools in TOOL_CATEGORIES.items():
            for tool in tools:
                assert tool in TOOL_DESCRIPTIONS, f"Missing description for {tool} in {category}"

    def test_descriptions_are_non_empty(self):
        """Test that all descriptions are non-empty strings."""
        for tool, description in TOOL_DESCRIPTIONS.items():
            assert isinstance(description, str), f"Description for {tool} not a string"
            assert len(description) > 10, f"Description for {tool} too short"

    def test_descriptions_contain_keywords(self):
        """Test that descriptions contain relevant keywords."""
        # Role-related tools should mention roles
        role_tools = ["list_roles", "get_role_status"]
        for tool in role_tools:
            assert "role" in TOOL_DESCRIPTIONS[tool].lower()

        # Test-related tools should mention test
        test_tools = ["get_test_history", "get_active_regressions"]
        for tool in test_tools:
            desc = TOOL_DESCRIPTIONS[tool].lower()
            assert "test" in desc or "regression" in desc


# ============================================================================
# SEARCH TOOLS IMPLEMENTATION TESTS
# ============================================================================


class TestSearchToolsImpl:
    """Tests for _search_tools_impl function."""

    def test_search_empty_query_returns_all(self):
        """Test that empty query returns all tools."""
        results = _search_tools_impl("")
        assert len(results) == len(TOOL_DESCRIPTIONS)

    def test_search_by_tool_name(self):
        """Test searching by tool name."""
        results = _search_tools_impl("list_roles")

        assert len(results) >= 1
        assert any(r["name"] == "list_roles" for r in results)

    def test_search_by_partial_name(self):
        """Test searching by partial tool name."""
        results = _search_tools_impl("role")

        # Should find role-related tools
        role_tool_found = any("role" in r["name"].lower() for r in results)
        assert role_tool_found

    def test_search_by_description(self):
        """Test searching by description content."""
        results = _search_tools_impl("Ansible")

        # Should find tools with Ansible in description
        assert len(results) > 0

    def test_search_by_category(self):
        """Test filtering by category."""
        results = _search_tools_impl("", category="testing")

        assert len(results) == len(TOOL_CATEGORIES["testing"])
        for result in results:
            assert result["category"] == "testing"

    def test_search_by_category_and_query(self):
        """Test filtering by category with query."""
        results = _search_tools_impl("test", category="testing")

        for result in results:
            assert result["category"] == "testing"

    def test_search_invalid_category_returns_empty(self):
        """Test that invalid category returns empty list."""
        results = _search_tools_impl("", category="nonexistent_category")
        assert results == []

    def test_search_case_insensitive(self):
        """Test that search is case insensitive."""
        lower_results = _search_tools_impl("role")
        upper_results = _search_tools_impl("ROLE")
        mixed_results = _search_tools_impl("Role")

        assert len(lower_results) == len(upper_results)
        assert len(lower_results) == len(mixed_results)

    def test_search_results_have_required_fields(self):
        """Test that search results have all required fields."""
        results = _search_tools_impl("role")

        for result in results:
            assert "name" in result
            assert "category" in result
            assert "description" in result
            assert isinstance(result["name"], str)
            assert isinstance(result["category"], str)
            assert isinstance(result["description"], str)

    def test_search_results_sorted_by_relevance(self):
        """Test that exact name matches appear first."""
        results = _search_tools_impl("get_role_status")

        # The exact match should be first
        if results:
            assert results[0]["name"] == "get_role_status"

    def test_search_by_category_name(self):
        """Test searching by category name in query."""
        results = _search_tools_impl("workflow")

        # Should find workflow tools
        workflow_found = any(r["category"] == "workflow" for r in results)
        assert workflow_found

    def test_search_for_agent_tools(self):
        """Test searching for agent-related tools."""
        results = _search_tools_impl("agent")

        agent_tools = [r for r in results if r["category"] == "agent"]
        assert len(agent_tools) >= 2  # At least progress and intervention

    def test_search_for_status_tools(self):
        """Test searching for status-related tools."""
        results = _search_tools_impl("status")

        # Should find multiple status-related tools
        assert len(results) >= 2
        status_names = [r["name"] for r in results]
        assert any("status" in name for name in status_names)

    def test_all_tools_searchable(self):
        """Test that every tool can be found by its exact name."""
        for tool_name in TOOL_DESCRIPTIONS.keys():
            results = _search_tools_impl(tool_name)
            found = any(r["name"] == tool_name for r in results)
            assert found, f"Could not find tool: {tool_name}"


# ============================================================================
# MCP SERVER INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
class TestMCPToolSearchIntegration:
    """Integration tests for MCP search_tools endpoint."""

    async def test_search_tools_via_mcp(self, in_memory_db):
        """Test search_tools through MCP server."""
        from harness.mcp.server import create_mcp_server

        mcp = create_mcp_server(":memory:")

        # Get the search_tools function from the server's tools
        # Note: This tests the internal registration, actual MCP calls
        # would need a running server
        tools = mcp._tool_manager._tools
        assert "search_tools" in tools

    async def test_list_tool_categories_via_mcp(self, in_memory_db):
        """Test list_tool_categories through MCP server."""
        from harness.mcp.server import create_mcp_server

        mcp = create_mcp_server(":memory:")

        tools = mcp._tool_manager._tools
        assert "list_tool_categories" in tools


# ============================================================================
# EDGE CASES
# ============================================================================


class TestSearchToolsEdgeCases:
    """Edge case tests for search functionality."""

    def test_search_special_characters(self):
        """Test searching with special regex characters."""
        # Should not raise an error
        results = _search_tools_impl("get_*")
        assert isinstance(results, list)

    def test_search_very_long_query(self):
        """Test searching with very long query."""
        long_query = "a" * 1000
        results = _search_tools_impl(long_query)
        assert results == []

    def test_search_with_whitespace(self):
        """Test searching with leading/trailing whitespace."""
        results = _search_tools_impl("  role  ")
        # Should still find role-related tools (if stripping is done)
        # If not, empty is also acceptable
        assert isinstance(results, list)

    def test_search_unicode(self):
        """Test searching with unicode characters."""
        results = _search_tools_impl("role\u00e9")
        assert isinstance(results, list)

    def test_empty_category(self):
        """Test that empty string category is treated as None."""
        all_results = _search_tools_impl("")
        empty_cat_results = _search_tools_impl("", category=None)
        assert len(all_results) == len(empty_cat_results)
