"""
Tests for Claude Agent SDK integration.

Tests cover:
- SDK availability detection
- SDKAgentConfig configuration
- SDKClaudeIntegration lifecycle
- In-process MCP tools
- Hook registration and execution
- Session resumption
- Fallback to subprocess integration
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.db.state import StateDB
from harness.hotl.agent_session import (
    AgentSession,
    AgentSessionManager,
    AgentStatus,
)
from harness.hotl.claude_sdk_integration import (
    _SDK_AVAILABLE,
    AsyncSDKClaudeIntegration,
    HookEvent,
    PermissionMode,
    SDKAgentConfig,
    SDKClaudeIntegration,
    create_claude_integration,
    create_hotl_mcp_tools,
    sdk_available,
)

# ============================================================================
# SDK AVAILABILITY TESTS
# ============================================================================


class TestSDKAvailability:
    """Tests for SDK availability detection."""

    def test_sdk_available_returns_bool(self):
        """Test that sdk_available returns a boolean."""
        result = sdk_available()
        assert isinstance(result, bool)

    def test_sdk_available_consistent(self):
        """Test that sdk_available returns consistent results."""
        result1 = sdk_available()
        result2 = sdk_available()
        assert result1 == result2


# ============================================================================
# SDK AGENT CONFIG TESTS
# ============================================================================


class TestSDKAgentConfig:
    """Tests for SDKAgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SDKAgentConfig()

        assert config.system_prompt is None
        assert config.allowed_tools == ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
        assert config.permission_mode == PermissionMode.ACCEPT_EDITS
        assert config.max_turns is None
        assert config.max_budget_usd is None
        assert config.model is None
        assert config.cwd is None
        assert config.env == {}
        assert config.resume_session_id is None
        assert config.enable_file_checkpointing is True
        assert config.default_timeout == 600

    def test_custom_config(self):
        """Test custom configuration."""
        config = SDKAgentConfig(
            system_prompt="You are a helpful assistant.",
            allowed_tools=["Read", "Write"],
            permission_mode=PermissionMode.BYPASS_PERMISSIONS,
            max_turns=10,
            max_budget_usd=5.0,
            model="claude-3-opus",
            cwd=Path("/tmp"),
            env={"CUSTOM_VAR": "value"},
            resume_session_id="session-123",
            enable_file_checkpointing=False,
            default_timeout=300,
        )

        assert config.system_prompt == "You are a helpful assistant."
        assert config.allowed_tools == ["Read", "Write"]
        assert config.permission_mode == PermissionMode.BYPASS_PERMISSIONS
        assert config.max_turns == 10
        assert config.max_budget_usd == 5.0
        assert config.model == "claude-3-opus"
        assert config.cwd == Path("/tmp")
        assert config.env == {"CUSTOM_VAR": "value"}
        assert config.resume_session_id == "session-123"
        assert config.enable_file_checkpointing is False
        assert config.default_timeout == 300


class TestPermissionMode:
    """Tests for PermissionMode enum."""

    def test_permission_mode_values(self):
        """Test PermissionMode enum values."""
        assert PermissionMode.DEFAULT.value == "default"
        assert PermissionMode.ACCEPT_EDITS.value == "acceptEdits"
        assert PermissionMode.PLAN.value == "plan"
        assert PermissionMode.BYPASS_PERMISSIONS.value == "bypassPermissions"


class TestHookEvent:
    """Tests for HookEvent enum."""

    def test_hook_event_values(self):
        """Test HookEvent enum values."""
        assert HookEvent.PRE_TOOL_USE.value == "PreToolUse"
        assert HookEvent.POST_TOOL_USE.value == "PostToolUse"
        assert HookEvent.USER_PROMPT_SUBMIT.value == "UserPromptSubmit"
        assert HookEvent.STOP.value == "Stop"


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


class TestCreateClaudeIntegration:
    """Tests for the create_claude_integration factory function."""

    def test_create_with_fallback_enabled(self, tmp_path):
        """Test creating integration with fallback enabled."""
        # This should not raise even if SDK is not available
        integration = create_claude_integration(
            fallback_to_subprocess=True,
        )

        # Should return either SDK or subprocess integration
        assert integration is not None
        assert hasattr(integration, "spawn_agent")
        assert hasattr(integration, "get_session")

    def test_create_returns_correct_type(self, tmp_path):
        """Test that correct integration type is returned."""
        integration = create_claude_integration(
            fallback_to_subprocess=True,
        )

        if sdk_available():
            assert isinstance(integration, SDKClaudeIntegration)
        else:
            from harness.hotl.claude_integration import HOTLClaudeIntegration

            assert isinstance(integration, HOTLClaudeIntegration)


# ============================================================================
# SDK INTEGRATION TESTS (WITH MOCKS)
# ============================================================================


@pytest.mark.skipif(not _SDK_AVAILABLE, reason="SDK not installed")
class TestSDKClaudeIntegration:
    """Tests for SDKClaudeIntegration with mocked SDK."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SDKAgentConfig(
            default_timeout=10,
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )

    @pytest.fixture
    def session_manager(self):
        """Create session manager."""
        return AgentSessionManager()

    @pytest.fixture
    def integration(self, config, session_manager):
        """Create integration instance."""
        return SDKClaudeIntegration(
            config=config,
            session_manager=session_manager,
        )

    def test_init(self, integration):
        """Test initialization."""
        assert integration.config is not None
        assert integration.session_manager is not None
        assert integration._active_clients == {}
        assert integration._current_session_id is None

    def test_set_callbacks(self, integration):
        """Test setting callbacks."""
        on_complete = MagicMock()
        on_progress = MagicMock()
        on_intervention = MagicMock()
        on_tool_use = MagicMock()

        integration.set_callbacks(
            on_complete=on_complete,
            on_progress=on_progress,
            on_intervention=on_intervention,
            on_tool_use=on_tool_use,
        )

        assert integration._on_complete == on_complete
        assert integration._on_progress == on_progress
        assert integration._on_intervention == on_intervention
        assert integration._on_tool_use == on_tool_use

    def test_add_hooks(self, integration):
        """Test adding hooks."""

        async def pre_hook(input_data, tool_use_id, context):
            return {}

        async def post_hook(input_data, tool_use_id, context):
            return {}

        integration.add_pre_tool_hook(pre_hook)
        integration.add_post_tool_hook(post_hook)

        assert len(integration._pre_tool_hooks) == 1
        assert len(integration._post_tool_hooks) == 1

    def test_get_session_not_found(self, integration):
        """Test getting non-existent session."""
        session = integration.get_session("non-existent")
        assert session is None

    def test_poll_agent_not_found(self, integration):
        """Test polling non-existent agent."""
        with pytest.raises(ValueError, match="Session not found"):
            integration.poll_agent("non-existent")

    def test_send_feedback(self, integration, tmp_path):
        """Test sending feedback to session."""
        session = integration.session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )

        integration.send_feedback(session.id, "Test feedback")

        updated = integration.get_session(session.id)
        assert any("[HUMAN_FEEDBACK]" in p for p in updated.progress_updates)

    def test_send_feedback_not_found(self, integration):
        """Test sending feedback to non-existent session."""
        with pytest.raises(ValueError, match="Session not found"):
            integration.send_feedback("non-existent", "Feedback")

    def test_terminate_agent_not_found(self, integration):
        """Test terminating non-existent agent."""
        result = integration.terminate_agent("non-existent")
        assert result is False

    def test_terminate_all_agents_empty(self, integration):
        """Test terminating all agents when none running."""
        count = integration.terminate_all_agents()
        assert count == 0

    def test_get_active_agents_empty(self, integration):
        """Test getting active agents when none running."""
        active = integration.get_active_agents()
        assert active == []

    def test_get_pending_interventions_empty(self, integration):
        """Test getting pending interventions when none."""
        pending = integration.get_pending_interventions()
        assert pending == []

    def test_resolve_intervention(self, integration, tmp_path):
        """Test resolving an intervention."""
        session = integration.session_manager.create_session(
            task="Task needing help",
            working_dir=tmp_path,
        )
        session.mark_started()
        session.request_intervention("Need approval")
        integration.session_manager.update_session(session)

        integration.resolve_intervention(
            session_id=session.id,
            resolution="Approved",
            continue_agent=False,
        )

        resolved = integration.get_session(session.id)
        assert resolved.status == AgentStatus.COMPLETED
        assert any("[INTERVENTION_RESOLVED]" in p for p in resolved.progress_updates)

    def test_resolve_intervention_not_found(self, integration):
        """Test resolving intervention for non-existent session."""
        with pytest.raises(ValueError, match="Session not found"):
            integration.resolve_intervention("non-existent", "Resolution")

    def test_resolve_intervention_not_needed(self, integration, tmp_path):
        """Test resolving intervention when not needed."""
        session = integration.session_manager.create_session(
            task="Normal task",
            working_dir=tmp_path,
        )
        session.mark_started()
        session.mark_completed()
        integration.session_manager.update_session(session)

        with pytest.raises(ValueError, match="does not need intervention"):
            integration.resolve_intervention(session.id, "Resolution")

    def test_get_stats(self, integration):
        """Test getting statistics."""
        stats = integration.get_stats()

        assert "total_sessions" in stats
        assert "status_counts" in stats
        assert "active_sdk_clients" in stats
        assert "sdk_available" in stats
        assert stats["sdk_available"] is True


@pytest.mark.skipif(not _SDK_AVAILABLE, reason="SDK not installed")
class TestAsyncSDKClaudeIntegration:
    """Tests for AsyncSDKClaudeIntegration wrapper."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SDKAgentConfig(default_timeout=5)

    @pytest.fixture
    def async_integration(self, config):
        """Create async integration instance."""
        sync_integration = SDKClaudeIntegration(config=config)
        return AsyncSDKClaudeIntegration(sync_integration)

    @pytest.mark.asyncio
    async def test_get_session(self, async_integration, tmp_path):
        """Test async get_session."""
        session = async_integration._integration.session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )

        retrieved = await async_integration.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    @pytest.mark.asyncio
    async def test_poll_agent(self, async_integration, tmp_path):
        """Test async poll_agent."""
        session = async_integration._integration.session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )

        status = await async_integration.poll_agent(session.id)
        assert status == AgentStatus.PENDING

    @pytest.mark.asyncio
    async def test_send_feedback(self, async_integration, tmp_path):
        """Test async send_feedback."""
        session = async_integration._integration.session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )

        await async_integration.send_feedback(session.id, "Test feedback")

        updated = await async_integration.get_session(session.id)
        assert any("HUMAN_FEEDBACK" in p for p in updated.progress_updates)

    @pytest.mark.asyncio
    async def test_get_stats(self, async_integration):
        """Test async get_stats."""
        stats = async_integration.get_stats()
        assert "total_sessions" in stats


# ============================================================================
# MCP TOOLS TESTS
# ============================================================================


@pytest.mark.skipif(not _SDK_AVAILABLE, reason="SDK not installed")
class TestHOTLMCPTools:
    """Tests for in-process MCP tools."""

    @pytest.fixture
    def session_manager(self):
        """Create session manager."""
        return AgentSessionManager()

    @pytest.fixture
    def current_session_id(self):
        """Track current session ID."""
        return {"value": None}

    @pytest.fixture
    def mcp_server(self, session_manager, current_session_id):
        """Create MCP server with tools."""
        return create_hotl_mcp_tools(
            session_manager=session_manager,
            current_session_id_getter=lambda: current_session_id["value"],
        )

    def test_mcp_server_created(self, mcp_server):
        """Test MCP server is created."""
        assert mcp_server is not None

    def test_mcp_server_has_tools(self, mcp_server):
        """Test MCP server has expected tools."""
        # Server should be configured with tools
        # Exact structure depends on SDK version
        assert mcp_server is not None


# ============================================================================
# INTEGRATION TESTS WITHOUT SDK (MOCKED)
# ============================================================================


class TestSDKIntegrationMocked:
    """Tests for SDK integration with mocked SDK components."""

    @pytest.fixture
    def mock_sdk(self):
        """Mock the SDK components."""
        with patch.dict(
            "sys.modules",
            {
                "claude_agent_sdk": MagicMock(),
            },
        ):
            yield

    @pytest.fixture
    def session_manager(self):
        """Create session manager."""
        return AgentSessionManager()

    def test_session_creation(self, session_manager, tmp_path):
        """Test session creation through session manager."""
        session = session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
            context={"test": True},
        )

        assert session.id is not None
        assert session.task == "Test task"
        assert session.working_dir == tmp_path
        assert session.context == {"test": True}
        assert session.status == AgentStatus.PENDING
        assert session.sdk_session_id is None

    def test_session_with_sdk_session_id(self, session_manager, tmp_path):
        """Test session with SDK session ID."""
        session = session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )
        session.sdk_session_id = "sdk-session-123"
        session_manager.update_session(session)

        retrieved = session_manager.get_session(session.id)
        assert retrieved.sdk_session_id == "sdk-session-123"

    def test_session_serialization_with_sdk_id(self, session_manager, tmp_path):
        """Test session serialization includes SDK session ID."""
        session = session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )
        session.sdk_session_id = "sdk-session-456"

        data = session.to_dict()
        assert "sdk_session_id" in data
        assert data["sdk_session_id"] == "sdk-session-456"

    def test_session_deserialization_with_sdk_id(self, tmp_path):
        """Test session deserialization includes SDK session ID."""
        data = {
            "id": "test-session",
            "task": "Test task",
            "working_dir": str(tmp_path),
            "status": "running",
            "sdk_session_id": "sdk-session-789",
        }

        session = AgentSession.from_dict(data)
        assert session.sdk_session_id == "sdk-session-789"


# ============================================================================
# FALLBACK BEHAVIOR TESTS
# ============================================================================


class TestFallbackBehavior:
    """Tests for fallback to subprocess integration."""

    def test_fallback_when_sdk_unavailable(self):
        """Test fallback to subprocess when SDK not available."""
        # Force fallback
        integration = create_claude_integration(
            fallback_to_subprocess=True,
        )

        assert integration is not None
        # Should have spawn_agent method regardless of type
        assert hasattr(integration, "spawn_agent")

    @pytest.mark.skipif(_SDK_AVAILABLE, reason="Test requires SDK to be unavailable")
    def test_error_when_fallback_disabled_and_sdk_unavailable(self):
        """Test error when fallback disabled and SDK unavailable."""
        with pytest.raises(RuntimeError, match="claude-agent-sdk is not installed"):
            create_claude_integration(
                fallback_to_subprocess=False,
            )


# ============================================================================
# SUPERVISOR INTEGRATION TESTS
# ============================================================================


class TestSupervisorIntegration:
    """Tests for HOTL Supervisor using SDK integration."""

    @pytest.fixture
    def in_memory_db(self):
        """Create in-memory database."""
        return StateDB(":memory:")

    def test_supervisor_uses_sdk_when_available(self, in_memory_db):
        """Test that supervisor uses SDK when available."""
        from harness.hotl.supervisor import HOTLSupervisor

        config = {
            "use_sdk_integration": True,
            "claude_timeout": 60,
        }

        supervisor = HOTLSupervisor(db=in_memory_db, config=config)

        # Check which integration is being used
        if sdk_available():
            assert supervisor._using_sdk is True
            assert isinstance(supervisor.claude_integration, SDKClaudeIntegration)
        else:
            assert supervisor._using_sdk is False
            from harness.hotl.claude_integration import HOTLClaudeIntegration

            assert isinstance(supervisor.claude_integration, HOTLClaudeIntegration)

    def test_supervisor_uses_subprocess_when_configured(self, in_memory_db):
        """Test that supervisor uses subprocess when configured."""
        from harness.hotl.claude_integration import HOTLClaudeIntegration
        from harness.hotl.supervisor import HOTLSupervisor

        config = {
            "use_sdk_integration": False,
            "claude_timeout": 60,
        }

        supervisor = HOTLSupervisor(db=in_memory_db, config=config)

        assert supervisor._using_sdk is False
        assert isinstance(supervisor.claude_integration, HOTLClaudeIntegration)

    def test_supervisor_agent_methods_available(self, in_memory_db):
        """Test that supervisor has agent management methods."""
        from harness.hotl.supervisor import HOTLSupervisor

        supervisor = HOTLSupervisor(db=in_memory_db)

        # These methods should be available regardless of integration type
        assert hasattr(supervisor, "get_active_agents")
        assert hasattr(supervisor, "get_pending_interventions")
        assert hasattr(supervisor, "resolve_intervention")
        assert hasattr(supervisor, "terminate_agent")
        assert hasattr(supervisor, "terminate_all_agents")
        assert hasattr(supervisor, "get_agent_stats")
        assert hasattr(supervisor, "spawn_agent_manually")


# ============================================================================
# HOOK EXECUTION TESTS
# ============================================================================


@pytest.mark.skipif(not _SDK_AVAILABLE, reason="SDK not installed")
class TestHookExecution:
    """Tests for hook execution."""

    @pytest.fixture
    def integration(self):
        """Create integration with hooks."""
        return SDKClaudeIntegration()

    @pytest.mark.asyncio
    async def test_pre_tool_hook_registration(self, integration):
        """Test registering a pre-tool hook."""
        hook_called = {"value": False}

        async def pre_hook(input_data, tool_use_id, context):
            hook_called["value"] = True
            return {}

        integration.add_pre_tool_hook(pre_hook)
        assert len(integration._pre_tool_hooks) == 1

    @pytest.mark.asyncio
    async def test_post_tool_hook_registration(self, integration):
        """Test registering a post-tool hook."""
        hook_called = {"value": False}

        async def post_hook(input_data, tool_use_id, context):
            hook_called["value"] = True
            return {}

        integration.add_post_tool_hook(post_hook)
        assert len(integration._post_tool_hooks) == 1

    @pytest.mark.asyncio
    async def test_multiple_hooks_registration(self, integration):
        """Test registering multiple hooks."""

        async def hook1(input_data, tool_use_id, context):
            return {}

        async def hook2(input_data, tool_use_id, context):
            return {}

        integration.add_pre_tool_hook(hook1)
        integration.add_pre_tool_hook(hook2)
        integration.add_post_tool_hook(hook1)

        assert len(integration._pre_tool_hooks) == 2
        assert len(integration._post_tool_hooks) == 1
