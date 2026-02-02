"""Tests for A2A (Agent-to-Agent) protocol implementation."""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.a2a import (
    # Protocol
    A2ACapability,
    # Client
    A2AClient,
    A2AClientError,
    A2AErrorCode,
    A2AMessage,
    A2AMessageType,
    # Server
    A2AServer,
    A2AServerError,
    A2AStatus,
    # Agent Card
    AgentCard,
    AgentEndpoint,
    AgentNotFoundError,
    CapabilityNotFoundError,
    ErrorMessage,
    HandoffRequest,
    HandoffResponse,
    HealthCheck,
    HealthResponse,
    InvocationError,
    InvokeRequest,
    InvokeResponse,
    create_harness_agent_card,
)

# ============================================================================
# PROTOCOL TESTS
# ============================================================================


class TestA2AProtocol:
    """Tests for A2A protocol message types."""

    def test_message_type_enum(self):
        """Test A2AMessageType enum values."""
        assert A2AMessageType.DISCOVER_REQUEST == "discover_request"
        assert A2AMessageType.INVOKE_REQUEST == "invoke_request"
        assert A2AMessageType.HANDOFF_REQUEST == "handoff_request"
        assert A2AMessageType.HEALTH_CHECK == "health_check"
        assert A2AMessageType.ERROR == "error"

    def test_capability_enum(self):
        """Test A2ACapability enum values."""
        assert A2ACapability.INVOKE == "invoke"
        assert A2ACapability.HANDOFF == "handoff"
        assert A2ACapability.CODE_ANALYSIS == "code_analysis"
        assert A2ACapability.ANSIBLE_ROLES == "ansible_roles"
        assert A2ACapability.MCP_CLIENT == "mcp_client"

    def test_status_enum(self):
        """Test A2AStatus enum values."""
        assert A2AStatus.SUCCESS == "success"
        assert A2AStatus.FAILURE == "failure"
        assert A2AStatus.PENDING == "pending"
        assert A2AStatus.TIMEOUT == "timeout"

    def test_base_message_creation(self):
        """Test creating a base A2A message."""
        msg = A2AMessage(
            message_type=A2AMessageType.DISCOVER_REQUEST,
            message_id="test-123",
            sender="agent-a",
        )
        assert msg.message_type == A2AMessageType.DISCOVER_REQUEST
        assert msg.message_id == "test-123"
        assert msg.sender == "agent-a"
        assert msg.timestamp is not None

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = A2AMessage(
            message_type=A2AMessageType.DISCOVER_REQUEST,
            message_id="test-123",
            sender="agent-a",
            recipient="agent-b",
        )
        data = msg.to_dict()
        assert data["message_type"] == "discover_request"
        assert data["message_id"] == "test-123"
        assert data["sender"] == "agent-a"
        assert data["recipient"] == "agent-b"

    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "message_type": "discover_request",
            "message_id": "test-456",
            "sender": "agent-x",
        }
        msg = A2AMessage.from_dict(data)
        assert msg.message_type == A2AMessageType.DISCOVER_REQUEST
        assert msg.message_id == "test-456"
        assert msg.sender == "agent-x"


class TestInvokeMessages:
    """Tests for invoke request/response messages."""

    def test_invoke_request_creation(self):
        """Test creating an invoke request."""
        req = InvokeRequest(
            message_id="inv-123",
            capability="code_analysis",
            task="Analyze the codebase",
            context={"role": "common"},
            timeout=60,
        )
        assert req.message_type == A2AMessageType.INVOKE_REQUEST
        assert req.capability == "code_analysis"
        assert req.task == "Analyze the codebase"
        assert req.context == {"role": "common"}
        assert req.timeout == 60

    def test_invoke_request_to_dict(self):
        """Test converting invoke request to dictionary."""
        req = InvokeRequest(
            message_id="inv-123",
            capability="testing",
            task="Run tests",
        )
        data = req.to_dict()
        assert data["message_type"] == "invoke_request"
        assert data["capability"] == "testing"
        assert data["task"] == "Run tests"

    def test_invoke_response_success(self):
        """Test creating a successful invoke response."""
        resp = InvokeResponse(
            message_id="resp-123",
            correlation_id="inv-123",
            status=A2AStatus.SUCCESS,
            result={"analysis": "complete"},
            execution_time_ms=150,
        )
        assert resp.status == A2AStatus.SUCCESS
        assert resp.result == {"analysis": "complete"}
        assert resp.error is None

    def test_invoke_response_failure(self):
        """Test creating a failed invoke response."""
        resp = InvokeResponse(
            message_id="resp-456",
            correlation_id="inv-456",
            status=A2AStatus.FAILURE,
            error="Analysis failed",
        )
        assert resp.status == A2AStatus.FAILURE
        assert resp.error == "Analysis failed"


class TestHandoffMessages:
    """Tests for handoff request/response messages."""

    def test_handoff_request_creation(self):
        """Test creating a handoff request."""
        req = HandoffRequest(
            message_id="ho-123",
            task="Continue fixing tests",
            context={"progress": 50},
            files_changed=["test_a.py", "test_b.py"],
            reason="Session limit reached",
            continuation_prompt="Please continue from where I left off",
        )
        assert req.message_type == A2AMessageType.HANDOFF_REQUEST
        assert req.task == "Continue fixing tests"
        assert len(req.files_changed) == 2
        assert req.reason == "Session limit reached"

    def test_handoff_response_accepted(self):
        """Test creating an accepted handoff response."""
        resp = HandoffResponse(
            message_id="ho-resp-123",
            correlation_id="ho-123",
            accepted=True,
            session_id="session-abc",
        )
        assert resp.message_type == A2AMessageType.HANDOFF_ACCEPT
        assert resp.accepted is True
        assert resp.session_id == "session-abc"

    def test_handoff_response_rejected(self):
        """Test creating a rejected handoff response."""
        resp = HandoffResponse(
            message_id="ho-resp-456",
            correlation_id="ho-456",
            accepted=False,
            reason="Agent is busy",
        )
        assert resp.message_type == A2AMessageType.HANDOFF_REJECT
        assert resp.accepted is False
        assert resp.reason == "Agent is busy"


class TestHealthMessages:
    """Tests for health check messages."""

    def test_health_check_creation(self):
        """Test creating a health check message."""
        hc = HealthCheck(message_id="hc-123")
        assert hc.message_type == A2AMessageType.HEALTH_CHECK

    def test_health_response_creation(self):
        """Test creating a health response."""
        resp = HealthResponse(
            message_id="hc-resp-123",
            correlation_id="hc-123",
            healthy=True,
            version="0.2.0",
            uptime_seconds=3600,
            active_sessions=5,
            load=0.5,
        )
        assert resp.healthy is True
        assert resp.version == "0.2.0"
        assert resp.uptime_seconds == 3600


class TestErrorMessage:
    """Tests for error messages."""

    def test_error_message_creation(self):
        """Test creating an error message."""
        err = ErrorMessage(
            message_id="err-123",
            error_code=A2AErrorCode.CAPABILITY_NOT_FOUND,
            error_message="Capability 'foo' not found",
            details={"requested": "foo"},
        )
        assert err.message_type == A2AMessageType.ERROR
        assert err.error_code == A2AErrorCode.CAPABILITY_NOT_FOUND


# ============================================================================
# AGENT CARD TESTS
# ============================================================================


class TestAgentCard:
    """Tests for AgentCard dataclass."""

    def test_agent_card_creation(self):
        """Test creating an agent card."""
        card = AgentCard(
            name="test-agent",
            description="A test agent",
            version="1.0.0",
            capabilities=["invoke", "handoff"],
        )
        assert card.name == "test-agent"
        assert card.version == "1.0.0"
        assert len(card.capabilities) == 2

    def test_has_capability_string(self):
        """Test checking capability with string."""
        card = AgentCard(
            name="test",
            description="test",
            version="1.0",
            capabilities=["invoke", "code_analysis"],
        )
        assert card.has_capability("invoke")
        assert card.has_capability("code_analysis")
        assert not card.has_capability("handoff")

    def test_has_capability_enum(self):
        """Test checking capability with enum."""
        card = AgentCard(
            name="test",
            description="test",
            version="1.0",
            capabilities=["invoke", "ansible_roles"],
        )
        assert card.has_capability(A2ACapability.INVOKE)
        assert card.has_capability(A2ACapability.ANSIBLE_ROLES)
        assert not card.has_capability(A2ACapability.HANDOFF)

    def test_agent_card_with_endpoints(self):
        """Test agent card with endpoints."""
        card = AgentCard(
            name="test",
            description="test",
            version="1.0",
            endpoints=[
                AgentEndpoint(
                    name="discover",
                    path="/a2a/discover",
                    method="GET",
                ),
                AgentEndpoint(
                    name="invoke",
                    path="/a2a/invoke",
                    method="POST",
                ),
            ],
        )
        assert len(card.endpoints) == 2
        assert card.get_endpoint("discover") is not None
        assert card.get_endpoint("discover").path == "/a2a/discover"
        assert card.get_endpoint("unknown") is None

    def test_agent_card_to_dict(self):
        """Test converting agent card to dictionary."""
        card = AgentCard(
            name="test-agent",
            description="A test agent",
            version="1.0.0",
            capabilities=["invoke"],
            tags=["test", "example"],
        )
        data = card.to_dict()
        assert data["name"] == "test-agent"
        assert data["version"] == "1.0.0"
        assert "invoke" in data["capabilities"]
        assert "test" in data["tags"]

    def test_agent_card_from_dict(self):
        """Test creating agent card from dictionary."""
        data = {
            "name": "loaded-agent",
            "description": "Loaded from dict",
            "version": "2.0.0",
            "capabilities": ["handoff", "testing"],
            "endpoints": [
                {"name": "health", "path": "/health", "method": "GET"},
            ],
        }
        card = AgentCard.from_dict(data)
        assert card.name == "loaded-agent"
        assert card.version == "2.0.0"
        assert len(card.endpoints) == 1

    def test_agent_card_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = AgentCard(
            name="roundtrip-test",
            description="Testing JSON roundtrip",
            version="1.0.0",
            capabilities=["invoke", "handoff"],
            metadata={"key": "value"},
        )
        json_str = original.to_json()
        loaded = AgentCard.from_json(json_str)
        assert loaded.name == original.name
        assert loaded.capabilities == original.capabilities
        assert loaded.metadata == original.metadata

    def test_agent_card_file_io(self, tmp_path):
        """Test saving and loading agent card from file."""
        card = AgentCard(
            name="file-test",
            description="File I/O test",
            version="1.0.0",
        )
        file_path = tmp_path / "agent-card.json"
        card.save_to_file(file_path)

        loaded = AgentCard.from_file(file_path)
        assert loaded.name == "file-test"

    def test_create_harness_agent_card(self):
        """Test creating the default harness agent card."""
        card = create_harness_agent_card(
            base_url="http://localhost:8000",
            version="0.2.0",
        )
        assert card.name == "dag-harness"
        assert card.version == "0.2.0"
        assert card.base_url == "http://localhost:8000"
        assert card.has_capability(A2ACapability.INVOKE)
        assert card.has_capability(A2ACapability.ANSIBLE_ROLES)
        assert card.has_capability(A2ACapability.MCP_CLIENT)
        assert len(card.endpoints) >= 4


# ============================================================================
# SERVER TESTS
# ============================================================================


class TestA2AServer:
    """Tests for A2A server."""

    def test_server_creation(self):
        """Test creating an A2A server."""
        server = A2AServer()
        assert server.agent_card is not None
        assert server.agent_card.name == "dag-harness"

    def test_server_with_custom_card(self):
        """Test server with custom agent card."""
        custom_card = AgentCard(
            name="custom-server",
            description="Custom server",
            version="1.0.0",
        )
        server = A2AServer(agent_card=custom_card)
        assert server.agent_card.name == "custom-server"

    def test_register_capability(self):
        """Test registering a capability handler."""
        server = A2AServer()

        def my_handler(task: str, context: dict):
            return {"result": "done"}

        server.register_capability(
            "custom_cap",
            handler=my_handler,
            description="Custom capability",
        )

        caps = server.get_registered_capabilities()
        assert "custom_cap" in caps
        assert "custom_cap" in server.agent_card.capabilities

    def test_handle_discover(self):
        """Test discovery endpoint handler."""
        server = A2AServer()
        result = server.handle_discover()
        assert result["name"] == "dag-harness"
        assert "capabilities" in result
        assert "endpoints" in result

    def test_handle_health(self):
        """Test health endpoint handler."""
        server = A2AServer(version="0.2.0")
        result = server.handle_health()
        assert result["healthy"] is True
        assert result["version"] == "0.2.0"
        assert "uptime_seconds" in result

    def test_handle_invoke_missing_capability(self):
        """Test invoking with missing capability."""
        server = A2AServer()

        with pytest.raises(A2AServerError) as exc_info:
            server.handle_invoke_sync(
                {
                    "capability": "nonexistent",
                    "task": "Do something",
                }
            )

        assert exc_info.value.error_code == A2AErrorCode.CAPABILITY_NOT_FOUND

    def test_handle_invoke_success(self):
        """Test successful invocation."""
        server = A2AServer()

        def handler(task: str, context: dict):
            return {"analyzed": True, "task": task}

        server.register_capability("test_cap", handler=handler)

        result = server.handle_invoke_sync(
            {
                "message_id": "test-123",
                "capability": "test_cap",
                "task": "Analyze code",
                "context": {},
            }
        )

        assert result["status"] == "success"
        assert result["result"]["analyzed"] is True
        assert "execution_time_ms" in result

    def test_handle_handoff_no_handler(self):
        """Test handoff with no handler registered."""
        server = A2AServer()

        result = server.handle_handoff_sync(
            {
                "task": "Continue work",
                "context": {},
            }
        )

        assert result["accepted"] is False
        assert "not supported" in result["reason"].lower()

    def test_handle_handoff_success(self):
        """Test successful handoff."""
        server = A2AServer()

        def handoff_handler(**kwargs):
            return {
                "accepted": True,
                "session_id": "new-session-123",
            }

        server.register_handoff_handler(handoff_handler)

        result = server.handle_handoff_sync(
            {
                "task": "Continue work",
                "context": {"key": "value"},
                "files_changed": ["file.py"],
                "reason": "Session limit",
            }
        )

        assert result["accepted"] is True
        assert result["session_id"] == "new-session-123"

    def test_server_stats(self):
        """Test getting server statistics."""
        server = A2AServer()
        server.register_capability("cap1", lambda **kw: None)
        server.register_capability("cap2", lambda **kw: None)

        stats = server.get_stats()
        assert stats["registered_capabilities"] == 2
        assert "cap1" in stats["capabilities"]
        assert "cap2" in stats["capabilities"]


@pytest.mark.asyncio
class TestA2AServerAsync:
    """Async tests for A2A server."""

    async def test_handle_invoke_async(self):
        """Test async invocation handler."""
        server = A2AServer()

        async def async_handler(task: str, context: dict):
            return {"async": True}

        server.register_capability(
            "async_cap",
            handler=async_handler,
            async_handler=True,
        )

        result = await server.handle_invoke(
            {
                "capability": "async_cap",
                "task": "Async task",
                "context": {},
            }
        )

        assert result["status"] == "success"
        assert result["result"]["async"] is True

    async def test_handle_handoff_async(self):
        """Test async handoff handler."""
        server = A2AServer()

        async def async_handoff(**kwargs):
            return {"accepted": True, "session_id": "async-session"}

        server.register_handoff_handler(async_handoff, async_handler=True)

        result = await server.handle_handoff(
            {
                "task": "Async handoff",
                "context": {},
            }
        )

        assert result["accepted"] is True


# ============================================================================
# CLIENT TESTS
# ============================================================================


class TestA2AClient:
    """Tests for A2A client."""

    def test_client_creation(self):
        """Test creating an A2A client."""
        client = A2AClient(timeout=30.0, max_retries=3)
        assert client.timeout == 30.0
        assert client.max_retries == 3

    def test_client_cache(self):
        """Test agent card caching."""
        client = A2AClient()
        assert len(client.get_cached_agents()) == 0
        client.clear_cache()
        assert len(client.get_cached_agents()) == 0


@pytest.mark.asyncio
class TestA2AClientAsync:
    """Async tests for A2A client."""

    async def test_discover_connection_error(self):
        """Test discovery with connection error."""
        client = A2AClient()

        with pytest.raises(AgentNotFoundError):
            await client.discover("http://nonexistent.local:9999")

        await client.close()

    async def test_discover_success(self):
        """Test successful discovery."""
        client = A2AClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "mock-agent",
            "description": "Mock agent",
            "version": "1.0.0",
            "capabilities": ["invoke"],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_client

            card = await client.discover("http://localhost:8000")
            assert card.name == "mock-agent"
            assert card.base_url == "http://localhost:8000"

        await client.close()

    async def test_invoke_capability_not_found(self):
        """Test invoking unsupported capability."""
        client = A2AClient()

        # Mock discovery to return a card without the capability
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "mock-agent",
            "description": "Mock",
            "version": "1.0",
            "capabilities": ["invoke"],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_client

            with pytest.raises(CapabilityNotFoundError):
                await client.invoke(
                    "http://localhost:8000",
                    "unsupported_capability",
                    "Do something",
                )

        await client.close()

    async def test_health_check_success(self):
        """Test successful health check."""
        client = A2AClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "healthy": True,
            "version": "0.2.0",
            "uptime_seconds": 1000,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_client

            health = await client.health_check("http://localhost:8000")
            assert health.healthy is True
            assert health.version == "0.2.0"

        await client.close()

    async def test_health_check_failure(self):
        """Test health check when agent is down."""
        client = A2AClient()

        with patch.object(client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))
            mock_get.return_value = mock_client

            health = await client.health_check("http://localhost:8000")
            assert health.healthy is False

        await client.close()


class TestA2AClientSync:
    """Sync tests for A2A client."""

    def test_discover_sync_connection_error(self):
        """Test sync discovery with connection error."""
        client = A2AClient()

        with pytest.raises(AgentNotFoundError):
            client.discover_sync("http://nonexistent.local:9999")

        client.close_sync()

    def test_health_check_sync(self):
        """Test sync health check."""
        client = A2AClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "healthy": True,
            "version": "0.2.0",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_sync_client") as mock_get:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get.return_value = mock_client

            health = client.health_check_sync("http://localhost:8000")
            assert health.healthy is True

        client.close_sync()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestA2AIntegration:
    """Integration tests for A2A protocol."""

    def test_server_client_discovery(self):
        """Test client discovering server capabilities."""
        # Create server
        server = A2AServer()
        server.register_capability(
            "test_cap",
            handler=lambda **kw: {"result": "ok"},
            description="Test capability",
        )

        # Get discovery data
        discovery_data = server.handle_discover()

        # Client loads the data
        card = AgentCard.from_dict(discovery_data)

        assert card.has_capability("test_cap")
        assert card.has_capability(A2ACapability.INVOKE)

    def test_end_to_end_invoke(self):
        """Test full invocation flow."""
        # Server setup
        server = A2AServer()

        def analyzer(task: str, context: dict):
            return {
                "task_received": task,
                "context_keys": list(context.keys()),
            }

        server.register_capability("analyzer", handler=analyzer)

        # Create invoke request
        request = InvokeRequest(
            message_id=str(uuid.uuid4()),
            capability="analyzer",
            task="Analyze this",
            context={"key1": "value1"},
        )

        # Handle request
        result = server.handle_invoke_sync(request.to_dict())

        assert result["status"] == "success"
        assert result["result"]["task_received"] == "Analyze this"
        assert "key1" in result["result"]["context_keys"]

    def test_handoff_flow(self):
        """Test handoff request flow."""
        server = A2AServer()

        sessions_created = []

        def handoff_handler(**kwargs):
            session_id = str(uuid.uuid4())
            sessions_created.append(
                {
                    "id": session_id,
                    "task": kwargs["task"],
                    "files": kwargs["files_changed"],
                }
            )
            return {
                "accepted": True,
                "session_id": session_id,
            }

        server.register_handoff_handler(handoff_handler)

        # Create handoff request
        request = HandoffRequest(
            message_id=str(uuid.uuid4()),
            task="Continue fixing",
            context={"progress": 75},
            files_changed=["a.py", "b.py"],
            reason="Token limit reached",
        )

        result = server.handle_handoff_sync(request.to_dict())

        assert result["accepted"] is True
        assert len(sessions_created) == 1
        assert sessions_created[0]["task"] == "Continue fixing"


# ============================================================================
# AGENT CARD JSON FILE TESTS
# ============================================================================


class TestAgentCardFile:
    """Tests for the agent-card.json file."""

    def test_load_harness_agent_card(self):
        """Test loading the harness agent card file."""
        card_path = Path(__file__).parent.parent / "agent-card.json"
        if card_path.exists():
            card = AgentCard.from_file(card_path)
            assert card.name == "dag-harness"
            assert card.has_capability("invoke")
            assert card.has_capability("handoff")
            assert len(card.endpoints) >= 4

    def test_agent_card_matches_code(self):
        """Test that agent-card.json matches create_harness_agent_card."""
        card_path = Path(__file__).parent.parent / "agent-card.json"
        if card_path.exists():
            file_card = AgentCard.from_file(card_path)
            code_card = create_harness_agent_card(version="0.2.0")

            assert file_card.name == code_card.name
            assert set(file_card.capabilities) == set(code_card.capabilities)
            assert len(file_card.endpoints) == len(code_card.endpoints)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_a2a_client_error(self):
        """Test A2AClientError exception."""
        error = A2AClientError(
            "Something went wrong",
            error_code=A2AErrorCode.INTERNAL_ERROR,
            details={"key": "value"},
        )
        assert str(error) == "Something went wrong"
        assert error.error_code == A2AErrorCode.INTERNAL_ERROR
        assert error.details == {"key": "value"}

    def test_agent_not_found_error(self):
        """Test AgentNotFoundError exception."""
        error = AgentNotFoundError("http://localhost:9999")
        assert "localhost:9999" in str(error)
        assert error.error_code == A2AErrorCode.AGENT_UNAVAILABLE

    def test_capability_not_found_error(self):
        """Test CapabilityNotFoundError exception."""
        error = CapabilityNotFoundError("unknown_cap", "test-agent")
        assert "unknown_cap" in str(error)
        assert "test-agent" in str(error)
        assert error.error_code == A2AErrorCode.CAPABILITY_NOT_FOUND

    def test_invocation_error(self):
        """Test InvocationError exception."""
        error = InvocationError(
            "Invocation failed",
            error_code=A2AErrorCode.TIMEOUT,
        )
        assert error.error_code == A2AErrorCode.TIMEOUT

    def test_a2a_server_error(self):
        """Test A2AServerError exception."""
        error = A2AServerError(
            "Server error",
            error_code=A2AErrorCode.CAPABILITY_NOT_FOUND,
            status_code=404,
        )
        assert error.status_code == 404
        assert error.error_code == A2AErrorCode.CAPABILITY_NOT_FOUND


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_capabilities(self):
        """Test agent card with no capabilities."""
        card = AgentCard(
            name="empty",
            description="Empty agent",
            version="1.0.0",
        )
        assert card.has_capability("anything") is False
        assert card.capabilities == []

    def test_empty_context(self):
        """Test invoke with empty context."""
        server = A2AServer()
        server.register_capability(
            "test",
            handler=lambda task, context: {"context_empty": len(context) == 0},
        )

        result = server.handle_invoke_sync(
            {
                "capability": "test",
                "task": "Test",
                "context": {},
            }
        )

        assert result["result"]["context_empty"] is True

    def test_large_context(self):
        """Test invoke with large context."""
        server = A2AServer()

        def handler(task, context):
            return {"keys_count": len(context)}

        server.register_capability("test", handler=handler)

        large_context = {f"key_{i}": f"value_{i}" for i in range(100)}

        result = server.handle_invoke_sync(
            {
                "capability": "test",
                "task": "Test",
                "context": large_context,
            }
        )

        assert result["result"]["keys_count"] == 100

    def test_special_characters_in_task(self):
        """Test task with special characters."""
        server = A2AServer()
        server.register_capability(
            "test",
            handler=lambda task, context: {"task": task},
        )

        special_task = "Analyze: `code` with 'quotes' and \"double quotes\""

        result = server.handle_invoke_sync(
            {
                "capability": "test",
                "task": special_task,
                "context": {},
            }
        )

        assert result["result"]["task"] == special_task
