"""A2A (Agent-to-Agent) Protocol implementation for the DAG harness.

This module provides a protocol for agent-to-agent communication, enabling:
- Agent discovery via agent cards
- Capability invocation
- Agent handoff and context transfer
- Health monitoring

Example usage:

    # Client-side discovery and invocation
    from harness.a2a import A2AClient, A2ACapability

    client = A2AClient()
    card = await client.discover("http://localhost:8000")

    if card.has_capability(A2ACapability.CODE_ANALYSIS):
        result = await client.invoke(
            "http://localhost:8000",
            A2ACapability.CODE_ANALYSIS,
            "Analyze the codebase structure"
        )

    # Server-side capability registration
    from harness.a2a import A2AServer, A2ACapability

    server = A2AServer()
    server.register_capability(
        A2ACapability.CODE_ANALYSIS,
        handler=my_analysis_function,
        description="Analyze code structure"
    )

    # Mount to FastAPI
    app = FastAPI()
    server.mount_to_app(app)
"""

from harness.a2a.agent_card import (
    AgentCard,
    AgentEndpoint,
    create_harness_agent_card,
)
from harness.a2a.client import (
    A2AClient,
    A2AClientError,
    AgentNotFoundError,
    CapabilityNotFoundError,
    InvocationError,
)
from harness.a2a.protocol import (
    A2ACapability,
    A2AErrorCode,
    A2AMessage,
    A2AMessageType,
    A2AStatus,
    DiscoverRequest,
    ErrorMessage,
    HandoffRequest,
    HandoffResponse,
    HealthCheck,
    HealthResponse,
    InvokeRequest,
    InvokeResponse,
)
from harness.a2a.server import (
    A2AServer,
    A2AServerError,
    CapabilityHandler,
)
from harness.a2a.supervisor_integration import (
    A2ASupervisorIntegration,
    create_a2a_integration,
)

__all__ = [
    # Agent Card
    "AgentCard",
    "AgentEndpoint",
    "create_harness_agent_card",
    # Client
    "A2AClient",
    "A2AClientError",
    "AgentNotFoundError",
    "CapabilityNotFoundError",
    "InvocationError",
    # Protocol
    "A2ACapability",
    "A2AErrorCode",
    "A2AMessage",
    "A2AMessageType",
    "A2AStatus",
    "DiscoverRequest",
    "ErrorMessage",
    "HandoffRequest",
    "HandoffResponse",
    "HealthCheck",
    "HealthResponse",
    "InvokeRequest",
    "InvokeResponse",
    # Server
    "A2AServer",
    "A2AServerError",
    "CapabilityHandler",
    # Supervisor Integration
    "A2ASupervisorIntegration",
    "create_a2a_integration",
]
