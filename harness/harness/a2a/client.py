"""A2A Client for agent discovery and invocation.

This module provides the client-side implementation of the A2A protocol,
allowing agents to discover other agents and invoke their capabilities.
"""

import logging
import time
import uuid
from typing import Any
from urllib.parse import urljoin

import httpx

from harness.a2a.agent_card import AgentCard
from harness.a2a.protocol import (
    A2ACapability,
    A2AErrorCode,
    A2AStatus,
    HandoffRequest,
    HandoffResponse,
    HealthResponse,
    InvokeRequest,
    InvokeResponse,
)

logger = logging.getLogger(__name__)


class A2AClientError(Exception):
    """Base exception for A2A client errors."""

    def __init__(
        self,
        message: str,
        error_code: str = A2AErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class AgentNotFoundError(A2AClientError):
    """Raised when an agent cannot be found."""

    def __init__(self, agent_url: str):
        super().__init__(
            f"Agent not found at {agent_url}",
            error_code=A2AErrorCode.AGENT_UNAVAILABLE,
        )


class CapabilityNotFoundError(A2AClientError):
    """Raised when a capability is not supported."""

    def __init__(self, capability: str, agent_name: str):
        super().__init__(
            f"Agent '{agent_name}' does not support capability '{capability}'",
            error_code=A2AErrorCode.CAPABILITY_NOT_FOUND,
        )


class InvocationError(A2AClientError):
    """Raised when an invocation fails."""

    pass


class A2AClient:
    """
    Client for A2A protocol communication.

    This client handles:
    - Agent discovery via agent cards
    - Capability invocation
    - Handoff requests
    - Health checks

    Example:
        client = A2AClient()
        card = await client.discover("http://localhost:8000")
        if card.has_capability("code_analysis"):
            result = await client.invoke(
                "http://localhost:8000",
                "code_analysis",
                "Analyze the file structure"
            )
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the A2A client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._discovered_agents: dict[str, AgentCard] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _get_sync_client(self) -> httpx.Client:
        """Get or create the sync HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.timeout)
        return self._sync_client

    async def close(self) -> None:
        """Close the client and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def close_sync(self) -> None:
        """Close the sync client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    # =========================================================================
    # DISCOVERY
    # =========================================================================

    async def discover(self, agent_url: str, force: bool = False) -> AgentCard:
        """Discover an agent's capabilities.

        Args:
            agent_url: Base URL of the agent
            force: If True, bypass cache and re-discover

        Returns:
            AgentCard describing the agent

        Raises:
            AgentNotFoundError: If agent cannot be reached
            A2AClientError: For other errors
        """
        # Check cache first
        if not force and agent_url in self._discovered_agents:
            return self._discovered_agents[agent_url]

        client = await self._get_client()
        discover_url = urljoin(agent_url.rstrip("/") + "/", "a2a/discover")

        try:
            response = await client.get(discover_url)
            response.raise_for_status()
            data = response.json()

            card = AgentCard.from_dict(data)
            card.base_url = agent_url
            self._discovered_agents[agent_url] = card

            logger.info(f"Discovered agent '{card.name}' at {agent_url}")
            return card

        except httpx.ConnectError as e:
            raise AgentNotFoundError(agent_url) from e
        except httpx.HTTPStatusError as e:
            raise A2AClientError(
                f"Discovery failed: {e.response.status_code}",
                error_code=A2AErrorCode.AGENT_UNAVAILABLE,
            ) from e
        except Exception as e:
            raise A2AClientError(f"Discovery failed: {e}") from e

    def discover_sync(self, agent_url: str, force: bool = False) -> AgentCard:
        """Synchronous version of discover.

        Args:
            agent_url: Base URL of the agent
            force: If True, bypass cache and re-discover

        Returns:
            AgentCard describing the agent
        """
        if not force and agent_url in self._discovered_agents:
            return self._discovered_agents[agent_url]

        client = self._get_sync_client()
        discover_url = urljoin(agent_url.rstrip("/") + "/", "a2a/discover")

        try:
            response = client.get(discover_url)
            response.raise_for_status()
            data = response.json()

            card = AgentCard.from_dict(data)
            card.base_url = agent_url
            self._discovered_agents[agent_url] = card

            logger.info(f"Discovered agent '{card.name}' at {agent_url}")
            return card

        except httpx.ConnectError as e:
            raise AgentNotFoundError(agent_url) from e
        except httpx.HTTPStatusError as e:
            raise A2AClientError(
                f"Discovery failed: {e.response.status_code}",
                error_code=A2AErrorCode.AGENT_UNAVAILABLE,
            ) from e
        except Exception as e:
            raise A2AClientError(f"Discovery failed: {e}") from e

    # =========================================================================
    # INVOCATION
    # =========================================================================

    async def invoke(
        self,
        agent_url: str,
        capability: str | A2ACapability,
        task: str,
        context: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> InvokeResponse:
        """Invoke a capability on an agent.

        Args:
            agent_url: Base URL of the agent
            capability: Capability to invoke
            task: Task description
            context: Additional context for the task
            timeout: Override default timeout (seconds)

        Returns:
            InvokeResponse with the result

        Raises:
            CapabilityNotFoundError: If capability not supported
            InvocationError: If invocation fails
        """
        # Discover agent first if not cached
        card = await self.discover(agent_url)

        cap_str = capability.value if isinstance(capability, A2ACapability) else capability
        if not card.has_capability(cap_str):
            raise CapabilityNotFoundError(cap_str, card.name)

        client = await self._get_client()
        invoke_url = urljoin(agent_url.rstrip("/") + "/", "a2a/invoke")

        request = InvokeRequest(
            message_id=str(uuid.uuid4()),
            capability=cap_str,
            task=task,
            context=context or {},
            timeout=timeout or int(self.timeout),
        )

        try:
            start_time = time.time()
            response = await client.post(
                invoke_url,
                json=request.to_dict(),
                timeout=timeout or self.timeout,
            )
            execution_time = int((time.time() - start_time) * 1000)
            response.raise_for_status()

            data = response.json()
            return InvokeResponse(
                message_id=data.get("message_id", str(uuid.uuid4())),
                correlation_id=request.message_id,
                status=A2AStatus(data.get("status", "success")),
                result=data.get("result"),
                error=data.get("error"),
                execution_time_ms=execution_time,
            )

        except httpx.TimeoutException as e:
            raise InvocationError(
                f"Invocation timed out after {timeout or self.timeout}s",
                error_code=A2AErrorCode.TIMEOUT,
            ) from e
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            raise InvocationError(
                error_data.get("error", f"HTTP {e.response.status_code}"),
                error_code=error_data.get("error_code", A2AErrorCode.INTERNAL_ERROR),
                details=error_data,
            ) from e
        except Exception as e:
            raise InvocationError(f"Invocation failed: {e}") from e

    def invoke_sync(
        self,
        agent_url: str,
        capability: str | A2ACapability,
        task: str,
        context: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> InvokeResponse:
        """Synchronous version of invoke.

        Args:
            agent_url: Base URL of the agent
            capability: Capability to invoke
            task: Task description
            context: Additional context for the task
            timeout: Override default timeout (seconds)

        Returns:
            InvokeResponse with the result
        """
        card = self.discover_sync(agent_url)

        cap_str = capability.value if isinstance(capability, A2ACapability) else capability
        if not card.has_capability(cap_str):
            raise CapabilityNotFoundError(cap_str, card.name)

        client = self._get_sync_client()
        invoke_url = urljoin(agent_url.rstrip("/") + "/", "a2a/invoke")

        request = InvokeRequest(
            message_id=str(uuid.uuid4()),
            capability=cap_str,
            task=task,
            context=context or {},
            timeout=timeout or int(self.timeout),
        )

        try:
            start_time = time.time()
            response = client.post(
                invoke_url,
                json=request.to_dict(),
                timeout=timeout or self.timeout,
            )
            execution_time = int((time.time() - start_time) * 1000)
            response.raise_for_status()

            data = response.json()
            return InvokeResponse(
                message_id=data.get("message_id", str(uuid.uuid4())),
                correlation_id=request.message_id,
                status=A2AStatus(data.get("status", "success")),
                result=data.get("result"),
                error=data.get("error"),
                execution_time_ms=execution_time,
            )

        except httpx.TimeoutException as e:
            raise InvocationError(
                f"Invocation timed out after {timeout or self.timeout}s",
                error_code=A2AErrorCode.TIMEOUT,
            ) from e
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            raise InvocationError(
                error_data.get("error", f"HTTP {e.response.status_code}"),
                error_code=error_data.get("error_code", A2AErrorCode.INTERNAL_ERROR),
                details=error_data,
            ) from e
        except Exception as e:
            raise InvocationError(f"Invocation failed: {e}") from e

    # =========================================================================
    # HANDOFF
    # =========================================================================

    async def request_handoff(
        self,
        agent_url: str,
        task: str,
        context: dict[str, Any] | None = None,
        files_changed: list[str] | None = None,
        reason: str = "",
        continuation_prompt: str = "",
    ) -> HandoffResponse:
        """Request a handoff to another agent.

        Args:
            agent_url: Base URL of the target agent
            task: Task description to hand off
            context: Context to transfer
            files_changed: List of files changed
            reason: Reason for handoff
            continuation_prompt: Prompt for continuing agent

        Returns:
            HandoffResponse indicating acceptance or rejection

        Raises:
            A2AClientError: If handoff request fails
        """
        card = await self.discover(agent_url)

        if not card.has_capability(A2ACapability.HANDOFF):
            raise CapabilityNotFoundError(A2ACapability.HANDOFF.value, card.name)

        client = await self._get_client()
        handoff_url = urljoin(agent_url.rstrip("/") + "/", "a2a/handoff")

        request = HandoffRequest(
            message_id=str(uuid.uuid4()),
            task=task,
            context=context or {},
            files_changed=files_changed or [],
            reason=reason,
            continuation_prompt=continuation_prompt,
        )

        try:
            response = await client.post(handoff_url, json=request.to_dict())
            response.raise_for_status()
            data = response.json()

            return HandoffResponse(
                message_id=data.get("message_id", str(uuid.uuid4())),
                correlation_id=request.message_id,
                accepted=data.get("accepted", False),
                session_id=data.get("session_id"),
                reason=data.get("reason"),
            )

        except Exception as e:
            raise A2AClientError(
                f"Handoff request failed: {e}",
                error_code=A2AErrorCode.HANDOFF_REJECTED,
            ) from e

    def request_handoff_sync(
        self,
        agent_url: str,
        task: str,
        context: dict[str, Any] | None = None,
        files_changed: list[str] | None = None,
        reason: str = "",
        continuation_prompt: str = "",
    ) -> HandoffResponse:
        """Synchronous version of request_handoff."""
        card = self.discover_sync(agent_url)

        if not card.has_capability(A2ACapability.HANDOFF):
            raise CapabilityNotFoundError(A2ACapability.HANDOFF.value, card.name)

        client = self._get_sync_client()
        handoff_url = urljoin(agent_url.rstrip("/") + "/", "a2a/handoff")

        request = HandoffRequest(
            message_id=str(uuid.uuid4()),
            task=task,
            context=context or {},
            files_changed=files_changed or [],
            reason=reason,
            continuation_prompt=continuation_prompt,
        )

        try:
            response = client.post(handoff_url, json=request.to_dict())
            response.raise_for_status()
            data = response.json()

            return HandoffResponse(
                message_id=data.get("message_id", str(uuid.uuid4())),
                correlation_id=request.message_id,
                accepted=data.get("accepted", False),
                session_id=data.get("session_id"),
                reason=data.get("reason"),
            )

        except Exception as e:
            raise A2AClientError(
                f"Handoff request failed: {e}",
                error_code=A2AErrorCode.HANDOFF_REJECTED,
            ) from e

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    async def health_check(self, agent_url: str) -> HealthResponse:
        """Check agent health.

        Args:
            agent_url: Base URL of the agent

        Returns:
            HealthResponse with health status
        """
        client = await self._get_client()
        health_url = urljoin(agent_url.rstrip("/") + "/", "a2a/health")

        try:
            response = await client.get(health_url)
            response.raise_for_status()
            data = response.json()

            return HealthResponse(
                message_id=str(uuid.uuid4()),
                correlation_id="",
                healthy=data.get("healthy", True),
                version=data.get("version", ""),
                uptime_seconds=data.get("uptime_seconds", 0),
                active_sessions=data.get("active_sessions", 0),
                load=data.get("load", 0.0),
            )

        except Exception:
            return HealthResponse(
                message_id=str(uuid.uuid4()),
                correlation_id="",
                healthy=False,
            )

    def health_check_sync(self, agent_url: str) -> HealthResponse:
        """Synchronous version of health_check."""
        client = self._get_sync_client()
        health_url = urljoin(agent_url.rstrip("/") + "/", "a2a/health")

        try:
            response = client.get(health_url)
            response.raise_for_status()
            data = response.json()

            return HealthResponse(
                message_id=str(uuid.uuid4()),
                correlation_id="",
                healthy=data.get("healthy", True),
                version=data.get("version", ""),
                uptime_seconds=data.get("uptime_seconds", 0),
                active_sessions=data.get("active_sessions", 0),
                load=data.get("load", 0.0),
            )

        except Exception:
            return HealthResponse(
                message_id=str(uuid.uuid4()),
                correlation_id="",
                healthy=False,
            )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_cached_agents(self) -> dict[str, AgentCard]:
        """Get all cached agent cards.

        Returns:
            Dictionary of agent URL to AgentCard
        """
        return self._discovered_agents.copy()

    def clear_cache(self) -> None:
        """Clear the discovered agents cache."""
        self._discovered_agents.clear()
