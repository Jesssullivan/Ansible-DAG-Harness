"""A2A Server for handling agent-to-agent requests.

This module provides the server-side implementation of the A2A protocol,
exposing endpoints for discovery, invocation, handoff, and health checks.
"""

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from harness.a2a.agent_card import AgentCard, create_harness_agent_card
from harness.a2a.protocol import (
    A2ACapability,
    A2AErrorCode,
    A2AStatus,
)

logger = logging.getLogger(__name__)


class A2AServerError(Exception):
    """Base exception for A2A server errors."""

    def __init__(
        self,
        message: str,
        error_code: str = A2AErrorCode.INTERNAL_ERROR,
        status_code: int = 500,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


class CapabilityHandler:
    """Handler for a specific capability."""

    def __init__(
        self,
        capability: str,
        handler: Callable[..., Any],
        description: str = "",
        async_handler: bool = False,
    ):
        self.capability = capability
        self.handler = handler
        self.description = description
        self.async_handler = async_handler


class A2AServer:
    """
    Server for A2A protocol communication.

    This server provides:
    - /a2a/discover - Agent card discovery
    - /a2a/invoke - Capability invocation
    - /a2a/handoff - Agent handoff
    - /a2a/health - Health check

    The server can be integrated with FastAPI/Starlette or used standalone.

    Example:
        server = A2AServer()
        server.register_capability(
            "code_analysis",
            handler=analyze_code,
            description="Analyze code structure"
        )

        # With FastAPI
        app = FastAPI()
        server.mount(app)

        # Or get routes for manual mounting
        routes = server.get_routes()
    """

    def __init__(
        self,
        agent_card: AgentCard | None = None,
        version: str = "0.2.0",
    ):
        """Initialize the A2A server.

        Args:
            agent_card: Custom agent card (uses default if not provided)
            version: Server version
        """
        self._agent_card = agent_card or create_harness_agent_card(version=version)
        self._version = version
        self._start_time = time.time()
        self._handlers: dict[str, CapabilityHandler] = {}
        self._active_sessions: int = 0
        self._handoff_handler: Callable[..., Any] | None = None
        self._async_handoff_handler: bool = False

    @property
    def agent_card(self) -> AgentCard:
        """Get the server's agent card."""
        return self._agent_card

    # =========================================================================
    # CAPABILITY REGISTRATION
    # =========================================================================

    def register_capability(
        self,
        capability: str | A2ACapability,
        handler: Callable[..., Any],
        description: str = "",
        async_handler: bool = False,
    ) -> None:
        """Register a capability handler.

        Args:
            capability: Capability name or enum
            handler: Function to handle invocation requests
            description: Human-readable description
            async_handler: True if handler is async
        """
        cap_str = capability.value if isinstance(capability, A2ACapability) else capability
        self._handlers[cap_str] = CapabilityHandler(
            capability=cap_str,
            handler=handler,
            description=description,
            async_handler=async_handler,
        )

        # Update agent card capabilities
        if cap_str not in self._agent_card.capabilities:
            self._agent_card.capabilities.append(cap_str)

        logger.info(f"Registered capability: {cap_str}")

    def register_handoff_handler(
        self,
        handler: Callable[..., Any],
        async_handler: bool = False,
    ) -> None:
        """Register the handoff handler.

        Args:
            handler: Function to handle handoff requests
            async_handler: True if handler is async
        """
        self._handoff_handler = handler
        self._async_handoff_handler = async_handler
        logger.info("Registered handoff handler")

    # =========================================================================
    # ENDPOINT HANDLERS
    # =========================================================================

    def handle_discover(self) -> dict[str, Any]:
        """Handle discovery request.

        Returns:
            Agent card as dictionary
        """
        return self._agent_card.to_dict()

    def handle_health(self) -> dict[str, Any]:
        """Handle health check request.

        Returns:
            Health status dictionary
        """
        uptime = int(time.time() - self._start_time)
        return {
            "healthy": True,
            "version": self._version,
            "uptime_seconds": uptime,
            "active_sessions": self._active_sessions,
            "load": self._active_sessions / 10.0,  # Simple load calculation
            "capabilities_registered": len(self._handlers),
        }

    async def handle_invoke(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Handle invocation request (async).

        Args:
            request_data: Request data dictionary

        Returns:
            Response dictionary
        """
        start_time = time.time()
        message_id = str(uuid.uuid4())

        try:
            capability = request_data.get("capability", "")
            task = request_data.get("task", "")
            context = request_data.get("context", {})

            if not capability:
                raise A2AServerError(
                    "Missing capability in request",
                    error_code=A2AErrorCode.INVALID_REQUEST,
                    status_code=400,
                )

            if capability not in self._handlers:
                raise A2AServerError(
                    f"Capability '{capability}' not found",
                    error_code=A2AErrorCode.CAPABILITY_NOT_FOUND,
                    status_code=404,
                )

            handler = self._handlers[capability]
            self._active_sessions += 1

            try:
                if handler.async_handler:
                    result = await handler.handler(task=task, context=context)
                else:
                    result = handler.handler(task=task, context=context)

                execution_time = int((time.time() - start_time) * 1000)

                return {
                    "message_id": message_id,
                    "correlation_id": request_data.get("message_id"),
                    "status": A2AStatus.SUCCESS.value,
                    "result": result,
                    "execution_time_ms": execution_time,
                }

            finally:
                self._active_sessions -= 1

        except A2AServerError:
            raise
        except Exception as e:
            logger.exception(f"Invocation failed: {e}")
            raise A2AServerError(
                str(e),
                error_code=A2AErrorCode.INTERNAL_ERROR,
                status_code=500,
            ) from e

    def handle_invoke_sync(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Handle invocation request (sync).

        Args:
            request_data: Request data dictionary

        Returns:
            Response dictionary
        """
        start_time = time.time()
        message_id = str(uuid.uuid4())

        try:
            capability = request_data.get("capability", "")
            task = request_data.get("task", "")
            context = request_data.get("context", {})

            if not capability:
                raise A2AServerError(
                    "Missing capability in request",
                    error_code=A2AErrorCode.INVALID_REQUEST,
                    status_code=400,
                )

            if capability not in self._handlers:
                raise A2AServerError(
                    f"Capability '{capability}' not found",
                    error_code=A2AErrorCode.CAPABILITY_NOT_FOUND,
                    status_code=404,
                )

            handler = self._handlers[capability]

            if handler.async_handler:
                raise A2AServerError(
                    "Async handler cannot be used with sync invocation",
                    error_code=A2AErrorCode.INTERNAL_ERROR,
                    status_code=500,
                )

            self._active_sessions += 1

            try:
                result = handler.handler(task=task, context=context)
                execution_time = int((time.time() - start_time) * 1000)

                return {
                    "message_id": message_id,
                    "correlation_id": request_data.get("message_id"),
                    "status": A2AStatus.SUCCESS.value,
                    "result": result,
                    "execution_time_ms": execution_time,
                }

            finally:
                self._active_sessions -= 1

        except A2AServerError:
            raise
        except Exception as e:
            logger.exception(f"Invocation failed: {e}")
            raise A2AServerError(
                str(e),
                error_code=A2AErrorCode.INTERNAL_ERROR,
                status_code=500,
            ) from e

    async def handle_handoff(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Handle handoff request (async).

        Args:
            request_data: Request data dictionary

        Returns:
            Response dictionary
        """
        message_id = str(uuid.uuid4())

        if not self._handoff_handler:
            return {
                "message_id": message_id,
                "correlation_id": request_data.get("message_id"),
                "accepted": False,
                "reason": "Handoff not supported",
            }

        try:
            task = request_data.get("task", "")
            context = request_data.get("context", {})
            files_changed = request_data.get("files_changed", [])
            reason = request_data.get("reason", "")
            continuation_prompt = request_data.get("continuation_prompt", "")

            self._active_sessions += 1

            try:
                if self._async_handoff_handler:
                    result = await self._handoff_handler(
                        task=task,
                        context=context,
                        files_changed=files_changed,
                        reason=reason,
                        continuation_prompt=continuation_prompt,
                    )
                else:
                    result = self._handoff_handler(
                        task=task,
                        context=context,
                        files_changed=files_changed,
                        reason=reason,
                        continuation_prompt=continuation_prompt,
                    )

                # Result should be a dict with 'accepted', 'session_id', 'reason'
                return {
                    "message_id": message_id,
                    "correlation_id": request_data.get("message_id"),
                    "accepted": result.get("accepted", False),
                    "session_id": result.get("session_id"),
                    "reason": result.get("reason"),
                }

            finally:
                self._active_sessions -= 1

        except Exception as e:
            logger.exception(f"Handoff failed: {e}")
            return {
                "message_id": message_id,
                "correlation_id": request_data.get("message_id"),
                "accepted": False,
                "reason": str(e),
            }

    def handle_handoff_sync(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Handle handoff request (sync).

        Args:
            request_data: Request data dictionary

        Returns:
            Response dictionary
        """
        message_id = str(uuid.uuid4())

        if not self._handoff_handler:
            return {
                "message_id": message_id,
                "correlation_id": request_data.get("message_id"),
                "accepted": False,
                "reason": "Handoff not supported",
            }

        try:
            task = request_data.get("task", "")
            context = request_data.get("context", {})
            files_changed = request_data.get("files_changed", [])
            reason = request_data.get("reason", "")
            continuation_prompt = request_data.get("continuation_prompt", "")

            if self._async_handoff_handler:
                raise A2AServerError(
                    "Async handler cannot be used with sync invocation",
                    error_code=A2AErrorCode.INTERNAL_ERROR,
                    status_code=500,
                )

            self._active_sessions += 1

            try:
                result = self._handoff_handler(
                    task=task,
                    context=context,
                    files_changed=files_changed,
                    reason=reason,
                    continuation_prompt=continuation_prompt,
                )

                return {
                    "message_id": message_id,
                    "correlation_id": request_data.get("message_id"),
                    "accepted": result.get("accepted", False),
                    "session_id": result.get("session_id"),
                    "reason": result.get("reason"),
                }

            finally:
                self._active_sessions -= 1

        except Exception as e:
            logger.exception(f"Handoff failed: {e}")
            return {
                "message_id": message_id,
                "correlation_id": request_data.get("message_id"),
                "accepted": False,
                "reason": str(e),
            }

    # =========================================================================
    # FASTAPI/STARLETTE INTEGRATION
    # =========================================================================

    def create_fastapi_router(self) -> Any:
        """Create a FastAPI router with A2A endpoints.

        Returns:
            FastAPI APIRouter

        Raises:
            ImportError: If FastAPI is not installed
        """
        try:
            from fastapi import APIRouter, Request
            from fastapi.responses import JSONResponse
        except ImportError as e:
            raise ImportError(
                "FastAPI is required for router creation. "
                "Install with: pip install 'dag-harness[proxy]'"
            ) from e

        router = APIRouter(prefix="/a2a", tags=["A2A Protocol"])

        @router.get("/discover")
        async def discover():
            """Discover agent capabilities."""
            return JSONResponse(content=self.handle_discover())

        @router.get("/health")
        async def health():
            """Health check endpoint."""
            return JSONResponse(content=self.handle_health())

        @router.post("/invoke")
        async def invoke(request: Request):
            """Invoke an agent capability."""
            try:
                data = await request.json()
                result = await self.handle_invoke(data)
                return JSONResponse(content=result)
            except A2AServerError as e:
                return JSONResponse(
                    status_code=e.status_code,
                    content={
                        "error": str(e),
                        "error_code": e.error_code,
                    },
                )

        @router.post("/handoff")
        async def handoff(request: Request):
            """Request handoff to this agent."""
            try:
                data = await request.json()
                result = await self.handle_handoff(data)
                return JSONResponse(content=result)
            except A2AServerError as e:
                return JSONResponse(
                    status_code=e.status_code,
                    content={
                        "error": str(e),
                        "error_code": e.error_code,
                    },
                )

        return router

    def mount_to_app(self, app: Any) -> None:
        """Mount A2A routes to a FastAPI/Starlette app.

        Args:
            app: FastAPI or Starlette application
        """
        router = self.create_fastapi_router()
        app.include_router(router)
        logger.info("Mounted A2A routes to application")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_registered_capabilities(self) -> list[str]:
        """Get list of registered capabilities.

        Returns:
            List of capability names
        """
        return list(self._handlers.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "version": self._version,
            "uptime_seconds": int(time.time() - self._start_time),
            "active_sessions": self._active_sessions,
            "registered_capabilities": len(self._handlers),
            "capabilities": list(self._handlers.keys()),
            "handoff_enabled": self._handoff_handler is not None,
        }
