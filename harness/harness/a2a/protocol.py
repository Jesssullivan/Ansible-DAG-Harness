"""A2A Protocol definitions and message types.

This module defines the core protocol for Agent-to-Agent (A2A) communication,
following patterns similar to MCP but focused on agent coordination and handoff.

Protocol flow:
1. Discovery: Agents discover each other via agent cards
2. Capability negotiation: Check supported capabilities
3. Invocation: Request an agent to perform a task
4. Handoff: Transfer context and control between agents
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class A2AMessageType(str, Enum):
    """Types of A2A protocol messages."""

    # Discovery messages
    DISCOVER_REQUEST = "discover_request"
    DISCOVER_RESPONSE = "discover_response"

    # Invocation messages
    INVOKE_REQUEST = "invoke_request"
    INVOKE_RESPONSE = "invoke_response"

    # Handoff messages
    HANDOFF_REQUEST = "handoff_request"
    HANDOFF_ACCEPT = "handoff_accept"
    HANDOFF_REJECT = "handoff_reject"
    HANDOFF_COMPLETE = "handoff_complete"

    # Status messages
    HEALTH_CHECK = "health_check"
    HEALTH_RESPONSE = "health_response"

    # Error messages
    ERROR = "error"


class A2ACapability(str, Enum):
    """Standard A2A capabilities that agents can advertise."""

    # Core capabilities
    INVOKE = "invoke"  # Can receive invocation requests
    HANDOFF = "handoff"  # Can accept handoff from other agents
    DISCOVERY = "discovery"  # Can respond to discovery requests

    # Task capabilities
    CODE_ANALYSIS = "code_analysis"  # Can analyze code
    CODE_MODIFICATION = "code_modification"  # Can modify code
    TESTING = "testing"  # Can run tests
    DOCUMENTATION = "documentation"  # Can generate docs

    # Domain capabilities
    ANSIBLE_ROLES = "ansible_roles"  # Understands Ansible role structure
    DEPENDENCY_ANALYSIS = "dependency_analysis"  # Can analyze dependencies
    WORKFLOW_EXECUTION = "workflow_execution"  # Can execute workflows

    # Integration capabilities
    MCP_CLIENT = "mcp_client"  # Has MCP client integration
    GITLAB_INTEGRATION = "gitlab_integration"  # Can interact with GitLab


class A2AStatus(str, Enum):
    """Status values for A2A operations."""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class A2AMessage:
    """Base class for A2A protocol messages."""

    message_type: A2AMessageType
    message_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    sender: str | None = None
    recipient: str | None = None
    correlation_id: str | None = None  # For request/response correlation
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_type": self.message_type.value,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "recipient": self.recipient,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "A2AMessage":
        """Create message from dictionary."""
        return cls(
            message_type=A2AMessageType(data["message_type"]),
            message_id=data["message_id"],
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            sender=data.get("sender"),
            recipient=data.get("recipient"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DiscoverRequest(A2AMessage):
    """Request to discover an agent's capabilities."""

    def __init__(
        self,
        message_id: str,
        sender: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            message_type=A2AMessageType.DISCOVER_REQUEST,
            message_id=message_id,
            sender=sender,
            **kwargs,
        )


@dataclass
class InvokeRequest(A2AMessage):
    """Request to invoke an agent capability."""

    capability: str = ""
    task: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    timeout: int = 300  # seconds

    def __init__(
        self,
        message_id: str,
        capability: str,
        task: str,
        context: dict[str, Any] | None = None,
        timeout: int = 300,
        sender: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            message_type=A2AMessageType.INVOKE_REQUEST,
            message_id=message_id,
            sender=sender,
            **kwargs,
        )
        self.capability = capability
        self.task = task
        self.context = context or {}
        self.timeout = timeout

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "capability": self.capability,
                "task": self.task,
                "context": self.context,
                "timeout": self.timeout,
            }
        )
        return data


@dataclass
class InvokeResponse(A2AMessage):
    """Response to an invocation request."""

    status: A2AStatus = A2AStatus.PENDING
    result: Any = None
    error: str | None = None
    execution_time_ms: int | None = None

    def __init__(
        self,
        message_id: str,
        correlation_id: str,
        status: A2AStatus = A2AStatus.SUCCESS,
        result: Any = None,
        error: str | None = None,
        execution_time_ms: int | None = None,
        sender: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            message_type=A2AMessageType.INVOKE_RESPONSE,
            message_id=message_id,
            correlation_id=correlation_id,
            sender=sender,
            **kwargs,
        )
        self.status = status
        self.result = result
        self.error = error
        self.execution_time_ms = execution_time_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "status": self.status.value,
                "result": self.result,
                "error": self.error,
                "execution_time_ms": self.execution_time_ms,
            }
        )
        return data


@dataclass
class HandoffRequest(A2AMessage):
    """Request to hand off work to another agent."""

    task: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    files_changed: list[str] = field(default_factory=list)
    reason: str = ""
    continuation_prompt: str = ""

    def __init__(
        self,
        message_id: str,
        task: str,
        context: dict[str, Any] | None = None,
        files_changed: list[str] | None = None,
        reason: str = "",
        continuation_prompt: str = "",
        sender: str | None = None,
        recipient: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            message_type=A2AMessageType.HANDOFF_REQUEST,
            message_id=message_id,
            sender=sender,
            recipient=recipient,
            **kwargs,
        )
        self.task = task
        self.context = context or {}
        self.files_changed = files_changed or []
        self.reason = reason
        self.continuation_prompt = continuation_prompt

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "task": self.task,
                "context": self.context,
                "files_changed": self.files_changed,
                "reason": self.reason,
                "continuation_prompt": self.continuation_prompt,
            }
        )
        return data


@dataclass
class HandoffResponse(A2AMessage):
    """Response to a handoff request."""

    accepted: bool = False
    session_id: str | None = None
    reason: str | None = None

    def __init__(
        self,
        message_id: str,
        correlation_id: str,
        accepted: bool,
        session_id: str | None = None,
        reason: str | None = None,
        sender: str | None = None,
        **kwargs: Any,
    ):
        msg_type = A2AMessageType.HANDOFF_ACCEPT if accepted else A2AMessageType.HANDOFF_REJECT
        super().__init__(
            message_type=msg_type,
            message_id=message_id,
            correlation_id=correlation_id,
            sender=sender,
            **kwargs,
        )
        self.accepted = accepted
        self.session_id = session_id
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "accepted": self.accepted,
                "session_id": self.session_id,
                "reason": self.reason,
            }
        )
        return data


@dataclass
class HealthCheck(A2AMessage):
    """Health check request."""

    def __init__(
        self,
        message_id: str,
        sender: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            message_type=A2AMessageType.HEALTH_CHECK,
            message_id=message_id,
            sender=sender,
            **kwargs,
        )


@dataclass
class HealthResponse(A2AMessage):
    """Health check response."""

    healthy: bool = True
    version: str = ""
    uptime_seconds: int = 0
    active_sessions: int = 0
    load: float = 0.0

    def __init__(
        self,
        message_id: str,
        correlation_id: str,
        healthy: bool = True,
        version: str = "",
        uptime_seconds: int = 0,
        active_sessions: int = 0,
        load: float = 0.0,
        sender: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            message_type=A2AMessageType.HEALTH_RESPONSE,
            message_id=message_id,
            correlation_id=correlation_id,
            sender=sender,
            **kwargs,
        )
        self.healthy = healthy
        self.version = version
        self.uptime_seconds = uptime_seconds
        self.active_sessions = active_sessions
        self.load = load

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "healthy": self.healthy,
                "version": self.version,
                "uptime_seconds": self.uptime_seconds,
                "active_sessions": self.active_sessions,
                "load": self.load,
            }
        )
        return data


@dataclass
class ErrorMessage(A2AMessage):
    """Error message."""

    error_code: str = ""
    error_message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        message_id: str,
        error_code: str,
        error_message: str,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        sender: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            message_type=A2AMessageType.ERROR,
            message_id=message_id,
            correlation_id=correlation_id,
            sender=sender,
            **kwargs,
        )
        self.error_code = error_code
        self.error_message = error_message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "error_code": self.error_code,
                "error_message": self.error_message,
                "details": self.details,
            }
        )
        return data


# Error codes
class A2AErrorCode:
    """Standard A2A error codes."""

    CAPABILITY_NOT_FOUND = "capability_not_found"
    AGENT_UNAVAILABLE = "agent_unavailable"
    HANDOFF_REJECTED = "handoff_rejected"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    INTERNAL_ERROR = "internal_error"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
