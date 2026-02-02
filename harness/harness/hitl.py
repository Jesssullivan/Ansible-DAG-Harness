"""
Human-in-the-Loop (HITL) integration for workflow breakpoints.

Provides:
- Approval request mechanism
- Input collection from humans
- Breakpoint management
- Resume workflow with human input
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from harness.db.state import StateDB
from harness.notifications import (
    NotificationService,
    notify_human_needed,
)


class HumanInputType(str, Enum):
    """Types of human input requests."""

    APPROVAL = "approval"  # Yes/No approval
    TEXT = "text"  # Free-form text input
    CHOICE = "choice"  # Select from options
    CONFIRMATION = "confirmation"  # Confirm to proceed
    CREDENTIAL = "credential"  # Credential input (masked)


@dataclass
class HumanInputRequest:
    """A request for human input."""

    id: str
    execution_id: int
    node_name: str
    input_type: HumanInputType
    prompt: str
    options: list[str] | None = None  # For CHOICE type
    default: str | None = None
    required: bool = True
    timeout_seconds: int | None = None  # None = no timeout
    created_at: datetime = field(default_factory=datetime.utcnow)
    responded_at: datetime | None = None
    response: Any | None = None


@dataclass
class HumanInputResponse:
    """Response to a human input request."""

    request_id: str
    value: Any
    responded_at: datetime = field(default_factory=datetime.utcnow)
    responder: str | None = None  # Who provided the input


class HumanInputHandler:
    """
    Handler for human-in-the-loop interactions.

    Manages:
    - Pending input requests
    - Input collection and validation
    - Notification of pending requests
    - Resume workflow with collected input
    """

    def __init__(self, db: StateDB, notification_service: NotificationService | None = None):
        self.db = db
        self.notification_service = notification_service
        self._pending_requests: dict[str, HumanInputRequest] = {}
        self._response_callbacks: dict[str, Callable[[HumanInputResponse], None]] = {}
        self._request_counter = 0

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"hitl-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self._request_counter}"

    async def request_approval(
        self, execution_id: int, node_name: str, prompt: str, timeout_seconds: int | None = None
    ) -> bool:
        """
        Request approval from a human.

        Args:
            execution_id: Workflow execution ID
            node_name: Current node name
            prompt: Approval prompt text
            timeout_seconds: Optional timeout

        Returns:
            True if approved, False if denied or timed out
        """
        request = HumanInputRequest(
            id=self._generate_request_id(),
            execution_id=execution_id,
            node_name=node_name,
            input_type=HumanInputType.APPROVAL,
            prompt=prompt,
            options=["approve", "deny"],
            timeout_seconds=timeout_seconds,
        )

        response = await self._wait_for_input(request)

        if response is None:
            return False

        return response.value in (True, "approve", "yes", "y", 1)

    async def request_text(
        self,
        execution_id: int,
        node_name: str,
        prompt: str,
        default: str | None = None,
        required: bool = True,
        timeout_seconds: int | None = None,
    ) -> str | None:
        """
        Request text input from a human.

        Args:
            execution_id: Workflow execution ID
            node_name: Current node name
            prompt: Input prompt
            default: Default value if not provided
            required: Whether input is required
            timeout_seconds: Optional timeout

        Returns:
            Text input or None if timed out/not provided
        """
        request = HumanInputRequest(
            id=self._generate_request_id(),
            execution_id=execution_id,
            node_name=node_name,
            input_type=HumanInputType.TEXT,
            prompt=prompt,
            default=default,
            required=required,
            timeout_seconds=timeout_seconds,
        )

        response = await self._wait_for_input(request)

        if response is None:
            return default

        return str(response.value) if response.value is not None else default

    async def request_choice(
        self,
        execution_id: int,
        node_name: str,
        prompt: str,
        options: list[str],
        default: str | None = None,
        timeout_seconds: int | None = None,
    ) -> str | None:
        """
        Request selection from options.

        Args:
            execution_id: Workflow execution ID
            node_name: Current node name
            prompt: Selection prompt
            options: Available options
            default: Default selection
            timeout_seconds: Optional timeout

        Returns:
            Selected option or None
        """
        request = HumanInputRequest(
            id=self._generate_request_id(),
            execution_id=execution_id,
            node_name=node_name,
            input_type=HumanInputType.CHOICE,
            prompt=prompt,
            options=options,
            default=default,
            timeout_seconds=timeout_seconds,
        )

        response = await self._wait_for_input(request)

        if response is None:
            return default

        value = str(response.value)
        return value if value in options else default

    async def request_confirmation(
        self, execution_id: int, node_name: str, prompt: str, timeout_seconds: int | None = None
    ) -> bool:
        """
        Request confirmation to proceed.

        Args:
            execution_id: Workflow execution ID
            node_name: Current node name
            prompt: Confirmation prompt
            timeout_seconds: Optional timeout

        Returns:
            True if confirmed
        """
        return await self.request_approval(execution_id, node_name, prompt, timeout_seconds)

    async def _wait_for_input(self, request: HumanInputRequest) -> HumanInputResponse | None:
        """
        Wait for human input with optional timeout.

        This stores the pending request and waits for submit_response().
        """
        self._pending_requests[request.id] = request

        # Send notification
        if self.notification_service:
            # Get role name from execution
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT r.name FROM workflow_executions we
                    JOIN roles r ON we.role_id = r.id
                    WHERE we.id = ?
                    """,
                    (request.execution_id,),
                ).fetchone()
                role_name = row["name"] if row else "unknown"

            await notify_human_needed(
                self.notification_service,
                role_name=role_name,
                execution_id=request.execution_id,
                reason=request.prompt,
                node_name=request.node_name,
            )

        # Store request in database for persistence
        self._save_request_to_db(request)

        # Create an event to wait for response
        response_event = asyncio.Event()
        response_value: HumanInputResponse | None = None

        def on_response(resp: HumanInputResponse):
            nonlocal response_value
            response_value = resp
            response_event.set()

        self._response_callbacks[request.id] = on_response

        try:
            # Wait with optional timeout
            if request.timeout_seconds:
                try:
                    await asyncio.wait_for(response_event.wait(), timeout=request.timeout_seconds)
                except TimeoutError:
                    return None
            else:
                await response_event.wait()

            return response_value

        finally:
            # Cleanup
            self._pending_requests.pop(request.id, None)
            self._response_callbacks.pop(request.id, None)

    def _save_request_to_db(self, request: HumanInputRequest) -> None:
        """Save pending request to database for persistence."""
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO tool_invocations (
                    context_id, tool_name, arguments, status, blocked_reason
                )
                VALUES (NULL, 'hitl_request', ?, 'pending', ?)
                """,
                (
                    json.dumps(
                        {
                            "request_id": request.id,
                            "execution_id": request.execution_id,
                            "node_name": request.node_name,
                            "input_type": request.input_type.value,
                            "prompt": request.prompt,
                            "options": request.options,
                            "default": request.default,
                        }
                    ),
                    request.prompt,
                ),
            )

    def submit_response(self, request_id: str, value: Any, responder: str | None = None) -> bool:
        """
        Submit a response to a pending input request.

        Args:
            request_id: The request ID to respond to
            value: The response value
            responder: Optional identifier of who responded

        Returns:
            True if response was accepted
        """
        if request_id not in self._pending_requests:
            return False

        request = self._pending_requests[request_id]
        response = HumanInputResponse(request_id=request_id, value=value, responder=responder)

        request.responded_at = response.responded_at
        request.response = value

        # Trigger callback
        callback = self._response_callbacks.get(request_id)
        if callback:
            callback(response)

        return True

    def get_pending_requests(self, execution_id: int | None = None) -> list[HumanInputRequest]:
        """
        Get all pending input requests.

        Args:
            execution_id: Optional filter by execution ID

        Returns:
            List of pending requests
        """
        requests = list(self._pending_requests.values())
        if execution_id is not None:
            requests = [r for r in requests if r.execution_id == execution_id]
        return requests

    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending input request.

        Args:
            request_id: The request ID to cancel

        Returns:
            True if request was cancelled
        """
        if request_id not in self._pending_requests:
            return False

        # Trigger callback with None to unblock
        callback = self._response_callbacks.get(request_id)
        if callback:
            callback(HumanInputResponse(request_id=request_id, value=None))

        return True


class BreakpointManager:
    """
    Manager for workflow breakpoints.

    Breakpoints pause workflow execution at specific nodes
    and require human intervention to continue.
    """

    def __init__(self, hitl_handler: HumanInputHandler):
        self.hitl = hitl_handler
        self._breakpoints: set[str] = set()
        self._one_shot_breakpoints: set[str] = set()

    def add_breakpoint(self, node_name: str, one_shot: bool = False) -> None:
        """
        Add a breakpoint at a node.

        Args:
            node_name: Node name to break at
            one_shot: If True, remove after first hit
        """
        if one_shot:
            self._one_shot_breakpoints.add(node_name)
        else:
            self._breakpoints.add(node_name)

    def remove_breakpoint(self, node_name: str) -> bool:
        """Remove a breakpoint."""
        removed = False
        if node_name in self._breakpoints:
            self._breakpoints.discard(node_name)
            removed = True
        if node_name in self._one_shot_breakpoints:
            self._one_shot_breakpoints.discard(node_name)
            removed = True
        return removed

    def clear_breakpoints(self) -> None:
        """Remove all breakpoints."""
        self._breakpoints.clear()
        self._one_shot_breakpoints.clear()

    def is_breakpoint(self, node_name: str) -> bool:
        """Check if a node has a breakpoint."""
        return node_name in self._breakpoints or node_name in self._one_shot_breakpoints

    async def handle_breakpoint(self, execution_id: int, node_name: str) -> bool:
        """
        Handle a breakpoint hit.

        Args:
            execution_id: Workflow execution ID
            node_name: Node that hit the breakpoint

        Returns:
            True if should continue, False if should abort
        """
        # Remove one-shot breakpoint
        if node_name in self._one_shot_breakpoints:
            self._one_shot_breakpoints.discard(node_name)

        # Request confirmation to continue
        return await self.hitl.request_confirmation(
            execution_id=execution_id,
            node_name=node_name,
            prompt=f"Breakpoint hit at '{node_name}'. Continue execution?",
        )

    def list_breakpoints(self) -> list[dict]:
        """List all active breakpoints."""
        result = []
        for node in self._breakpoints:
            result.append({"node": node, "type": "persistent"})
        for node in self._one_shot_breakpoints:
            result.append({"node": node, "type": "one_shot"})
        return result
