"""Verification hooks for validation subagents.

This module provides hooks for verification subagents that validate
agent work before it is considered complete. Verification hooks can:

- Validate file changes before commit
- Run automated tests on modifications
- Check code quality and style
- Verify configuration syntax
- Ensure security compliance

Usage:
    hook = VerificationHook(
        name="test_runner",
        verification_fn=run_tests,
    )
    manager.register(hook)
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from harness.hooks.base import (
    HookContext,
    HookPriority,
    SubagentStopHook,
)

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of a verification check."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class VerificationResult:
    """Result of a verification check.

    Attributes:
        name: Name of the verification
        status: Verification status
        message: Human-readable message
        details: Additional details or output
        duration_ms: Time taken in milliseconds
        timestamp: When verification completed
        errors: List of error messages if failed
        warnings: List of warning messages
        metadata: Additional metadata
    """

    name: str
    status: VerificationStatus
    message: str = ""
    details: str | None = None
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationResult":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            status=VerificationStatus(data["status"]),
            message=data.get("message", ""),
            details=data.get("details"),
            duration_ms=data.get("duration_ms", 0),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.utcnow(),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            metadata=data.get("metadata", {}),
        )


# Type alias for verification functions
VerificationFn = Callable[
    [str, dict[str, Any], HookContext],
    VerificationResult,
]


class VerificationHook(SubagentStopHook):
    """Hook that runs verification checks when a subagent completes.

    The verification hook runs one or more verification checks on the
    work done by a subagent. If any verification fails, it can trigger
    a retry or escalation.

    Attributes:
        verification_fn: Function that performs the verification
        required: Whether verification must pass for agent to succeed
        retry_on_failure: Whether to retry the agent on failure
        max_retries: Maximum retry attempts
        notify_on_failure: Whether to send notifications on failure
    """

    def __init__(
        self,
        name: str,
        verification_fn: VerificationFn,
        priority: HookPriority = HookPriority.HIGH,
        required: bool = True,
        retry_on_failure: bool = False,
        max_retries: int = 2,
        notify_on_failure: bool = True,
        skip_on_error: bool = False,
    ):
        """Initialize verification hook.

        Args:
            name: Unique name for this verification
            verification_fn: Function to run verification
            priority: Hook priority (default HIGH)
            required: Whether verification must pass
            retry_on_failure: Whether to retry on failure
            max_retries: Maximum retries if retry enabled
            notify_on_failure: Send notification on failure
            skip_on_error: Skip verification on agent error
        """
        super().__init__(name=name, priority=priority)
        self.verification_fn = verification_fn
        self.required = required
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        self.notify_on_failure = notify_on_failure
        self.skip_on_error = skip_on_error

        # Track verification history
        self._history: list[VerificationResult] = []
        self._retry_counts: dict[str, int] = {}

    async def on_subagent_stop(
        self,
        agent_id: str,
        result: dict[str, Any],
        context: HookContext,
    ) -> None:
        """Run verification when subagent completes.

        Args:
            agent_id: ID of the completed subagent
            result: Result dict with status, output, errors
            context: Hook context
        """
        # Check if we should skip verification
        agent_status = result.get("status", "unknown")
        if self.skip_on_error and agent_status in ("failed", "error"):
            logger.info(f"Skipping verification {self.name} for failed agent {agent_id}")
            return

        # Run verification
        import time

        start_time = time.time()

        try:
            verification_result = await self._run_verification(agent_id, result, context)
        except Exception as e:
            logger.error(f"Verification {self.name} error: {e}")
            verification_result = VerificationResult(
                name=self.name,
                status=VerificationStatus.ERROR,
                message=f"Verification error: {e}",
                errors=[str(e)],
            )

        # Record duration
        verification_result.duration_ms = int((time.time() - start_time) * 1000)

        # Store in history
        self._history.append(verification_result)

        # Handle result
        if verification_result.status == VerificationStatus.FAILED:
            await self._handle_failure(agent_id, verification_result, context)
        elif verification_result.status == VerificationStatus.PASSED:
            logger.info(f"Verification {self.name} passed for agent {agent_id}")

        # Add verification result to context metadata
        context.metadata.setdefault("verifications", []).append(verification_result.to_dict())

    async def _run_verification(
        self,
        agent_id: str,
        result: dict[str, Any],
        context: HookContext,
    ) -> VerificationResult:
        """Run the verification function.

        Args:
            agent_id: Agent ID
            result: Agent result
            context: Hook context

        Returns:
            VerificationResult from the check
        """
        verification_result = self.verification_fn(agent_id, result, context)

        # Handle async verification functions
        if asyncio.iscoroutine(verification_result):
            verification_result = await verification_result

        return verification_result

    async def _handle_failure(
        self,
        agent_id: str,
        verification_result: VerificationResult,
        context: HookContext,
    ) -> None:
        """Handle a failed verification.

        Args:
            agent_id: Agent ID
            verification_result: Failed result
            context: Hook context
        """
        logger.warning(
            f"Verification {self.name} failed for agent {agent_id}: {verification_result.message}"
        )

        # Track retries
        retry_count = self._retry_counts.get(agent_id, 0)

        if self.retry_on_failure and retry_count < self.max_retries:
            self._retry_counts[agent_id] = retry_count + 1
            context.metadata["retry_requested"] = True
            context.metadata["retry_reason"] = (
                f"Verification {self.name} failed: {verification_result.message}"
            )
            logger.info(
                f"Requesting retry {retry_count + 1}/{self.max_retries} for agent {agent_id}"
            )

        if self.notify_on_failure:
            context.metadata["notify_failure"] = True
            context.metadata["failure_details"] = verification_result.to_dict()

    def get_history(
        self,
        limit: int = 100,
        status: VerificationStatus | None = None,
    ) -> list[VerificationResult]:
        """Get verification history.

        Args:
            limit: Maximum results to return
            status: Filter by status

        Returns:
            List of verification results
        """
        results = self._history
        if status:
            results = [r for r in results if r.status == status]
        return results[-limit:]

    def get_pass_rate(self) -> float:
        """Get the pass rate for this verification.

        Returns:
            Pass rate as a percentage (0-100)
        """
        if not self._history:
            return 0.0

        passed = sum(1 for r in self._history if r.status == VerificationStatus.PASSED)
        return (passed / len(self._history)) * 100


# Pre-built verification functions
def create_test_verification(
    test_command: str = "pytest",
    working_dir: Path | None = None,
) -> VerificationFn:
    """Create a verification function that runs tests.

    Args:
        test_command: Command to run tests
        working_dir: Working directory for tests

    Returns:
        VerificationFn that runs the tests
    """

    async def verify_tests(
        agent_id: str,
        result: dict[str, Any],
        context: HookContext,
    ) -> VerificationResult:
        import subprocess

        try:
            cwd = working_dir or context.metadata.get("working_dir")
            proc = subprocess.run(
                test_command.split(),
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if proc.returncode == 0:
                return VerificationResult(
                    name="tests",
                    status=VerificationStatus.PASSED,
                    message="All tests passed",
                    details=proc.stdout[-5000:] if proc.stdout else None,
                )
            else:
                return VerificationResult(
                    name="tests",
                    status=VerificationStatus.FAILED,
                    message=f"Tests failed with return code {proc.returncode}",
                    details=proc.stdout[-5000:] if proc.stdout else None,
                    errors=[proc.stderr[-2000:]] if proc.stderr else [],
                )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                name="tests",
                status=VerificationStatus.ERROR,
                message="Test execution timed out",
                errors=["Timeout after 300 seconds"],
            )
        except Exception as e:
            return VerificationResult(
                name="tests",
                status=VerificationStatus.ERROR,
                message=f"Test execution error: {e}",
                errors=[str(e)],
            )

    return verify_tests


def create_lint_verification(
    lint_command: str = "ruff check",
    working_dir: Path | None = None,
) -> VerificationFn:
    """Create a verification function that runs linting.

    Args:
        lint_command: Command to run linting
        working_dir: Working directory

    Returns:
        VerificationFn that runs linting
    """

    async def verify_lint(
        agent_id: str,
        result: dict[str, Any],
        context: HookContext,
    ) -> VerificationResult:
        import subprocess

        try:
            cwd = working_dir or context.metadata.get("working_dir")
            proc = subprocess.run(
                lint_command.split(),
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if proc.returncode == 0:
                return VerificationResult(
                    name="lint",
                    status=VerificationStatus.PASSED,
                    message="Linting passed",
                )
            else:
                # Extract warnings from output
                warnings = []
                errors = []
                for line in proc.stdout.split("\n"):
                    if line.strip():
                        if "error" in line.lower():
                            errors.append(line)
                        else:
                            warnings.append(line)

                return VerificationResult(
                    name="lint",
                    status=VerificationStatus.FAILED,
                    message=f"Linting failed with {len(errors)} errors",
                    details=proc.stdout[-3000:] if proc.stdout else None,
                    errors=errors[:20],
                    warnings=warnings[:20],
                )

        except Exception as e:
            return VerificationResult(
                name="lint",
                status=VerificationStatus.ERROR,
                message=f"Lint execution error: {e}",
                errors=[str(e)],
            )

    return verify_lint


def create_file_exists_verification(
    required_files: list[str],
) -> VerificationFn:
    """Create a verification that checks required files exist.

    Args:
        required_files: List of file paths that must exist

    Returns:
        VerificationFn that checks file existence
    """

    async def verify_files(
        agent_id: str,
        result: dict[str, Any],
        context: HookContext,
    ) -> VerificationResult:
        from pathlib import Path

        working_dir = context.metadata.get("working_dir", ".")
        missing = []

        for file_path in required_files:
            full_path = Path(working_dir) / file_path
            if not full_path.exists():
                missing.append(file_path)

        if not missing:
            return VerificationResult(
                name="required_files",
                status=VerificationStatus.PASSED,
                message=f"All {len(required_files)} required files exist",
            )
        else:
            return VerificationResult(
                name="required_files",
                status=VerificationStatus.FAILED,
                message=f"Missing {len(missing)} required files",
                errors=[f"Missing: {f}" for f in missing],
            )

    return verify_files
