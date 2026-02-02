"""
E2B Sandbox Integration for isolated execution of destructive operations.

Provides:
- SandboxedExecution: Main class for running code in isolated E2B sandboxes
- ExecutionResult: Result dataclass containing execution output and metadata
- SandboxTemplate: Pre-configured sandbox templates for Python/Bash
- LocalFallbackExecutor: Fallback when E2B is not available

Usage:
    from harness.sandbox import SandboxedExecution, ExecutionResult, e2b_available

    # Check if E2B is available
    if e2b_available():
        executor = SandboxedExecution(template="python", timeout=300)
        result = await executor.run("print('Hello, isolated world!')")
    else:
        # Automatic fallback to local execution
        executor = SandboxedExecution()
        result = await executor.run("print('Running locally')")

Environment:
    E2B_API_KEY: Required for E2B sandbox execution
    E2B_TEMPLATE: Optional default template override
"""

from __future__ import annotations

import logging

from harness.sandbox.execution import (
    ExecutionResult,
    ResourceLimits,
    SandboxedExecution,
)
from harness.sandbox.fallback import LocalFallbackExecutor
from harness.sandbox.templates import (
    BashTemplate,
    PythonTemplate,
    SandboxTemplate,
    get_template,
)

# Check if e2b is available
try:
    import e2b  # noqa: F401

    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False


def e2b_available() -> bool:
    """
    Check if E2B package is installed and available.

    Returns:
        True if e2b package is installed, False otherwise
    """
    return E2B_AVAILABLE


__all__ = [
    # Core classes
    "SandboxedExecution",
    "ExecutionResult",
    "ResourceLimits",
    # Templates
    "SandboxTemplate",
    "PythonTemplate",
    "BashTemplate",
    "get_template",
    # Fallback
    "LocalFallbackExecutor",
    # Availability check
    "e2b_available",
    "E2B_AVAILABLE",
]

# Set up module logger
logger = logging.getLogger(__name__)

if not E2B_AVAILABLE:
    logger.info(
        "E2B package not installed. Sandbox execution will use local fallback. "
        "Install with: pip install 'dag-harness[sandbox]'"
    )
