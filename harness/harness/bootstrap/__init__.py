"""Bootstrap package for self-installing harness setup.

This package provides a complete bootstrap system for setting up the
DAG harness from scratch within MCP client.

Usage:
    harness bootstrap              # Full interactive setup
    harness bootstrap --check-only # Verify current state

Components:
    - BootstrapRunner: Main orchestrator
    - BootstrapWizard: Interactive setup wizard
    - CredentialDiscovery: Credential detection and validation
    - PathResolver: Path resolution and validation
    - SelfTester: Post-install verification tests
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from harness.bootstrap.credentials import (
    CredentialDiscovery,
    CredentialCheckResult,
    CredentialStatus,
)
from harness.bootstrap.paths import (
    PathResolver,
    PathCheckResult,
    PathStatus,
)
from harness.bootstrap.selftest import (
    SelfTester,
    SelfTestResult,
    TestStatus,
)
from harness.bootstrap.wizard import (
    BootstrapWizard,
    WizardResult,
    WizardStep,
)


@dataclass
class BootstrapResult:
    """Result of running the bootstrap process."""
    success: bool
    message: str
    credentials: Optional[CredentialCheckResult] = None
    paths: Optional[PathCheckResult] = None
    selftests: Optional[SelfTestResult] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BootstrapRunner:
    """Main orchestrator for the bootstrap process.

    This class coordinates all bootstrap components:
    - Environment detection
    - Credential discovery
    - Path resolution
    - Database initialization
    - MCP client integration
    - Self-tests

    Example:
        runner = BootstrapRunner()
        result = runner.run()
        if result.success:
            print("Bootstrap complete!")
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        console: Optional[Console] = None,
        interactive: bool = True
    ):
        """Initialize the bootstrap runner.

        Args:
            project_root: Root directory of the project
            console: Rich console for output
            interactive: Whether to prompt for user input
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.console = console or Console()
        self.interactive = interactive

        # Initialize components
        self.credential_discovery = CredentialDiscovery(self.project_root)
        self.path_resolver = PathResolver(self.project_root)
        self.self_tester = SelfTester(self.project_root)
        self.wizard = BootstrapWizard(
            self.project_root,
            self.console,
            self.interactive
        )

    def run(self, check_only: bool = False) -> BootstrapResult:
        """Run the bootstrap process.

        Args:
            check_only: Only check current state without making changes

        Returns:
            BootstrapResult with status and details
        """
        wizard_result = self.wizard.run(check_only=check_only)

        return BootstrapResult(
            success=wizard_result.success,
            message=wizard_result.message,
            credentials=None,  # Could populate from wizard state
            paths=None,
            selftests=None,
            errors=wizard_result.state.errors,
            warnings=wizard_result.state.warnings
        )

    def check_prerequisites(self) -> tuple[bool, list[str]]:
        """Quick check of prerequisites without full bootstrap.

        Returns:
            Tuple of (all_ok, list_of_issues)
        """
        issues = []

        # Check Python version
        import sys
        if sys.version_info < (3, 10):
            issues.append(f"Python 3.10+ required (found {sys.version_info.major}.{sys.version_info.minor})")

        # Check for required packages
        try:
            import typer
            import rich
            import langgraph
        except ImportError as e:
            issues.append(f"Missing required package: {e.name}")

        # Check git is available
        import subprocess
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            issues.append("git not available")

        # Check uv is available
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            issues.append("uv not available (install with: curl -LsSf https://astral.sh/uv/install.sh | sh)")

        return len(issues) == 0, issues

    def quick_check(self) -> bool:
        """Perform a quick check if bootstrap is needed.

        Returns:
            True if harness appears to be properly set up
        """
        # Check for .claude/settings.json with MCP config
        settings_path = self.project_root / ".claude" / "settings.json"
        if not settings_path.exists():
            return False

        try:
            import json
            with open(settings_path) as f:
                settings = json.load(f)
            if "mcpServers" not in settings or "dag-harness" not in settings["mcpServers"]:
                return False
        except Exception:
            return False

        # Check for database
        db_path = os.environ.get(
            "HARNESS_DB_PATH",
            str(self.project_root / "harness" / "harness.db")
        )
        if not Path(db_path).exists():
            return False

        return True


# Convenience function for CLI
def bootstrap(
    check_only: bool = False,
    project_root: Optional[Path] = None
) -> BootstrapResult:
    """Run the bootstrap process.

    Args:
        check_only: Only check current state
        project_root: Override project root

    Returns:
        BootstrapResult
    """
    runner = BootstrapRunner(project_root=project_root)
    return runner.run(check_only=check_only)


__all__ = [
    # Main classes
    "BootstrapRunner",
    "BootstrapResult",
    "BootstrapWizard",
    "WizardResult",
    "WizardStep",
    # Credential discovery
    "CredentialDiscovery",
    "CredentialCheckResult",
    "CredentialStatus",
    # Path resolution
    "PathResolver",
    "PathCheckResult",
    "PathStatus",
    # Self-tests
    "SelfTester",
    "SelfTestResult",
    "TestStatus",
    # Convenience function
    "bootstrap",
]
