"""Bootstrap package for self-installing harness setup.

This package provides a complete bootstrap system for setting up the
DAG harness from scratch within MCP client.

Usage:
    harness bootstrap              # Full interactive setup
    harness bootstrap --check-only # Verify current state
    harness credentials            # Discover and validate credentials
    harness credentials --prompt   # Interactive credential setup
    harness upgrade                # Check for and install updates
    harness upgrade --check        # Check only, don't install

Components:
    - BootstrapRunner: Main orchestrator
    - BootstrapWizard: Interactive setup wizard
    - CredentialDiscovery: Credential detection and validation
    - PathResolver: Path resolution and validation
    - SelfTester: Post-install verification tests
    - PlatformInfo: Platform detection
    - Installer: Installation logic
    - KeyDiscovery: Parallel key discovery
    - CredentialValidator: Async validation
    - CredentialPrompts: Interactive prompts
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from harness.bootstrap.credentials import (
    CredentialCheckResult,
    CredentialDiscovery,
    CredentialStatus,
)
from harness.bootstrap.discovery import (
    KEY_REGISTRY,
    KeyConfig,
    KeyDiscovery,
    KeyInfo,
    KeyStatus,
)
from harness.bootstrap.installer import (
    Installer,
    InstallMethod,
    InstallResult,
    InstallStatus,
    install,
)
from harness.bootstrap.paths import (
    PathCheckResult,
    PathResolver,
    PathStatus,
)

# New modules
from harness.bootstrap.platform import (
    OS,
    Architecture,
    PackageManager,
    PlatformInfo,
    check_binary_compatibility,
    detect_platform,
    find_python,
)
from harness.bootstrap.prompts import (
    CredentialPrompts,
    interactive_setup,
)
from harness.bootstrap.selftest import (
    SelfTester,
    SelfTestResult,
    TestStatus,
)
from harness.bootstrap.upgrade import (
    UpgradeResult,
    UpgradeStatus,
    VersionInfo,
    check_for_upgrade,
    upgrade,
)
from harness.bootstrap.validation import (
    CredentialValidator,
    ValidationResult,
    ValidationStatus,
    validate_credentials,
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
    credentials: CredentialCheckResult | None = None
    paths: PathCheckResult | None = None
    selftests: SelfTestResult | None = None
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
        project_root: Path | None = None,
        console: Console | None = None,
        interactive: bool = True,
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
        self.wizard = BootstrapWizard(self.project_root, self.console, self.interactive)

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
            warnings=wizard_result.state.warnings,
        )

    def check_prerequisites(self) -> tuple[bool, list[str]]:
        """Quick check of prerequisites without full bootstrap.

        Returns:
            Tuple of (all_ok, list_of_issues)
        """
        issues = []

        # Check Python version
        import sys

        # Check for required packages
        try:
            import langgraph
            import rich
            import typer
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
            issues.append(
                "uv not available (install with: curl -LsSf https://astral.sh/uv/install.sh | sh)"
            )

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
            "HARNESS_DB_PATH", str(self.project_root / "harness" / "harness.db")
        )
        if not Path(db_path).exists():
            return False

        return True


# Convenience function for CLI
def bootstrap(check_only: bool = False, project_root: Path | None = None) -> BootstrapResult:
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
    # Credential discovery (legacy)
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
    # Platform detection
    "PlatformInfo",
    "OS",
    "Architecture",
    "PackageManager",
    "detect_platform",
    "find_python",
    "check_binary_compatibility",
    # Installer
    "Installer",
    "InstallResult",
    "InstallMethod",
    "InstallStatus",
    "install",
    # Upgrade
    "VersionInfo",
    "UpgradeResult",
    "UpgradeStatus",
    "check_for_upgrade",
    "upgrade",
    # Key discovery (new)
    "KeyDiscovery",
    "KeyInfo",
    "KeyConfig",
    "KeyStatus",
    "KEY_REGISTRY",
    # Validation
    "CredentialValidator",
    "ValidationResult",
    "ValidationStatus",
    "validate_credentials",
    # Prompts
    "CredentialPrompts",
    "interactive_setup",
    # Convenience function
    "bootstrap",
]
