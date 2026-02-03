"""Self-installation system for MCP client integration.

This module provides commands to install, uninstall, check, and upgrade
the harness integration with MCP client.

Assets are bundled inside the wheel via harness.assets.claude and deployed
using importlib.resources -- no adjacent checkout or hardcoded paths needed.

Usage:
    harness install run           # Install harness into MCP client
    harness install check         # Verify installation status
    harness install uninstall     # Remove harness from MCP client
    harness install upgrade       # Upgrade existing installation
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from rich.console import Console

console = Console()


class InstallStatus(Enum):
    """Installation status codes."""

    NOT_INSTALLED = "not_installed"
    PARTIAL = "partial"
    COMPLETE = "complete"
    OUTDATED = "outdated"


@dataclass
class ComponentStatus:
    """Status of an individual component."""

    name: str
    installed: bool
    path: Path | None = None
    version: str | None = None
    issues: list[str] = field(default_factory=list)


@dataclass
class InstallResult:
    """Result of an installation operation."""

    success: bool
    components: list[ComponentStatus] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class MCPInstaller:
    """Self-installation manager for MCP client integration.

    This class handles:
    - Creating .claude directory structure
    - Deploying MCP server configuration
    - Installing hook scripts (from bundled assets)
    - Deploying skill definitions (from bundled assets)
    - Verifying installation status

    Assets are loaded from the wheel via harness.assets.loader,
    not from filesystem paths relative to __file__.
    """

    INSTALL_VERSION = "2.0.0"

    REQUIRED_HOOKS = [
        "validate-box-up-env.sh",
        "notify-box-up-status.sh",
        "rate-limiter-hook.py",
        "pre-commit-role.sh",
    ]

    REQUIRED_SKILLS = [
        "box-up-role",
        "hotl",
        "observability",
    ]

    def __init__(self, project_root: Path | None = None):
        """Initialize installer.

        Args:
            project_root: Root directory of the project. Defaults to cwd.
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.claude_dir = self.project_root / ".claude"
        self.settings_path = self.claude_dir / "settings.json"
        self.hooks_dir = self.claude_dir / "hooks"
        self.skills_dir = self.claude_dir / "skills"

    def install(
        self,
        force: bool = False,
        skip_hooks: bool = False,
        skip_skills: bool = False,
        skip_mcp: bool = False,
    ) -> InstallResult:
        """Install harness into MCP client.

        Args:
            force: Overwrite existing files without prompting
            skip_hooks: Don't install hook scripts
            skip_skills: Don't install skill definitions
            skip_mcp: Don't configure MCP server

        Returns:
            InstallResult with status of each component
        """
        result = InstallResult(success=True)

        # Create directory structure
        try:
            self._create_directories()
            result.components.append(
                ComponentStatus(name="directories", installed=True, path=self.claude_dir)
            )
        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to create directories: {e}")
            return result

        # Install MCP server configuration + settings
        if not skip_mcp:
            mcp_result = self._install_mcp_config(force)
            result.components.append(mcp_result)
            if not mcp_result.installed:
                result.success = False

        # Install hooks from bundled assets
        if not skip_hooks:
            hooks_result = self._install_hooks(force)
            result.components.extend(hooks_result)
            for hook in hooks_result:
                if not hook.installed:
                    result.warnings.append(f"Hook not installed: {hook.name}")

        # Install skills from bundled assets
        if not skip_skills:
            skills_result = self._install_skills(force)
            result.components.extend(skills_result)
            for skill in skills_result:
                if not skill.installed:
                    result.warnings.append(f"Skill not installed: {skill.name}")

        # Write installation marker
        self._write_install_marker()

        return result

    def uninstall(self, remove_data: bool = False) -> bool:
        """Remove harness from MCP client.

        Args:
            remove_data: Also remove any data files (databases, logs)

        Returns:
            True if uninstall was successful
        """
        try:
            # Remove hooks
            if self.hooks_dir.exists():
                for hook_file in self.hooks_dir.glob("*.sh"):
                    hook_file.unlink()
                for hook_file in self.hooks_dir.glob("*.py"):
                    hook_file.unlink()

            # Remove skills
            for skill in self.REQUIRED_SKILLS:
                skill_dir = self.skills_dir / skill
                if skill_dir.exists():
                    shutil.rmtree(skill_dir)

            # Remove MCP config from settings.json
            if self.settings_path.exists():
                self._remove_mcp_config()

            # Remove installation marker
            marker_path = self.claude_dir / ".harness-installed"
            if marker_path.exists():
                marker_path.unlink()

            if remove_data:
                # Remove rate limit database
                rate_limit_db = Path.home() / ".claude" / "rate_limits.db"
                if rate_limit_db.exists():
                    rate_limit_db.unlink()

            return True

        except Exception as e:
            console.print(f"[red]Uninstall error: {e}[/red]")
            return False

    def check(self) -> tuple[InstallStatus, list[ComponentStatus]]:
        """Check installation status.

        Returns:
            Tuple of (overall status, list of component statuses)
        """
        components = []

        # Check directories
        dirs_ok = self.claude_dir.exists() and self.hooks_dir.exists() and self.skills_dir.exists()
        components.append(
            ComponentStatus(
                name="directories",
                installed=dirs_ok,
                path=self.claude_dir if dirs_ok else None,
                issues=[] if dirs_ok else ["Missing .claude directory structure"],
            )
        )

        # Check MCP server configuration
        mcp_ok = False
        mcp_issues = []
        if self.settings_path.exists():
            try:
                with open(self.settings_path) as f:
                    settings = json.load(f)
                if "mcpServers" in settings and "dag-harness" in settings["mcpServers"]:
                    mcp_ok = True
                else:
                    mcp_issues.append("MCP server not configured in settings.json")
            except json.JSONDecodeError:
                mcp_issues.append("settings.json is invalid JSON")
        else:
            mcp_issues.append("settings.json not found")

        components.append(
            ComponentStatus(
                name="mcp_server",
                installed=mcp_ok,
                path=self.settings_path if mcp_ok else None,
                issues=mcp_issues,
            )
        )

        # Check hooks
        for hook_name in self.REQUIRED_HOOKS:
            hook_path = self.hooks_dir / hook_name
            hook_ok = hook_path.exists()
            components.append(
                ComponentStatus(
                    name=f"hook:{hook_name}",
                    installed=hook_ok,
                    path=hook_path if hook_ok else None,
                    issues=[] if hook_ok else [f"Hook file missing: {hook_name}"],
                )
            )

        # Check skills
        for skill_name in self.REQUIRED_SKILLS:
            skill_path = self.skills_dir / skill_name
            skill_md = skill_path / "SKILL.md"
            skill_ok = skill_path.exists() and skill_md.exists()
            components.append(
                ComponentStatus(
                    name=f"skill:{skill_name}",
                    installed=skill_ok,
                    path=skill_path if skill_ok else None,
                    issues=[] if skill_ok else [f"Skill missing or incomplete: {skill_name}"],
                )
            )

        # Determine overall status
        installed_count = sum(1 for c in components if c.installed)
        total_count = len(components)

        if installed_count == 0:
            status = InstallStatus.NOT_INSTALLED
        elif installed_count < total_count:
            status = InstallStatus.PARTIAL
        else:
            marker_path = self.claude_dir / ".harness-installed"
            if marker_path.exists():
                try:
                    marker = json.loads(marker_path.read_text())
                    if marker.get("version") != self.INSTALL_VERSION:
                        status = InstallStatus.OUTDATED
                    else:
                        status = InstallStatus.COMPLETE
                except Exception:
                    status = InstallStatus.COMPLETE
            else:
                status = InstallStatus.COMPLETE

        return status, components

    def upgrade(self) -> InstallResult:
        """Upgrade existing installation.

        Re-installs hooks and skills from bundled assets while preserving
        custom permissions in settings.json.

        Returns:
            InstallResult with status of upgrade
        """
        # Back up current settings
        backup_settings = None
        if self.settings_path.exists():
            try:
                with open(self.settings_path) as f:
                    backup_settings = json.load(f)
            except Exception:
                pass

        # Re-install with force
        result = self.install(force=True)

        # Restore custom settings
        if backup_settings and result.success:
            try:
                with open(self.settings_path) as f:
                    new_settings = json.load(f)

                # Preserve user's permissions
                if "permissions" in backup_settings:
                    new_settings["permissions"] = backup_settings["permissions"]

                with open(self.settings_path, "w") as f:
                    json.dump(new_settings, f, indent=2)
                    f.write("\n")

            except Exception as e:
                result.warnings.append(f"Could not restore custom settings: {e}")

        return result

    def _create_directories(self):
        """Create required directory structure."""
        self.claude_dir.mkdir(exist_ok=True)
        self.hooks_dir.mkdir(exist_ok=True)
        self.skills_dir.mkdir(exist_ok=True)

        for skill in self.REQUIRED_SKILLS:
            (self.skills_dir / skill).mkdir(exist_ok=True)

    def _install_mcp_config(self, force: bool) -> ComponentStatus:
        """Install MCP server configuration from bundled settings template."""
        from harness.assets import loader

        try:
            if self.settings_path.exists() and not force:
                # Merge MCP config into existing settings
                with open(self.settings_path) as f:
                    settings = json.load(f)
            else:
                # Load from bundled template (strip Jinja2 since we have static defaults)
                try:
                    template_text = loader.read_text("settings.json.j2")
                    # The template is valid JSON as-is (no Jinja2 variables used)
                    settings = json.loads(template_text)
                except (FileNotFoundError, json.JSONDecodeError):
                    settings = {}

            # Ensure MCP server config uses the installed harness command
            if "mcpServers" not in settings:
                settings["mcpServers"] = {}

            settings["mcpServers"]["dag-harness"] = {
                "command": "harness",
                "args": ["mcp-server"],
            }

            # Add default permissions if not present
            if "permissions" not in settings:
                settings["permissions"] = {
                    "allow": [
                        "Bash(npm run *)",
                        "Bash(harness *)",
                        "Bash(glab *)",
                        "Bash(git *)",
                        "Bash(molecule *)",
                        "Bash(pytest *)",
                        "Bash(python scripts/*)",
                        "Bash(./scripts/*)",
                        "Bash(uv run *)",
                    ],
                    "deny": [
                        "Bash(*--force*push*)",
                        "Bash(*reset --hard*)",
                    ],
                }

            # Add hooks configuration if not present
            if "hooks" not in settings:
                settings["hooks"] = {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "${workspaceFolder}/.claude/hooks/validate-box-up-env.sh",
                                }
                            ],
                        },
                    ],
                    "PostToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "${workspaceFolder}/.claude/hooks/notify-box-up-status.sh",
                                }
                            ],
                        },
                    ],
                }

            with open(self.settings_path, "w") as f:
                json.dump(settings, f, indent=2)
                f.write("\n")

            return ComponentStatus(name="mcp_server", installed=True, path=self.settings_path)

        except Exception as e:
            return ComponentStatus(name="mcp_server", installed=False, issues=[str(e)])

    def _install_hooks(self, force: bool) -> list[ComponentStatus]:
        """Install hook scripts from bundled assets."""
        from harness.assets import loader

        results = []

        for hook_name in self.REQUIRED_HOOKS:
            dest = self.hooks_dir / hook_name

            try:
                if dest.exists() and not force:
                    results.append(
                        ComponentStatus(
                            name=f"hook:{hook_name}",
                            installed=True,
                            path=dest,
                            issues=["Already exists (use --force to overwrite)"],
                        )
                    )
                    continue

                # Read from bundled assets
                content = loader.read_text(f"hooks/{hook_name}")
                dest.write_text(content, encoding="utf-8")
                # Make executable
                dest.chmod(dest.stat().st_mode | 0o111)
                results.append(
                    ComponentStatus(name=f"hook:{hook_name}", installed=True, path=dest)
                )

            except FileNotFoundError:
                results.append(
                    ComponentStatus(
                        name=f"hook:{hook_name}",
                        installed=False,
                        issues=[f"Asset not found in bundle: hooks/{hook_name}"],
                    )
                )
            except Exception as e:
                results.append(
                    ComponentStatus(name=f"hook:{hook_name}", installed=False, issues=[str(e)])
                )

        return results

    def _install_skills(self, force: bool) -> list[ComponentStatus]:
        """Install skill definitions from bundled assets."""
        from harness.assets import loader

        results = []

        for skill_name in self.REQUIRED_SKILLS:
            dest = self.skills_dir / skill_name

            try:
                if dest.exists() and not force:
                    if (dest / "SKILL.md").exists():
                        results.append(
                            ComponentStatus(
                                name=f"skill:{skill_name}",
                                installed=True,
                                path=dest,
                                issues=["Already exists (use --force to overwrite)"],
                            )
                        )
                        continue

                # Deploy entire skill directory from bundled assets
                deployed = loader.deploy_directory(
                    f"skills/{skill_name}",
                    dest,
                    executable_extensions=set(),  # Skills are not executable
                )

                if deployed:
                    results.append(
                        ComponentStatus(name=f"skill:{skill_name}", installed=True, path=dest)
                    )
                else:
                    results.append(
                        ComponentStatus(
                            name=f"skill:{skill_name}",
                            installed=False,
                            issues=[f"No assets found for skill: {skill_name}"],
                        )
                    )

            except FileNotFoundError:
                results.append(
                    ComponentStatus(
                        name=f"skill:{skill_name}",
                        installed=False,
                        issues=[f"Asset not found in bundle: skills/{skill_name}"],
                    )
                )
            except Exception as e:
                results.append(
                    ComponentStatus(name=f"skill:{skill_name}", installed=False, issues=[str(e)])
                )

        return results

    def _remove_mcp_config(self):
        """Remove MCP server configuration from settings.json."""
        try:
            with open(self.settings_path) as f:
                settings = json.load(f)

            if "mcpServers" in settings:
                settings["mcpServers"].pop("dag-harness", None)
                settings["mcpServers"].pop("ems-harness", None)

            with open(self.settings_path, "w") as f:
                json.dump(settings, f, indent=2)
                f.write("\n")

        except Exception:
            pass

    def _write_install_marker(self):
        """Write installation marker file."""
        from harness import __version__

        marker = {
            "version": self.INSTALL_VERSION,
            "installed_at": __import__("datetime").datetime.now().isoformat(),
            "harness_version": __version__,
        }

        marker_path = self.claude_dir / ".harness-installed"
        with open(marker_path, "w") as f:
            json.dump(marker, f, indent=2)
            f.write("\n")


def print_install_result(result: InstallResult):
    """Print installation result in a user-friendly format."""
    if result.success:
        console.print("\n[bold green]Installation successful![/bold green]\n")
    else:
        console.print("\n[bold red]Installation failed![/bold red]\n")

    for component in result.components:
        status = "[green]OK[/green]" if component.installed else "[red]MISSING[/red]"
        console.print(f"  {status} {component.name}")
        for issue in component.issues:
            console.print(f"       [dim]{issue}[/dim]")

    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.errors:
            console.print(f"  - {error}")

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  - {warning}")


def print_check_result(status: InstallStatus, components: list[ComponentStatus]):
    """Print installation check result."""
    status_colors = {
        InstallStatus.NOT_INSTALLED: "[red]NOT INSTALLED[/red]",
        InstallStatus.PARTIAL: "[yellow]PARTIAL[/yellow]",
        InstallStatus.COMPLETE: "[green]COMPLETE[/green]",
        InstallStatus.OUTDATED: "[yellow]OUTDATED[/yellow]",
    }

    console.print(f"\n[bold]Installation Status:[/bold] {status_colors[status]}\n")

    for component in components:
        status_str = "[green]OK[/green]" if component.installed else "[red]MISSING[/red]"
        console.print(f"  {status_str} {component.name}")
        if component.path:
            console.print(f"       [dim]{component.path}[/dim]")
        for issue in component.issues:
            console.print(f"       [yellow]{issue}[/yellow]")
