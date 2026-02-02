"""Rich UI prompts for interactive credential setup.

This module provides:
- Interactive prompts for missing credentials
- Secure password input
- Progress indicators
- Confirmation dialogs
"""

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from harness.bootstrap.discovery import KEY_REGISTRY, KeyConfig, KeyInfo, KeyStatus
from harness.bootstrap.validation import ValidationResult


class CredentialPrompts:
    """Interactive prompts for credential configuration."""

    def __init__(self, console: Console | None = None):
        """Initialize prompts.

        Args:
            console: Rich console for output
        """
        self.console = console or Console()

    def prompt_for_credential(
        self,
        key_name: str,
        show_help: bool = True,
    ) -> str | None:
        """Prompt user to enter a credential.

        Args:
            key_name: Name of the credential
            show_help: Show help text about the credential

        Returns:
            Entered value or None if skipped
        """
        config = KEY_REGISTRY.get(key_name)
        if not config:
            return Prompt.ask(f"Enter {key_name}")

        if show_help:
            self._show_credential_help(config)

        # Determine if this should be masked input
        is_sensitive = "TOKEN" in key_name or "KEY" in key_name or "PASSWORD" in key_name

        required_text = "[red](required)[/red]" if config.required else "[dim](optional)[/dim]"

        value = Prompt.ask(
            f"Enter {key_name} {required_text}",
            password=is_sensitive,
            default="" if not config.required else ...,
        )

        if not value and not config.required:
            return None

        return value

    def _show_credential_help(self, config: KeyConfig) -> None:
        """Show help text for a credential.

        Args:
            config: Key configuration
        """
        help_lines = [
            f"[bold]{config.name}[/bold]",
            f"  {config.description}",
        ]

        if config.env_vars:
            vars_str = ", ".join(config.env_vars)
            help_lines.append(f"  [dim]Environment variables: {vars_str}[/dim]")

        if config.prefix:
            help_lines.append(f"  [dim]Expected prefix: {config.prefix}[/dim]")

        self.console.print("\n".join(help_lines))

    def prompt_for_missing(
        self,
        keys: dict[str, KeyInfo],
        required_only: bool = False,
    ) -> dict[str, str]:
        """Prompt for all missing credentials.

        Args:
            keys: Dict of KeyInfo from discovery
            required_only: Only prompt for required credentials

        Returns:
            Dict of key names to entered values
        """
        entered = {}

        for name, info in keys.items():
            if info.status != KeyStatus.FOUND:
                config = KEY_REGISTRY.get(name)
                if not config:
                    continue

                if required_only and not config.required:
                    continue

                self.console.print()
                value = self.prompt_for_credential(name)
                if value:
                    entered[name] = value

        return entered

    def confirm_save(
        self,
        credentials: dict[str, str],
        target_file: Path,
    ) -> bool:
        """Confirm saving credentials to .env file.

        Args:
            credentials: Credentials to save
            target_file: Target .env file path

        Returns:
            True if user confirms
        """
        if not credentials:
            return False

        self.console.print()
        self.console.print(
            Panel(
                f"Save {len(credentials)} credential(s) to [cyan]{target_file}[/cyan]?",
                title="[bold]Confirm Save[/bold]",
            )
        )

        # Show what will be saved (masked)
        table = Table(show_header=False, box=None)
        for name, value in credentials.items():
            masked = self._mask_value(value)
            table.add_row(f"  {name}", f"= {masked}")

        self.console.print(table)
        self.console.print()

        return Confirm.ask("Proceed?", default=True)

    def save_to_env(
        self,
        credentials: dict[str, str],
        env_file: Path,
    ) -> bool:
        """Save credentials to .env file.

        Args:
            credentials: Credentials to save
            env_file: Target .env file

        Returns:
            True if saved successfully
        """
        try:
            # Read existing content
            existing_lines = []
            existing_keys = set()

            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        existing_lines.append(line)
                        stripped = line.strip()
                        if stripped and not stripped.startswith("#") and "=" in stripped:
                            key = stripped.split("=", 1)[0].strip()
                            existing_keys.add(key)

            # Prepare new content
            new_lines = []
            updated_keys = set()

            for line in existing_lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0].strip()
                    if key in credentials:
                        # Update existing key
                        new_lines.append(f'{key}="{credentials[key]}"\n')
                        updated_keys.add(key)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            # Add new keys
            for key, value in credentials.items():
                if key not in updated_keys:
                    if new_lines and not new_lines[-1].endswith("\n"):
                        new_lines.append("\n")
                    new_lines.append(f'{key}="{value}"\n')

            # Write back
            env_file.parent.mkdir(parents=True, exist_ok=True)
            with open(env_file, "w") as f:
                f.writelines(new_lines)

            self.console.print(f"[green]Saved to {env_file}[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Error saving to {env_file}: {e}[/red]")
            return False

    def display_discovery_results(
        self,
        keys: dict[str, KeyInfo],
        show_values: bool = False,
    ) -> None:
        """Display credential discovery results.

        Args:
            keys: Dict of KeyInfo from discovery
            show_values: Show masked values
        """
        table = Table(title="Credential Discovery Results")
        table.add_column("Credential", style="cyan")
        table.add_column("Status")
        table.add_column("Source")
        if show_values:
            table.add_column("Value")

        for name, info in keys.items():
            config = KEY_REGISTRY.get(name)
            required = config.required if config else False

            # Status with color
            if info.status == KeyStatus.FOUND:
                status = "[green]Found[/green]"
            elif info.status == KeyStatus.INVALID:
                status = "[red]Invalid[/red]"
            elif required:
                status = "[red]Missing (required)[/red]"
            else:
                status = "[yellow]Missing[/yellow]"

            # Source
            source = info.source or "-"

            if show_values:
                value = info.masked_value or "-"
                table.add_row(name, status, source, value)
            else:
                table.add_row(name, status, source)

        self.console.print(table)

    def display_validation_results(
        self,
        results: dict[str, ValidationResult],
    ) -> None:
        """Display credential validation results.

        Args:
            results: Dict of ValidationResult from validation
        """
        from harness.bootstrap.validation import ValidationStatus

        table = Table(title="Credential Validation Results")
        table.add_column("Credential", style="cyan")
        table.add_column("Status")
        table.add_column("Message")
        table.add_column("Time")

        for name, result in results.items():
            # Status with color
            if result.status == ValidationStatus.VALID:
                status = "[green]Valid[/green]"
            elif result.status == ValidationStatus.INVALID:
                status = "[red]Invalid[/red]"
            elif result.status == ValidationStatus.TIMEOUT:
                status = "[yellow]Timeout[/yellow]"
            elif result.status == ValidationStatus.SKIPPED:
                status = "[dim]Skipped[/dim]"
            else:
                status = "[yellow]Error[/yellow]"

            # Duration
            time_str = f"{result.duration_ms:.0f}ms" if result.duration_ms else "-"

            table.add_row(name, status, result.message, time_str)

        self.console.print(table)

    def show_progress(self, message: str):
        """Create a progress context for long operations.

        Args:
            message: Progress message

        Returns:
            Progress context manager
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        )

    def _mask_value(self, value: str) -> str:
        """Mask a credential value for display.

        Args:
            value: Value to mask

        Returns:
            Masked value
        """
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def interactive_setup(
    project_root: Path | None = None,
    required_only: bool = False,
    save: bool = True,
) -> dict[str, str]:
    """Run interactive credential setup.

    Args:
        project_root: Project root directory
        required_only: Only prompt for required credentials
        save: Save credentials to .env file

    Returns:
        Dict of configured credentials
    """
    from harness.bootstrap.discovery import KeyDiscovery

    console = Console()
    prompts = CredentialPrompts(console)

    # Discover existing credentials
    console.print("\n[bold]Discovering existing credentials...[/bold]")
    discovery = KeyDiscovery(project_root)
    keys = discovery.discover_all()

    # Display results
    prompts.display_discovery_results(keys, show_values=True)

    # Check for missing required
    missing_required = [
        name
        for name, info in keys.items()
        if info.status != KeyStatus.FOUND and KEY_REGISTRY.get(name, KeyConfig("", "")).required
    ]

    if not missing_required and required_only:
        console.print("\n[green]All required credentials found![/green]")
        return {}

    # Prompt for missing
    console.print("\n[bold]Configure missing credentials:[/bold]")
    entered = prompts.prompt_for_missing(keys, required_only=required_only)

    if not entered:
        console.print("\n[dim]No credentials configured.[/dim]")
        return {}

    # Save if requested
    if save:
        env_file = Path(project_root or os.getcwd()) / ".env"
        if prompts.confirm_save(entered, env_file):
            prompts.save_to_env(entered, env_file)

    return entered
