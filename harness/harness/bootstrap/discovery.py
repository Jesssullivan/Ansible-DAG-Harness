"""Parallel key discovery for dag-harness bootstrap.

This module provides:
- KEY_REGISTRY with supported credentials (GITLAB_TOKEN required, HOTL_EMAIL_TO optional)
- Parallel discovery from 9+ locations
- glab CLI config parsing
- macOS Keychain integration
"""

import asyncio
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class KeyStatus(Enum):
    """Status of a discovered key."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    INVALID = "invalid"
    EXPIRED = "expired"


@dataclass
class KeyInfo:
    """Information about a discovered key."""

    name: str
    status: KeyStatus
    value: str | None = None
    source: str | None = None  # Where it was found
    masked_value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyConfig:
    """Configuration for a key in the registry."""

    name: str
    description: str
    required: bool = False
    required_condition: str | None = None  # Feature flag for conditional requirement
    env_vars: list[str] = field(default_factory=list)
    validation: str | None = None  # Validation method name
    timeout: int = 10  # Validation timeout in seconds
    prefix: str | None = None  # Expected value prefix (e.g., "glpat-")


# Key registry with supported credentials
# Simplified to only 2 keys:
# - GITLAB_TOKEN (required): For GitLab API operations
# - HOTL_EMAIL_TO (optional): For email notifications
#
# Removed keys (not needed for bootstrap):
# - ANTHROPIC_API_KEY: Users authenticated via Claude Code (Max subscription)
# - DISCORD_WEBHOOK_URL: Stick to email notifications only
# - POSTGRES_URL: SQLite is sufficient for single-user deployments
# - LANGCHAIN_API_KEY: Advanced feature, not needed during bootstrap
# - E2B_API_KEY: Advanced feature, not needed during bootstrap
# - KEEPASSXC_DB_PASSWORD: Not needed for core functionality
KEY_REGISTRY: dict[str, KeyConfig] = {
    "GITLAB_TOKEN": KeyConfig(
        name="GITLAB_TOKEN",
        description="GitLab API token for creating issues/MRs",
        required=True,
        env_vars=["GITLAB_TOKEN", "GL_TOKEN", "GLAB_TOKEN"],
        validation="gitlab_api",
        timeout=10,
        prefix="glpat-",
    ),
    "HOTL_EMAIL_TO": KeyConfig(
        name="HOTL_EMAIL_TO",
        description="Email address for HOTL notifications",
        required=False,
        env_vars=["HOTL_EMAIL_TO"],
        validation="email_format",
        timeout=1,
    ),
}


class KeyDiscovery:
    """Discovers credentials from multiple sources.

    Sources are checked in priority order:
    1. Process environment variables
    2. .env in current directory
    3. .env.local in current directory
    4. .env in git repo root
    5. Parent directories (up to 5 levels)
    6. ~/.env
    7. ~/.config/harness/.env
    8. ~/.harness/.env
    9. glab CLI config
    10. macOS Keychain
    """

    def __init__(self, project_root: Path | None = None):
        """Initialize key discovery.

        Args:
            project_root: Root directory for .env file search
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self._cache: dict[str, KeyInfo] = {}

    def discover_all(
        self,
        keys: list[str] | None = None,
        include_optional: bool = True,
    ) -> dict[str, KeyInfo]:
        """Discover all keys synchronously.

        Args:
            keys: Specific keys to discover (default: all)
            include_optional: Include optional keys

        Returns:
            Dict mapping key names to KeyInfo
        """
        if keys is None:
            keys = list(KEY_REGISTRY.keys())
            if not include_optional:
                keys = [k for k in keys if KEY_REGISTRY[k].required]

        results = {}
        for key_name in keys:
            if key_name in KEY_REGISTRY:
                results[key_name] = self._discover_key(key_name)

        return results

    async def discover_all_async(
        self,
        keys: list[str] | None = None,
        include_optional: bool = True,
    ) -> dict[str, KeyInfo]:
        """Discover all keys in parallel using asyncio.

        Args:
            keys: Specific keys to discover (default: all)
            include_optional: Include optional keys

        Returns:
            Dict mapping key names to KeyInfo
        """
        if keys is None:
            keys = list(KEY_REGISTRY.keys())
            if not include_optional:
                keys = [k for k in keys if KEY_REGISTRY[k].required]

        # Create tasks for parallel discovery
        tasks = [
            asyncio.to_thread(self._discover_key, key_name)
            for key_name in keys
            if key_name in KEY_REGISTRY
        ]

        results = await asyncio.gather(*tasks)

        return {key_name: result for key_name, result in zip(keys, results)}

    def _discover_key(self, key_name: str) -> KeyInfo:
        """Discover a single key from all sources.

        Args:
            key_name: Key to discover

        Returns:
            KeyInfo with discovery results
        """
        config = KEY_REGISTRY.get(key_name)
        if not config:
            return KeyInfo(name=key_name, status=KeyStatus.NOT_FOUND)

        # Check cache first
        if key_name in self._cache:
            return self._cache[key_name]

        # Try each source in priority order
        sources = [
            ("environment", self._check_environment),
            ("env_files", self._check_env_files),
            ("glab_config", self._check_glab_config),
            ("keychain", self._check_keychain),
        ]

        for source_name, check_func in sources:
            value = check_func(config)
            if value:
                info = KeyInfo(
                    name=key_name,
                    status=KeyStatus.FOUND,
                    value=value,
                    source=source_name,
                    masked_value=self._mask_value(value),
                )
                self._cache[key_name] = info
                return info

        info = KeyInfo(name=key_name, status=KeyStatus.NOT_FOUND)
        self._cache[key_name] = info
        return info

    def _check_environment(self, config: KeyConfig) -> str | None:
        """Check environment variables for key.

        Args:
            config: Key configuration

        Returns:
            Value if found
        """
        for env_var in config.env_vars:
            value = os.environ.get(env_var)
            if value:
                return value
        return None

    def _check_env_files(self, config: KeyConfig) -> str | None:
        """Check .env files for key.

        Args:
            config: Key configuration

        Returns:
            Value if found
        """
        env_files = self._get_env_file_paths()

        for env_file in env_files:
            if not env_file.exists():
                continue

            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        # Handle export prefix
                        if line.startswith("export "):
                            line = line[7:]

                        if "=" not in line:
                            continue

                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")

                        if key in config.env_vars and val:
                            return val
            except Exception:
                continue

        return None

    def _check_glab_config(self, config: KeyConfig) -> str | None:
        """Check glab CLI config for GitLab token.

        Args:
            config: Key configuration

        Returns:
            Value if found (only for GITLAB_TOKEN)
        """
        if "GITLAB_TOKEN" not in config.env_vars and config.name != "GITLAB_TOKEN":
            return None

        glab_config_path = Path.home() / ".config" / "glab-cli" / "config.yml"
        if not glab_config_path.exists():
            return None

        try:
            with open(glab_config_path) as f:
                glab_config = yaml.safe_load(f)

            # Navigate to hosts -> gitlab.com -> token
            hosts = glab_config.get("hosts", {})
            for host, host_config in hosts.items():
                if isinstance(host_config, dict):
                    token = host_config.get("token")
                    if token:
                        return token
        except Exception:
            pass

        return None

    def _check_keychain(self, config: KeyConfig) -> str | None:
        """Check macOS Keychain for key.

        Args:
            config: Key configuration

        Returns:
            Value if found (macOS only)
        """
        import platform

        if platform.system() != "Darwin":
            return None

        # Try each env var name as keychain service name
        for service_name in config.env_vars + [config.name]:
            try:
                result = subprocess.run(
                    ["security", "find-generic-password", "-s", service_name, "-w"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                continue

        return None

    def _get_env_file_paths(self) -> list[Path]:
        """Get list of .env file paths to search.

        Returns:
            List of paths in priority order
        """
        paths = []

        # 1. Current directory
        paths.extend(
            [
                self.project_root / ".env",
                self.project_root / ".env.local",
                self.project_root / ".env.box-up-role",
            ]
        )

        # 2. Git repo root (if different from project root)
        git_root = self._find_git_root()
        if git_root and git_root != self.project_root:
            paths.extend(
                [
                    git_root / ".env",
                    git_root / ".env.local",
                ]
            )

        # 3. Parent directories (up to 5 levels)
        current = self.project_root
        for _ in range(5):
            parent = current.parent
            if parent == current:
                break
            paths.append(parent / ".env")
            current = parent

        # 4. Home directory locations
        home = Path.home()
        paths.extend(
            [
                home / ".env",
                home / ".config" / "harness" / ".env",
                home / ".harness" / ".env",
                home / ".claude" / ".env",
            ]
        )

        # 5. Common sibling project directories
        git_dir = home / "git"
        if git_dir.exists():
            for sibling in ["ems", "crush-dots", "tinyland", "upgrading-dw"]:
                sibling_path = git_dir / sibling
                if sibling_path.exists():
                    paths.extend(
                        [
                            sibling_path / ".env",
                            sibling_path / ".env.box-up-role",
                        ]
                    )

        # 6. Config directories
        config_dir = home / ".config"
        if config_dir.exists():
            paths.extend(
                [
                    config_dir / "crush" / ".env",
                    config_dir / "claude-flow" / ".env.glm",
                ]
            )

        return paths

    def _find_git_root(self) -> Path | None:
        """Find git repository root.

        Returns:
            Path to git root or None
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=5,
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass

        return None

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

    def get_missing_required(self) -> list[str]:
        """Get list of missing required keys.

        Returns:
            List of missing required key names
        """
        missing = []
        for name, config in KEY_REGISTRY.items():
            if config.required:
                info = self._discover_key(name)
                if info.status != KeyStatus.FOUND:
                    missing.append(name)
        return missing

    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()
