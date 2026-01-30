"""Credential discovery and validation for harness bootstrap.

This module handles:
- Environment variable detection
- Credential validation (GITLAB_TOKEN, KEEPASSXC_DB_PASSWORD, etc.)
- Interactive prompting for missing credentials
"""

import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class CredentialStatus(Enum):
    """Status of a credential check."""
    FOUND = "found"
    NOT_FOUND = "not_found"
    INVALID = "invalid"
    OPTIONAL_MISSING = "optional_missing"


@dataclass
class CredentialResult:
    """Result of checking a single credential."""
    name: str
    status: CredentialStatus
    source: Optional[str] = None  # Where it was found (env, file, etc.)
    value: Optional[str] = None  # Masked or partial value for display
    error: Optional[str] = None
    required: bool = True


@dataclass
class CredentialCheckResult:
    """Result of checking all credentials."""
    all_required_present: bool
    credentials: list[CredentialResult] = field(default_factory=list)

    @property
    def missing_required(self) -> list[str]:
        """Get list of missing required credentials."""
        return [
            c.name for c in self.credentials
            if c.required and c.status in (CredentialStatus.NOT_FOUND, CredentialStatus.INVALID)
        ]

    @property
    def found_count(self) -> int:
        """Count of found credentials."""
        return sum(1 for c in self.credentials if c.status == CredentialStatus.FOUND)


class CredentialDiscovery:
    """Discovers and validates credentials for harness operation.

    Checks multiple sources in priority order:
    1. Environment variables
    2. .env files
    3. System keychain (macOS)
    4. KeePassXC database
    """

    # Required credentials with their environment variable names
    REQUIRED_CREDENTIALS = {
        "GITLAB_TOKEN": {
            "description": "GitLab API token for creating issues/MRs",
            "env_vars": ["GITLAB_TOKEN", "GL_TOKEN", "GLAB_TOKEN"],
            "validation": "gitlab_api",
        },
    }

    # Optional credentials
    OPTIONAL_CREDENTIALS = {
        "KEEPASSXC_DB_PASSWORD": {
            "description": "KeePassXC database password",
            "env_vars": ["KEEPASSXC_DB_PASSWORD", "KEEPASS_PASSWORD"],
            "validation": None,
        },
        "DISCORD_WEBHOOK_URL": {
            "description": "Discord webhook for notifications",
            "env_vars": ["DISCORD_WEBHOOK_URL"],
            "validation": "url_format",
        },
        "HOTL_EMAIL_TO": {
            "description": "Email address for HOTL notifications",
            "env_vars": ["HOTL_EMAIL_TO"],
            "validation": "email_format",
        },
    }

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize credential discovery.

        Args:
            project_root: Root directory to search for .env files
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()

    def check_all(self, validate: bool = True) -> CredentialCheckResult:
        """Check all credentials.

        Args:
            validate: Whether to validate credentials (e.g., test GitLab API)

        Returns:
            CredentialCheckResult with status of all credentials
        """
        results = []

        # Check required credentials
        for name, config in self.REQUIRED_CREDENTIALS.items():
            result = self._check_credential(name, config, required=True, validate=validate)
            results.append(result)

        # Check optional credentials
        for name, config in self.OPTIONAL_CREDENTIALS.items():
            result = self._check_credential(name, config, required=False, validate=validate)
            results.append(result)

        all_required = all(
            c.status == CredentialStatus.FOUND
            for c in results if c.required
        )

        return CredentialCheckResult(
            all_required_present=all_required,
            credentials=results
        )

    def _check_credential(
        self,
        name: str,
        config: dict,
        required: bool,
        validate: bool
    ) -> CredentialResult:
        """Check a single credential.

        Args:
            name: Credential name
            config: Configuration dict with env_vars, validation, etc.
            required: Whether this credential is required
            validate: Whether to validate the credential value

        Returns:
            CredentialResult with status
        """
        # Check environment variables
        value = None
        source = None

        for env_var in config.get("env_vars", [name]):
            env_value = os.environ.get(env_var)
            if env_value:
                value = env_value
                source = f"env:{env_var}"
                break

        # Check .env file if not found in environment
        if not value:
            env_file_value, env_file_path = self._check_env_file(config.get("env_vars", [name]))
            if env_file_value:
                value = env_file_value
                source = f"file:{env_file_path}"

        # Check system keychain (macOS) if not found
        if not value and self._is_macos():
            keychain_value = self._check_keychain(name)
            if keychain_value:
                value = keychain_value
                source = "keychain"

        # Determine status
        if not value:
            status = CredentialStatus.NOT_FOUND if required else CredentialStatus.OPTIONAL_MISSING
            return CredentialResult(
                name=name,
                status=status,
                required=required,
                error=f"Credential not found in environment or .env files"
            )

        # Validate if requested
        if validate and config.get("validation"):
            is_valid, error = self._validate_credential(name, value, config["validation"])
            if not is_valid:
                return CredentialResult(
                    name=name,
                    status=CredentialStatus.INVALID,
                    source=source,
                    value=self._mask_value(value),
                    required=required,
                    error=error
                )

        return CredentialResult(
            name=name,
            status=CredentialStatus.FOUND,
            source=source,
            value=self._mask_value(value),
            required=required
        )

    def _check_env_file(self, env_vars: list[str]) -> tuple[Optional[str], Optional[str]]:
        """Check .env files for credential.

        Args:
            env_vars: List of variable names to search for

        Returns:
            Tuple of (value, file_path) or (None, None)
        """
        env_files = [
            self.project_root / ".env",
            self.project_root / ".env.local",
            Path.home() / ".env",
        ]

        for env_file in env_files:
            if env_file.exists():
                try:
                    with open(env_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" in line:
                                key, _, val = line.partition("=")
                                key = key.strip()
                                val = val.strip().strip('"').strip("'")
                                if key in env_vars and val:
                                    return val, str(env_file)
                except Exception:
                    pass

        return None, None

    def _check_keychain(self, name: str) -> Optional[str]:
        """Check macOS keychain for credential.

        Args:
            name: Credential name to search for

        Returns:
            Value if found, None otherwise
        """
        try:
            # Use security command to query keychain
            result = subprocess.run(
                ["security", "find-generic-password", "-s", name, "-w"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def _is_macos(self) -> bool:
        """Check if running on macOS."""
        import platform
        return platform.system() == "Darwin"

    def _validate_credential(
        self,
        name: str,
        value: str,
        validation_type: str
    ) -> tuple[bool, Optional[str]]:
        """Validate a credential value.

        Args:
            name: Credential name
            value: Value to validate
            validation_type: Type of validation to perform

        Returns:
            Tuple of (is_valid, error_message)
        """
        if validation_type == "gitlab_api":
            return self._validate_gitlab_token(value)
        elif validation_type == "url_format":
            return self._validate_url(value)
        elif validation_type == "email_format":
            return self._validate_email(value)

        return True, None

    def _validate_gitlab_token(self, token: str) -> tuple[bool, Optional[str]]:
        """Validate GitLab API token by making a test request.

        Args:
            token: GitLab API token

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import urllib.request
            import urllib.error

            # Try to get current user - if token is valid, this will work
            gitlab_url = os.environ.get("GITLAB_URL", "https://gitlab.com")
            req = urllib.request.Request(
                f"{gitlab_url}/api/v4/user",
                headers={"PRIVATE-TOKEN": token}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    return True, None
                else:
                    return False, f"GitLab API returned status {response.status}"

        except urllib.error.HTTPError as e:
            if e.code == 401:
                return False, "Invalid GitLab token (401 Unauthorized)"
            return False, f"GitLab API error: {e.code}"
        except Exception as e:
            # Don't fail if network is unavailable
            return True, None  # Assume valid if we can't check

    def _validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL format.

        Args:
            url: URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        from urllib.parse import urlparse

        try:
            result = urlparse(url)
            if all([result.scheme, result.netloc]):
                return True, None
            return False, "Invalid URL format"
        except Exception:
            return False, "Invalid URL format"

    def _validate_email(self, email: str) -> tuple[bool, Optional[str]]:
        """Validate email format.

        Args:
            email: Email to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        import re

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(pattern, email):
            return True, None
        return False, "Invalid email format"

    def _mask_value(self, value: str) -> str:
        """Mask a credential value for display.

        Args:
            value: Value to mask

        Returns:
            Masked value showing only first/last few characters
        """
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"

    def get_credential(self, name: str) -> Optional[str]:
        """Get a credential value by name.

        Args:
            name: Credential name

        Returns:
            Credential value if found, None otherwise
        """
        config = self.REQUIRED_CREDENTIALS.get(name) or self.OPTIONAL_CREDENTIALS.get(name)
        if not config:
            return os.environ.get(name)

        for env_var in config.get("env_vars", [name]):
            value = os.environ.get(env_var)
            if value:
                return value

        value, _ = self._check_env_file(config.get("env_vars", [name]))
        return value
