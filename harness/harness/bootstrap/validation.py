"""Async validation for discovered credentials.

This module provides:
- Async validation with configurable timeouts
- Validation methods for supported key types (gitlab_api, email_format, url_format)
- Rate limiting and retry logic
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from harness.bootstrap.discovery import KEY_REGISTRY, KeyConfig, KeyInfo, KeyStatus


class ValidationStatus(Enum):
    """Validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of credential validation."""

    key_name: str
    status: ValidationStatus
    message: str
    details: dict[str, Any] | None = None
    duration_ms: float | None = None


class CredentialValidator:
    """Validates credentials against their respective services."""

    def __init__(self, default_timeout: int = 10):
        """Initialize validator.

        Args:
            default_timeout: Default timeout in seconds
        """
        self.default_timeout = default_timeout

        # Register validation methods
        # Simplified to only validators needed for the 2 supported keys
        self._validators = {
            "gitlab_api": self._validate_gitlab,
            "email_format": self._validate_email,
            "url_format": self._validate_url,
        }

    async def validate_all(
        self,
        keys: dict[str, KeyInfo],
        parallel: bool = True,
    ) -> dict[str, ValidationResult]:
        """Validate all discovered keys.

        Args:
            keys: Dict of KeyInfo objects from discovery
            parallel: Run validations in parallel

        Returns:
            Dict mapping key names to ValidationResult
        """
        if parallel:
            tasks = [
                self._validate_key_async(name, info)
                for name, info in keys.items()
                if info.status == KeyStatus.FOUND
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            return {
                name: (
                    result
                    if isinstance(result, ValidationResult)
                    else ValidationResult(
                        key_name=name,
                        status=ValidationStatus.ERROR,
                        message=str(result),
                    )
                )
                for name, result in zip(
                    [n for n, i in keys.items() if i.status == KeyStatus.FOUND],
                    results,
                )
            }
        else:
            results = {}
            for name, info in keys.items():
                if info.status == KeyStatus.FOUND:
                    results[name] = await self._validate_key_async(name, info)
            return results

    async def _validate_key_async(
        self,
        name: str,
        info: KeyInfo,
    ) -> ValidationResult:
        """Validate a single key asynchronously.

        Args:
            name: Key name
            info: Key information

        Returns:
            ValidationResult
        """
        config = KEY_REGISTRY.get(name)
        if not config or not config.validation:
            return ValidationResult(
                key_name=name,
                status=ValidationStatus.SKIPPED,
                message="No validation defined",
            )

        if not info.value:
            return ValidationResult(
                key_name=name,
                status=ValidationStatus.INVALID,
                message="No value to validate",
            )

        validator = self._validators.get(config.validation)
        if not validator:
            return ValidationResult(
                key_name=name,
                status=ValidationStatus.SKIPPED,
                message=f"Unknown validation type: {config.validation}",
            )

        timeout = config.timeout or self.default_timeout

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(validator, info.value, config),
                timeout=timeout,
            )
            return result
        except TimeoutError:
            return ValidationResult(
                key_name=name,
                status=ValidationStatus.TIMEOUT,
                message=f"Validation timed out after {timeout}s",
            )
        except Exception as e:
            return ValidationResult(
                key_name=name,
                status=ValidationStatus.ERROR,
                message=f"Validation error: {str(e)}",
            )

    def validate_sync(self, name: str, value: str) -> ValidationResult:
        """Validate a credential synchronously.

        Args:
            name: Key name
            value: Credential value

        Returns:
            ValidationResult
        """
        config = KEY_REGISTRY.get(name)
        if not config or not config.validation:
            return ValidationResult(
                key_name=name,
                status=ValidationStatus.SKIPPED,
                message="No validation defined",
            )

        validator = self._validators.get(config.validation)
        if not validator:
            return ValidationResult(
                key_name=name,
                status=ValidationStatus.SKIPPED,
                message=f"Unknown validation type: {config.validation}",
            )

        return validator(value, config)

    def _validate_gitlab(self, token: str, config: KeyConfig) -> ValidationResult:
        """Validate GitLab API token.

        Args:
            token: GitLab token
            config: Key configuration

        Returns:
            ValidationResult with user info if valid
        """
        import time

        start = time.time()

        gitlab_url = os.environ.get("GITLAB_URL", "https://gitlab.com")

        try:
            req = Request(
                f"{gitlab_url}/api/v4/user",
                headers={"PRIVATE-TOKEN": token},
            )

            with urlopen(req, timeout=config.timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    duration = (time.time() - start) * 1000

                    # Check scopes
                    scopes = self._get_gitlab_scopes(token, gitlab_url, config.timeout)

                    return ValidationResult(
                        key_name=config.name,
                        status=ValidationStatus.VALID,
                        message=f"Valid token for @{data.get('username', 'unknown')}",
                        details={
                            "username": data.get("username"),
                            "name": data.get("name"),
                            "scopes": scopes,
                        },
                        duration_ms=duration,
                    )

        except HTTPError as e:
            duration = (time.time() - start) * 1000
            if e.code == 401:
                return ValidationResult(
                    key_name=config.name,
                    status=ValidationStatus.INVALID,
                    message="Invalid token (401 Unauthorized)",
                    duration_ms=duration,
                )
            return ValidationResult(
                key_name=config.name,
                status=ValidationStatus.ERROR,
                message=f"GitLab API error: HTTP {e.code}",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                key_name=config.name,
                status=ValidationStatus.ERROR,
                message=str(e),
                duration_ms=duration,
            )

    def _get_gitlab_scopes(
        self,
        token: str,
        gitlab_url: str,
        timeout: int,
    ) -> list[str] | None:
        """Get GitLab token scopes.

        Args:
            token: GitLab token
            gitlab_url: GitLab instance URL
            timeout: Request timeout

        Returns:
            List of scopes or None
        """
        try:
            req = Request(
                f"{gitlab_url}/api/v4/personal_access_tokens/self",
                headers={"PRIVATE-TOKEN": token},
            )

            with urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    return data.get("scopes", [])
        except Exception:
            pass

        return None

    def _validate_email(self, email: str, config: KeyConfig) -> ValidationResult:
        """Validate email format.

        Args:
            email: Email address
            config: Key configuration

        Returns:
            ValidationResult
        """
        import time

        start = time.time()

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if re.match(pattern, email):
            return ValidationResult(
                key_name=config.name,
                status=ValidationStatus.VALID,
                message="Email format is valid",
                duration_ms=(time.time() - start) * 1000,
            )

        return ValidationResult(
            key_name=config.name,
            status=ValidationStatus.INVALID,
            message="Invalid email format",
            duration_ms=(time.time() - start) * 1000,
        )

    def _validate_url(self, url: str, config: KeyConfig) -> ValidationResult:
        """Validate URL format.

        Args:
            url: URL to validate
            config: Key configuration

        Returns:
            ValidationResult
        """
        import time
        from urllib.parse import urlparse

        start = time.time()

        try:
            result = urlparse(url)
            if all([result.scheme, result.netloc]):
                return ValidationResult(
                    key_name=config.name,
                    status=ValidationStatus.VALID,
                    message="URL format is valid",
                    duration_ms=(time.time() - start) * 1000,
                )
        except Exception:
            pass

        return ValidationResult(
            key_name=config.name,
            status=ValidationStatus.INVALID,
            message="Invalid URL format",
            duration_ms=(time.time() - start) * 1000,
        )


async def validate_credentials(
    keys: dict[str, KeyInfo],
    parallel: bool = True,
) -> dict[str, ValidationResult]:
    """Convenience function to validate discovered credentials.

    Args:
        keys: Dict of KeyInfo from discovery
        parallel: Run validations in parallel

    Returns:
        Dict of ValidationResult
    """
    validator = CredentialValidator()
    return await validator.validate_all(keys, parallel=parallel)
