"""Upgrade mechanism for dag-harness.

This module provides:
- Version checking against PyPI and GitHub releases
- Upgrade execution for different installation methods
- Version comparison logic
"""

import json
import subprocess
from dataclasses import dataclass
from enum import Enum
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

from harness import __version__ as CURRENT_VERSION

PACKAGE_NAME = "dag-harness"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
GITHUB_API_URL = "https://api.github.com/repos/user/dag-harness/releases/latest"


class UpgradeStatus(Enum):
    """Upgrade operation status."""

    UP_TO_DATE = "up_to_date"
    UPGRADE_AVAILABLE = "upgrade_available"
    UPGRADED = "upgraded"
    FAILED = "failed"
    CHECK_FAILED = "check_failed"


@dataclass
class VersionInfo:
    """Version information from package registry."""

    current: str
    latest: str
    upgrade_available: bool
    source: str  # "pypi" or "github"


@dataclass
class UpgradeResult:
    """Result of upgrade operation."""

    status: UpgradeStatus
    message: str
    current_version: str | None = None
    new_version: str | None = None
    error: str | None = None


def parse_version(version_str: str) -> Version | None:
    """Parse version string to Version object.

    Args:
        version_str: Version string (e.g., "0.2.0", "v0.2.0")

    Returns:
        Version object or None if invalid
    """
    # Strip leading 'v' if present
    if version_str.startswith("v"):
        version_str = version_str[1:]

    try:
        return Version(version_str)
    except InvalidVersion:
        return None


def check_pypi_version() -> str | None:
    """Check latest version available on PyPI.

    Returns:
        Latest version string or None if check fails
    """
    try:
        with urlopen(PYPI_URL, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get("info", {}).get("version")
    except Exception:
        return None


def check_github_version() -> str | None:
    """Check latest version from GitHub releases.

    Returns:
        Latest version string or None if check fails
    """
    try:
        req = Request(
            GITHUB_API_URL,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            tag_name = data.get("tag_name", "")
            # Strip 'v' prefix if present
            return tag_name.lstrip("v") if tag_name else None
    except Exception:
        return None


def check_for_upgrade() -> VersionInfo:
    """Check if an upgrade is available.

    Checks PyPI first, then falls back to GitHub releases.

    Returns:
        VersionInfo with current/latest versions and upgrade status
    """
    current = CURRENT_VERSION
    current_ver = parse_version(current)

    # Try PyPI first
    pypi_version = check_pypi_version()
    if pypi_version:
        latest_ver = parse_version(pypi_version)
        if latest_ver and current_ver:
            return VersionInfo(
                current=current,
                latest=pypi_version,
                upgrade_available=latest_ver > current_ver,
                source="pypi",
            )

    # Fall back to GitHub
    github_version = check_github_version()
    if github_version:
        latest_ver = parse_version(github_version)
        if latest_ver and current_ver:
            return VersionInfo(
                current=current,
                latest=github_version,
                upgrade_available=latest_ver > current_ver,
                source="github",
            )

    # Couldn't determine latest version
    return VersionInfo(
        current=current,
        latest=current,  # Assume current if we can't check
        upgrade_available=False,
        source="unknown",
    )


def detect_install_method() -> str | None:
    """Detect how dag-harness was installed.

    Returns:
        Installation method: "uv", "pip", "pipx", "binary", or None
    """
    import shutil

    harness_path = shutil.which("harness")
    if not harness_path:
        return None

    # Check if installed via uv
    try:
        result = subprocess.run(
            ["uv", "tool", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if PACKAGE_NAME in result.stdout:
            return "uv"
    except Exception:
        pass

    # Check if installed via pipx
    try:
        result = subprocess.run(
            ["pipx", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if PACKAGE_NAME in result.stdout:
            return "pipx"
    except Exception:
        pass

    # Check if it's a standalone binary (not in a Python site-packages)
    if "/site-packages/" not in harness_path and "/.local/bin/" in harness_path:
        # Could be binary or pip --user install
        # Check if pip knows about it
        try:
            result = subprocess.run(
                ["pip", "show", PACKAGE_NAME],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return "pip"
        except Exception:
            pass

        return "binary"

    return "pip"  # Default assumption


def upgrade_via_uv() -> UpgradeResult:
    """Upgrade using uv tool.

    Returns:
        UpgradeResult
    """
    try:
        result = subprocess.run(
            ["uv", "tool", "upgrade", PACKAGE_NAME],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            # Get new version
            new_version = check_pypi_version() or CURRENT_VERSION
            return UpgradeResult(
                status=UpgradeStatus.UPGRADED,
                message="Upgraded via uv tool",
                current_version=CURRENT_VERSION,
                new_version=new_version,
            )

        return UpgradeResult(
            status=UpgradeStatus.FAILED,
            message="uv upgrade failed",
            error=result.stderr,
        )

    except Exception as e:
        return UpgradeResult(
            status=UpgradeStatus.FAILED,
            message="uv upgrade error",
            error=str(e),
        )


def upgrade_via_pip() -> UpgradeResult:
    """Upgrade using pip.

    Returns:
        UpgradeResult
    """
    try:
        result = subprocess.run(
            ["pip", "install", "--upgrade", "--user", PACKAGE_NAME],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            new_version = check_pypi_version() or CURRENT_VERSION
            return UpgradeResult(
                status=UpgradeStatus.UPGRADED,
                message="Upgraded via pip",
                current_version=CURRENT_VERSION,
                new_version=new_version,
            )

        return UpgradeResult(
            status=UpgradeStatus.FAILED,
            message="pip upgrade failed",
            error=result.stderr,
        )

    except Exception as e:
        return UpgradeResult(
            status=UpgradeStatus.FAILED,
            message="pip upgrade error",
            error=str(e),
        )


def upgrade_via_pipx() -> UpgradeResult:
    """Upgrade using pipx.

    Returns:
        UpgradeResult
    """
    try:
        result = subprocess.run(
            ["pipx", "upgrade", PACKAGE_NAME],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            new_version = check_pypi_version() or CURRENT_VERSION
            return UpgradeResult(
                status=UpgradeStatus.UPGRADED,
                message="Upgraded via pipx",
                current_version=CURRENT_VERSION,
                new_version=new_version,
            )

        return UpgradeResult(
            status=UpgradeStatus.FAILED,
            message="pipx upgrade failed",
            error=result.stderr,
        )

    except Exception as e:
        return UpgradeResult(
            status=UpgradeStatus.FAILED,
            message="pipx upgrade error",
            error=str(e),
        )


def upgrade_binary() -> UpgradeResult:
    """Upgrade binary installation by re-downloading.

    Returns:
        UpgradeResult
    """
    from harness.bootstrap.installer import Installer
    from harness.bootstrap.platform import detect_platform

    try:
        platform_info = detect_platform()
        installer = Installer(platform_info)
        result = installer._install_binary()

        if result.status.value == "success":
            return UpgradeResult(
                status=UpgradeStatus.UPGRADED,
                message="Binary upgraded",
                current_version=CURRENT_VERSION,
                new_version=result.version,
            )

        return UpgradeResult(
            status=UpgradeStatus.FAILED,
            message="Binary upgrade failed",
            error=result.error,
        )

    except Exception as e:
        return UpgradeResult(
            status=UpgradeStatus.FAILED,
            message="Binary upgrade error",
            error=str(e),
        )


def upgrade(check_only: bool = False) -> UpgradeResult:
    """Check for and optionally install upgrades.

    Args:
        check_only: If True, only check for upgrades without installing

    Returns:
        UpgradeResult with status and details
    """
    # Check for available upgrade
    version_info = check_for_upgrade()

    if not version_info.upgrade_available:
        return UpgradeResult(
            status=UpgradeStatus.UP_TO_DATE,
            message=f"Already at latest version ({version_info.current})",
            current_version=version_info.current,
        )

    if check_only:
        return UpgradeResult(
            status=UpgradeStatus.UPGRADE_AVAILABLE,
            message=f"Upgrade available: {version_info.current} -> {version_info.latest}",
            current_version=version_info.current,
            new_version=version_info.latest,
        )

    # Detect installation method and upgrade accordingly
    install_method = detect_install_method()

    upgrade_functions = {
        "uv": upgrade_via_uv,
        "pip": upgrade_via_pip,
        "pipx": upgrade_via_pipx,
        "binary": upgrade_binary,
    }

    upgrade_func = upgrade_functions.get(install_method)
    if upgrade_func:
        return upgrade_func()

    return UpgradeResult(
        status=UpgradeStatus.FAILED,
        message="Unknown installation method",
        error=f"Cannot upgrade: unknown install method '{install_method}'",
    )
