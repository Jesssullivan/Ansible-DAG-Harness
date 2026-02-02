"""Installation logic for dag-harness.

This module provides:
- Installation via uv, pip, pipx, or binary
- Verification of successful installation
- Rollback on failure
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from harness.bootstrap.platform import (
    PackageManager,
    PlatformInfo,
    check_binary_compatibility,
    get_binary_url,
    get_install_path,
)


class InstallMethod(Enum):
    """Installation method used."""

    UV_TOOL = "uv_tool"
    UV_PIP = "uv_pip"
    PIP = "pip"
    PIPX = "pipx"
    BINARY = "binary"
    LOCAL = "local"


class InstallStatus(Enum):
    """Installation status."""

    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class InstallResult:
    """Result of installation attempt."""

    status: InstallStatus
    method: InstallMethod | None
    message: str
    install_path: str | None = None
    version: str | None = None
    error: str | None = None


PACKAGE_NAME = "dag-harness"


class Installer:
    """Handles installation of dag-harness package."""

    def __init__(
        self,
        platform_info: PlatformInfo,
        package_source: str | None = None,
        verbose: bool = False,
    ):
        """Initialize installer.

        Args:
            platform_info: Platform detection results
            package_source: Override package source (PyPI name, path, or URL)
            verbose: Enable verbose output
        """
        self.platform = platform_info
        self.package_source = package_source or PACKAGE_NAME
        self.verbose = verbose
        self._backup_path: Path | None = None

    def install(self, prefer_binary: bool = False) -> InstallResult:
        """Install dag-harness using the best available method.

        Installation priority:
        1. uv tool install (if uv available)
        2. pip install --user
        3. pipx install
        4. Binary download (if no Python available)

        Args:
            prefer_binary: Skip Python-based installation, use binary

        Returns:
            InstallResult with status and details
        """
        if prefer_binary:
            return self._install_binary()

        # Try installation methods in order
        methods = [
            (PackageManager.UV, self._install_via_uv),
            (PackageManager.PIP, self._install_via_pip),
            (PackageManager.PIPX, self._install_via_pipx),
        ]

        for pkg_mgr, install_func in methods:
            if self.platform.package_manager == pkg_mgr:
                result = install_func()
                if result.status == InstallStatus.SUCCESS:
                    return result
                # Log failure and try next method
                if self.verbose:
                    print(f"  {pkg_mgr.value} installation failed: {result.error}")

        # Fallback to binary if all Python methods failed
        if check_binary_compatibility(self.platform):
            return self._install_binary()

        return InstallResult(
            status=InstallStatus.FAILED,
            method=None,
            message="All installation methods failed",
            error="No suitable installation method available",
        )

    def _install_via_uv(self) -> InstallResult:
        """Install using uv tool install.

        Returns:
            InstallResult
        """
        try:
            # First try uv tool install
            result = subprocess.run(
                ["uv", "tool", "install", self.package_source],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                install_path = self._find_installed_binary()
                version = self._get_installed_version()
                return InstallResult(
                    status=InstallStatus.SUCCESS,
                    method=InstallMethod.UV_TOOL,
                    message="Installed via uv tool",
                    install_path=install_path,
                    version=version,
                )

            # Try uv pip install as fallback
            result = subprocess.run(
                ["uv", "pip", "install", "--user", self.package_source],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                install_path = self._find_installed_binary()
                version = self._get_installed_version()
                return InstallResult(
                    status=InstallStatus.SUCCESS,
                    method=InstallMethod.UV_PIP,
                    message="Installed via uv pip",
                    install_path=install_path,
                    version=version,
                )

            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.UV_TOOL,
                message="uv installation failed",
                error=result.stderr,
            )

        except subprocess.TimeoutExpired:
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.UV_TOOL,
                message="Installation timed out",
                error="uv command timed out after 120 seconds",
            )
        except Exception as e:
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.UV_TOOL,
                message="uv installation error",
                error=str(e),
            )

    def _install_via_pip(self) -> InstallResult:
        """Install using pip install --user.

        Returns:
            InstallResult
        """
        try:
            # Use Python's pip module to avoid PATH issues
            result = subprocess.run(
                [
                    self.platform.python_path,
                    "-m",
                    "pip",
                    "install",
                    "--user",
                    self.package_source,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                install_path = self._find_installed_binary()
                version = self._get_installed_version()
                return InstallResult(
                    status=InstallStatus.SUCCESS,
                    method=InstallMethod.PIP,
                    message="Installed via pip",
                    install_path=install_path,
                    version=version,
                )

            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.PIP,
                message="pip installation failed",
                error=result.stderr,
            )

        except subprocess.TimeoutExpired:
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.PIP,
                message="Installation timed out",
                error="pip command timed out after 120 seconds",
            )
        except Exception as e:
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.PIP,
                message="pip installation error",
                error=str(e),
            )

    def _install_via_pipx(self) -> InstallResult:
        """Install using pipx.

        Returns:
            InstallResult
        """
        try:
            result = subprocess.run(
                ["pipx", "install", self.package_source],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                install_path = self._find_installed_binary()
                version = self._get_installed_version()
                return InstallResult(
                    status=InstallStatus.SUCCESS,
                    method=InstallMethod.PIPX,
                    message="Installed via pipx",
                    install_path=install_path,
                    version=version,
                )

            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.PIPX,
                message="pipx installation failed",
                error=result.stderr,
            )

        except subprocess.TimeoutExpired:
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.PIPX,
                message="Installation timed out",
                error="pipx command timed out after 120 seconds",
            )
        except Exception as e:
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.PIPX,
                message="pipx installation error",
                error=str(e),
            )

    def _install_binary(self) -> InstallResult:
        """Download and install pre-built binary.

        Returns:
            InstallResult
        """
        if not check_binary_compatibility(self.platform):
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.BINARY,
                message="No binary available for this platform",
                error=f"Platform {self.platform.platform_string} not supported",
            )

        try:
            url = get_binary_url(self.platform)
            install_dir = Path(get_install_path())
            install_path = install_dir / "harness"

            # Create install directory
            install_dir.mkdir(parents=True, exist_ok=True)

            # Backup existing binary
            if install_path.exists():
                self._backup_path = install_path.with_suffix(".backup")
                shutil.copy2(install_path, self._backup_path)

            # Download binary
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

                try:
                    with urlopen(url, timeout=60) as response:
                        tmp_file.write(response.read())
                except URLError as e:
                    return InstallResult(
                        status=InstallStatus.FAILED,
                        method=InstallMethod.BINARY,
                        message="Binary download failed",
                        error=str(e),
                    )

            # Move to install location
            shutil.move(str(tmp_path), str(install_path))
            install_path.chmod(0o755)

            # Verify binary works
            result = subprocess.run(
                [str(install_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                self._rollback()
                return InstallResult(
                    status=InstallStatus.ROLLBACK,
                    method=InstallMethod.BINARY,
                    message="Binary verification failed, rolled back",
                    error=result.stderr,
                )

            # Clean up backup
            if self._backup_path and self._backup_path.exists():
                self._backup_path.unlink()

            version = result.stdout.strip().split()[-1] if result.stdout else None
            return InstallResult(
                status=InstallStatus.SUCCESS,
                method=InstallMethod.BINARY,
                message="Installed pre-built binary",
                install_path=str(install_path),
                version=version,
            )

        except Exception as e:
            self._rollback()
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.BINARY,
                message="Binary installation error",
                error=str(e),
            )

    def install_local(self, path: Path) -> InstallResult:
        """Install from local source directory.

        Args:
            path: Path to project directory containing pyproject.toml

        Returns:
            InstallResult
        """
        pyproject = path / "pyproject.toml"
        if not pyproject.exists():
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.LOCAL,
                message="Local source not found",
                error=f"No pyproject.toml at {path}",
            )

        try:
            if self.platform.package_manager == PackageManager.UV:
                result = subprocess.run(
                    ["uv", "pip", "install", "-e", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            else:
                result = subprocess.run(
                    [self.platform.python_path, "-m", "pip", "install", "-e", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

            if result.returncode == 0:
                version = self._get_installed_version()
                return InstallResult(
                    status=InstallStatus.SUCCESS,
                    method=InstallMethod.LOCAL,
                    message="Installed from local source",
                    install_path=str(path),
                    version=version,
                )

            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.LOCAL,
                message="Local installation failed",
                error=result.stderr,
            )

        except Exception as e:
            return InstallResult(
                status=InstallStatus.FAILED,
                method=InstallMethod.LOCAL,
                message="Local installation error",
                error=str(e),
            )

    def _rollback(self) -> None:
        """Rollback to previous installation if backup exists."""
        if self._backup_path and self._backup_path.exists():
            install_path = self._backup_path.with_suffix("")
            shutil.move(str(self._backup_path), str(install_path))

    def _find_installed_binary(self) -> str | None:
        """Find the installed harness binary.

        Returns:
            Path to binary or None
        """
        return shutil.which("harness")

    def _get_installed_version(self) -> str | None:
        """Get version of installed harness.

        Returns:
            Version string or None
        """
        try:
            result = subprocess.run(
                ["harness", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse version from output like "harness 0.2.0"
                parts = result.stdout.strip().split()
                return parts[-1] if parts else None
        except Exception:
            pass

        return None

    def verify_installation(self) -> bool:
        """Verify harness is properly installed and functional.

        Returns:
            True if verification passes
        """
        binary_path = self._find_installed_binary()
        if not binary_path:
            return False

        try:
            # Test basic command
            result = subprocess.run(
                [binary_path, "--help"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False


def install(
    platform_info: PlatformInfo | None = None,
    prefer_binary: bool = False,
    verbose: bool = False,
) -> InstallResult:
    """Convenience function to install dag-harness.

    Args:
        platform_info: Platform info (auto-detected if not provided)
        prefer_binary: Skip Python methods, use binary
        verbose: Enable verbose output

    Returns:
        InstallResult
    """
    if platform_info is None:
        from harness.bootstrap.platform import detect_platform

        platform_info = detect_platform()

    installer = Installer(platform_info, verbose=verbose)
    return installer.install(prefer_binary=prefer_binary)
