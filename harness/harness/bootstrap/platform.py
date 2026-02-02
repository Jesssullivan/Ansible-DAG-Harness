"""Platform detection and environment analysis for bootstrap.

This module provides:
- OS and architecture detection
- Python version checking
- Package manager detection (uv, pip, pipx)
- Binary compatibility checking
"""

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum


class OS(Enum):
    """Supported operating systems."""

    DARWIN = "darwin"
    LINUX = "linux"
    ROCKY = "rocky"  # Rocky Linux (special case)
    UNKNOWN = "unknown"


class Architecture(Enum):
    """Supported CPU architectures."""

    ARM64 = "arm64"
    X86_64 = "x86_64"
    UNKNOWN = "unknown"


class PackageManager(Enum):
    """Available package managers for installation."""

    UV = "uv"
    PIP = "pip"
    PIPX = "pipx"
    NONE = "none"


@dataclass
class PlatformInfo:
    """Information about the current platform."""

    os: OS
    arch: Architecture
    os_version: str
    python_version: tuple[int, int, int]
    python_path: str
    package_manager: PackageManager
    package_manager_path: str | None
    is_virtual_env: bool
    home_dir: str

    @property
    def platform_string(self) -> str:
        """Get platform identifier string (e.g., 'darwin-arm64')."""
        os_name = self.os.value
        if self.os == OS.ROCKY:
            os_name = "rocky"
        return f"{os_name}-{self.arch.value}"

    @property
    def python_version_string(self) -> str:
        """Get Python version as string (e.g., '3.11.5')."""
        return ".".join(str(v) for v in self.python_version)

    @property
    def meets_python_requirement(self) -> bool:
        """Check if Python version meets minimum requirement (3.11+)."""
        return self.python_version >= (3, 11, 0)


def detect_os() -> OS:
    """Detect the operating system.

    Returns:
        OS enum value
    """
    system = platform.system().lower()

    if system == "darwin":
        return OS.DARWIN
    elif system == "linux":
        # Check for Rocky Linux specifically
        if os.path.exists("/etc/rocky-release"):
            return OS.ROCKY
        return OS.LINUX

    return OS.UNKNOWN


def detect_architecture() -> Architecture:
    """Detect the CPU architecture.

    Returns:
        Architecture enum value
    """
    machine = platform.machine().lower()

    if machine in ("arm64", "aarch64"):
        return Architecture.ARM64
    elif machine in ("x86_64", "amd64"):
        return Architecture.X86_64

    return Architecture.UNKNOWN


def detect_package_manager() -> tuple[PackageManager, str | None]:
    """Detect the best available package manager.

    Priority: uv > pipx > pip

    Returns:
        Tuple of (PackageManager, path_to_executable)
    """
    # Check for uv (preferred)
    uv_path = shutil.which("uv")
    if uv_path:
        return PackageManager.UV, uv_path

    # Check for pipx
    pipx_path = shutil.which("pipx")
    if pipx_path:
        return PackageManager.PIPX, pipx_path

    # Check for pip (either pip or pip3)
    pip_path = shutil.which("pip3") or shutil.which("pip")
    if pip_path:
        return PackageManager.PIP, pip_path

    return PackageManager.NONE, None


def is_virtual_environment() -> bool:
    """Check if running inside a virtual environment.

    Returns:
        True if in a virtualenv, venv, or conda environment
    """
    return (
        hasattr(sys, "real_prefix")  # virtualenv
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)  # venv
        or os.environ.get("CONDA_DEFAULT_ENV") is not None  # conda
    )


def get_os_version() -> str:
    """Get detailed OS version string.

    Returns:
        OS version string
    """
    system = platform.system()

    if system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver}"
    elif system == "Linux":
        # Try to get distro info
        try:
            import distro

            return f"{distro.name()} {distro.version()}"
        except ImportError:
            pass

        # Fallback: read /etc/os-release
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')

        return f"Linux {platform.release()}"

    return platform.platform()


def detect_platform() -> PlatformInfo:
    """Detect full platform information.

    Returns:
        PlatformInfo with all detected information
    """
    os_type = detect_os()
    arch = detect_architecture()
    pkg_mgr, pkg_mgr_path = detect_package_manager()

    return PlatformInfo(
        os=os_type,
        arch=arch,
        os_version=get_os_version(),
        python_version=sys.version_info[:3],
        python_path=sys.executable,
        package_manager=pkg_mgr,
        package_manager_path=pkg_mgr_path,
        is_virtual_env=is_virtual_environment(),
        home_dir=str(os.path.expanduser("~")),
    )


def find_python(min_version: tuple[int, int] = (3, 11)) -> str | None:
    """Find a suitable Python interpreter.

    Args:
        min_version: Minimum required Python version (major, minor)

    Returns:
        Path to Python interpreter or None if not found
    """
    candidates = [
        "python3.13",
        "python3.12",
        "python3.11",
        "python3",
        "python",
    ]

    for candidate in candidates:
        python_path = shutil.which(candidate)
        if python_path:
            try:
                result = subprocess.run(
                    [python_path, "-c", "import sys; print(sys.version_info[:2])"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    version = eval(result.stdout.strip())
                    if version >= min_version:
                        return python_path
            except Exception:
                continue

    return None


def check_binary_compatibility(platform_info: PlatformInfo) -> bool:
    """Check if pre-built binaries are available for this platform.

    Args:
        platform_info: Platform information

    Returns:
        True if binaries are available
    """
    supported_platforms = {
        "darwin-arm64",
        "darwin-x86_64",
        "linux-x86_64",
        "rocky-x86_64",
    }

    return platform_info.platform_string in supported_platforms


def get_binary_url(platform_info: PlatformInfo, version: str = "latest") -> str:
    """Get download URL for platform binary.

    Args:
        platform_info: Platform information
        version: Version tag (default: "latest")

    Returns:
        URL to download binary
    """
    base_url = "https://github.com/user/dag-harness/releases"
    binary_name = f"harness-{platform_info.platform_string}"

    if version == "latest":
        return f"{base_url}/latest/download/{binary_name}"
    else:
        return f"{base_url}/download/{version}/{binary_name}"


def get_install_path() -> str:
    """Get recommended installation path for binaries.

    Returns:
        Path to install binaries
    """
    # Check for user-local bin directory
    local_bin = os.path.expanduser("~/.local/bin")

    # On macOS, also consider /usr/local/bin if writable
    if platform.system() == "Darwin":
        usr_local_bin = "/usr/local/bin"
        if os.access(usr_local_bin, os.W_OK):
            return usr_local_bin

    return local_bin
