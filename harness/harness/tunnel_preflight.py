"""
SSH tunnel pre-flight checks for Windows VM connectivity.

Handles:
- Detection of 8xx series hosts requiring tunnels (vmnode851/852/876)
- Tunnel availability checks via port probing
- Tunnel startup via EMS project's scripts/start-winrm-tunnels.sh
- Graceful degradation with clear error messages

Architecture:
    Local -> Tailscale -> xoxd-bates -> Bates Network -> Windows VMs

Port Mapping:
    localhost:15851 -> vmnode851:5986
    localhost:15852 -> vmnode852:5986
    localhost:15876 -> vmnode876:5986

Usage:
    from harness.tunnel_preflight import ensure_tunnel_connectivity

    success, message = ensure_tunnel_connectivity(
        host="vmnode852",
        repo_root=Path("/path/to/ems"),
        auto_start=True,
    )
    if not success:
        # Handle tunnel unavailable
        pass
"""

from __future__ import annotations

import logging
import socket
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# 8xx series hosts that require SSH tunneling
TUNNELED_HOSTS: frozenset[str] = frozenset({"vmnode851", "vmnode852", "vmnode876"})

# Port mapping: host -> localhost port for tunnel
TUNNEL_PORTS: dict[str, int] = {
    "vmnode851": 15851,
    "vmnode852": 15852,
    "vmnode876": 15876,
}


def is_tunneled_host(host: str) -> bool:
    """
    Check if host requires SSH tunneling (8xx series).

    Args:
        host: Hostname to check (e.g., "vmnode852").

    Returns:
        True if host is in TUNNELED_HOSTS set.

    Examples:
        >>> is_tunneled_host("vmnode852")
        True
        >>> is_tunneled_host("vmnode520")
        False
    """
    return host in TUNNELED_HOSTS


def get_tunnel_port(host: str) -> int | None:
    """
    Get the localhost port for a tunneled host.

    Args:
        host: Hostname (e.g., "vmnode852").

    Returns:
        Port number (e.g., 15852) or None if not a tunneled host.
    """
    return TUNNEL_PORTS.get(host)


def check_tunnel_available(host: str, timeout: float = 1.0) -> bool:
    """
    Check if tunnel is running by probing localhost port.

    Uses a TCP socket connection attempt to verify the tunnel port
    is listening and accepting connections.

    Args:
        host: Target hostname (e.g., "vmnode852").
        timeout: Socket timeout in seconds.

    Returns:
        True if the tunnel port is open and responding.

    Examples:
        >>> check_tunnel_available("vmnode852")
        True  # If tunnel is running
        >>> check_tunnel_available("vmnode520")
        False  # Not a tunneled host
    """
    port = TUNNEL_PORTS.get(host)
    if not port:
        return False

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex(("localhost", port))
            return result == 0
    except OSError as e:
        logger.debug(f"Socket error checking tunnel for {host}: {e}")
        return False
    except Exception as e:
        logger.debug(f"Unexpected error checking tunnel for {host}: {e}")
        return False


def start_tunnel(
    host: str,
    repo_root: Path,
    timeout: int = 30,
) -> tuple[bool, str]:
    """
    Start tunnel via EMS project's scripts/start-winrm-tunnels.sh.

    The tunnel script is expected to be in the target repository,
    not in the harness package. This ensures the tunnel configuration
    matches the target project's requirements.

    Args:
        host: Target hostname (e.g., "vmnode852").
        repo_root: Path to the repository root containing the tunnel script.
        timeout: Maximum time to wait for tunnel startup in seconds.

    Returns:
        Tuple of (success: bool, message: str).
        On success, message describes what was started.
        On failure, message contains the error details.

    Examples:
        >>> start_tunnel("vmnode852", Path("/path/to/ems"))
        (True, "Tunnel started for vmnode852 on localhost:15852")
    """
    if host not in TUNNELED_HOSTS:
        return False, f"{host} is not a tunneled host"

    port = TUNNEL_PORTS.get(host)

    # Look for tunnel script in repo
    tunnel_script = repo_root / "scripts" / "start-winrm-tunnels.sh"
    if not tunnel_script.exists():
        # Try alternate locations
        alt_script = repo_root / "start-winrm-tunnels.sh"
        if alt_script.exists():
            tunnel_script = alt_script
        else:
            return False, f"Tunnel script not found: {tunnel_script}"

    if not tunnel_script.is_file():
        return False, f"Tunnel script is not a file: {tunnel_script}"

    try:
        # Run the tunnel script with 'start' command and host argument
        result = subprocess.run(
            ["bash", str(tunnel_script), "start", host],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo_root),
        )

        if result.returncode == 0:
            # Verify tunnel actually started
            # Give it a moment to establish
            import time

            time.sleep(0.5)

            if check_tunnel_available(host):
                logger.info(f"Tunnel started for {host} on localhost:{port}")
                return True, f"Tunnel started for {host} on localhost:{port}"

            # Script succeeded but port not open - might be async startup
            # Give it a bit more time
            for _ in range(5):
                time.sleep(1)
                if check_tunnel_available(host):
                    logger.info(f"Tunnel started for {host} on localhost:{port}")
                    return True, f"Tunnel started for {host} on localhost:{port}"

            return False, f"Tunnel script succeeded but port {port} not responding"

        # Script failed
        error_output = result.stderr.strip() or result.stdout.strip()
        logger.warning(f"Tunnel startup failed for {host}: {error_output}")
        return False, f"Tunnel startup failed: {error_output}"

    except subprocess.TimeoutExpired:
        logger.error(f"Tunnel startup timed out after {timeout}s for {host}")
        return False, f"Tunnel startup timed out after {timeout}s"
    except FileNotFoundError:
        return False, "bash not found in PATH"
    except PermissionError:
        return False, f"Permission denied executing {tunnel_script}"
    except Exception as e:
        logger.error(f"Tunnel startup error for {host}: {e}")
        return False, str(e)


def stop_tunnel(
    host: str,
    repo_root: Path,
    timeout: int = 10,
) -> tuple[bool, str]:
    """
    Stop tunnel for a host via the tunnel script.

    Args:
        host: Target hostname (e.g., "vmnode852").
        repo_root: Path to the repository root containing the tunnel script.
        timeout: Maximum time to wait for tunnel shutdown in seconds.

    Returns:
        Tuple of (success: bool, message: str).
    """
    if host not in TUNNELED_HOSTS:
        return False, f"{host} is not a tunneled host"

    tunnel_script = repo_root / "scripts" / "start-winrm-tunnels.sh"
    if not tunnel_script.exists():
        return False, f"Tunnel script not found: {tunnel_script}"

    try:
        result = subprocess.run(
            ["bash", str(tunnel_script), "stop", host],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo_root),
        )

        if result.returncode == 0:
            return True, f"Tunnel stopped for {host}"

        error_output = result.stderr.strip() or result.stdout.strip()
        return False, f"Tunnel stop failed: {error_output}"

    except subprocess.TimeoutExpired:
        return False, f"Tunnel stop timed out after {timeout}s"
    except Exception as e:
        return False, str(e)


def get_tunnel_status(host: str) -> dict[str, bool | int | None]:
    """
    Get comprehensive tunnel status for a host.

    Args:
        host: Target hostname (e.g., "vmnode852").

    Returns:
        Dict with status information:
        - requires_tunnel: bool - whether host needs tunneling
        - tunnel_port: int | None - localhost port if tunneled
        - tunnel_available: bool - whether tunnel is responding
    """
    requires_tunnel = is_tunneled_host(host)
    port = get_tunnel_port(host)

    return {
        "requires_tunnel": requires_tunnel,
        "tunnel_port": port,
        "tunnel_available": check_tunnel_available(host) if requires_tunnel else True,
    }


def ensure_tunnel_connectivity(
    host: str,
    repo_root: Path,
    auto_start: bool = True,
    startup_timeout: int = 30,
) -> tuple[bool, str]:
    """
    Ensure tunnel connectivity for a host.

    This is the main entry point for tunnel pre-flight checks.
    It checks if the host requires tunneling, verifies connectivity,
    and optionally starts the tunnel if needed.

    Args:
        host: Target hostname (e.g., "vmnode852").
        repo_root: Path to repository root (for locating tunnel script).
        auto_start: If True, attempt to start tunnel if not running.
        startup_timeout: Timeout for tunnel startup in seconds.

    Returns:
        Tuple of (success: bool, message: str).
        success is True if connectivity is established (or not needed).
        message provides details about the result.

    Examples:
        >>> ensure_tunnel_connectivity("vmnode852", Path("/path/to/ems"))
        (True, "Tunnel active on localhost:15852")

        >>> ensure_tunnel_connectivity("vmnode520", Path("/path/to/ems"))
        (True, "vmnode520 does not require tunneling")

        >>> ensure_tunnel_connectivity("vmnode876", Path("/path/to/ems"), auto_start=False)
        (False, "Tunnel not running for vmnode876 (localhost:15876)")
    """
    # Check if host needs tunneling
    if not is_tunneled_host(host):
        logger.debug(f"{host} does not require tunneling (not in 8xx series)")
        return True, f"{host} does not require tunneling"

    port = TUNNEL_PORTS[host]

    # Check if tunnel is already running
    if check_tunnel_available(host):
        logger.debug(f"Tunnel already active for {host} on localhost:{port}")
        return True, f"Tunnel active on localhost:{port}"

    # Tunnel not running - try to start if auto_start enabled
    if auto_start:
        logger.info(f"Tunnel not running for {host}, attempting to start...")
        success, message = start_tunnel(host, repo_root, timeout=startup_timeout)
        if success:
            return True, message
        # Start failed - return the error
        return False, f"Auto-start failed: {message}"

    # auto_start disabled and tunnel not running
    return False, f"Tunnel not running for {host} (localhost:{port})"


__all__ = [
    # Constants
    "TUNNELED_HOSTS",
    "TUNNEL_PORTS",
    # Functions
    "is_tunneled_host",
    "get_tunnel_port",
    "check_tunnel_available",
    "start_tunnel",
    "stop_tunnel",
    "get_tunnel_status",
    "ensure_tunnel_connectivity",
]
