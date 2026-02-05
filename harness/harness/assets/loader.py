"""Asset loader using importlib.resources for bundled .claude assets.

Provides utilities to read bundled assets from the installed wheel,
making them available for deployment during `harness init`.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path


def _assets_root() -> importlib.resources.abc.Traversable:
    """Return the traversable root of the bundled assets."""
    return importlib.resources.files("harness.assets.claude")


def read_text(relative_path: str) -> str:
    """Read a text asset from the bundle.

    Args:
        relative_path: Path relative to assets/claude/, e.g. "hooks/validate-box-up-env.sh"

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError: If the asset does not exist.
    """
    parts = relative_path.split("/")
    resource = _assets_root()
    for part in parts:
        resource = resource.joinpath(part)
    try:
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError) as exc:
        raise FileNotFoundError(f"Bundled asset not found: {relative_path}") from exc


def read_bytes(relative_path: str) -> bytes:
    """Read a binary asset from the bundle.

    Args:
        relative_path: Path relative to assets/claude/

    Returns:
        File contents as bytes.
    """
    parts = relative_path.split("/")
    resource = _assets_root()
    for part in parts:
        resource = resource.joinpath(part)
    try:
        return resource.read_bytes()
    except (FileNotFoundError, TypeError) as exc:
        raise FileNotFoundError(f"Bundled asset not found: {relative_path}") from exc


def list_assets(relative_dir: str = "") -> list[str]:
    """List assets under a directory in the bundle.

    Args:
        relative_dir: Directory relative to assets/claude/, e.g. "hooks"

    Returns:
        List of filenames in the directory.
    """
    resource = _assets_root()
    if relative_dir:
        for part in relative_dir.split("/"):
            resource = resource.joinpath(part)
    try:
        return [item.name for item in resource.iterdir() if not item.name.startswith("__")]
    except (FileNotFoundError, TypeError):
        return []


def deploy_file(relative_path: str, dest: Path, executable: bool = False) -> None:
    """Deploy a bundled asset to a destination path.

    Args:
        relative_path: Path relative to assets/claude/
        dest: Destination file path.
        executable: If True, set the executable bit.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = read_text(relative_path)
    dest.write_text(content, encoding="utf-8")
    if executable:
        dest.chmod(dest.stat().st_mode | 0o111)


def deploy_directory(
    relative_dir: str, dest_dir: Path, executable_extensions: set[str] | None = None
) -> list[Path]:
    """Deploy all assets from a bundled directory to a destination.

    Args:
        relative_dir: Directory relative to assets/claude/, e.g. "hooks"
        dest_dir: Destination directory.
        executable_extensions: File extensions to make executable (e.g. {".sh", ".py"}).

    Returns:
        List of deployed file paths.
    """
    if executable_extensions is None:
        executable_extensions = {".sh", ".py"}

    deployed = []
    resource = _assets_root()
    if relative_dir:
        for part in relative_dir.split("/"):
            resource = resource.joinpath(part)

    dest_dir.mkdir(parents=True, exist_ok=True)

    for item in resource.iterdir():
        if item.name.startswith("__"):
            continue
        dest_path = dest_dir / item.name
        if item.is_file():
            dest_path.write_text(item.read_text(encoding="utf-8"), encoding="utf-8")
            if any(item.name.endswith(ext) for ext in executable_extensions):
                dest_path.chmod(dest_path.stat().st_mode | 0o111)
            deployed.append(dest_path)
        elif item.is_dir():
            sub_deployed = _deploy_directory_recursive(item, dest_path, executable_extensions)
            deployed.extend(sub_deployed)

    return deployed


def _deploy_directory_recursive(
    resource: importlib.resources.abc.Traversable,
    dest_dir: Path,
    executable_extensions: set[str],
) -> list[Path]:
    """Recursively deploy a traversable directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    deployed = []

    for item in resource.iterdir():
        if item.name.startswith("__"):
            continue
        dest_path = dest_dir / item.name
        if item.is_file():
            dest_path.write_text(item.read_text(encoding="utf-8"), encoding="utf-8")
            if any(item.name.endswith(ext) for ext in executable_extensions):
                dest_path.chmod(dest_path.stat().st_mode | 0o111)
            deployed.append(dest_path)
        elif item.is_dir():
            sub_deployed = _deploy_directory_recursive(item, dest_path, executable_extensions)
            deployed.extend(sub_deployed)

    return deployed
