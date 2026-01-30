"""Path resolution and validation for harness bootstrap.

This module handles:
- Detecting and validating project paths
- Resolving roles directory location
- Setting up worktree base paths
- Database path configuration
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class PathStatus(Enum):
    """Status of a path check."""
    VALID = "valid"
    NOT_FOUND = "not_found"
    INVALID = "invalid"
    WRITABLE = "writable"
    NOT_WRITABLE = "not_writable"


@dataclass
class PathResult:
    """Result of checking a single path."""
    name: str
    path: Optional[Path]
    status: PathStatus
    description: str
    error: Optional[str] = None
    auto_detected: bool = False


@dataclass
class PathCheckResult:
    """Result of checking all paths."""
    all_valid: bool
    paths: list[PathResult] = field(default_factory=list)

    def get_path(self, name: str) -> Optional[Path]:
        """Get a path by name."""
        for p in self.paths:
            if p.name == name:
                return p.path
        return None

    @property
    def missing(self) -> list[str]:
        """Get list of missing/invalid paths."""
        return [
            p.name for p in self.paths
            if p.status in (PathStatus.NOT_FOUND, PathStatus.INVALID, PathStatus.NOT_WRITABLE)
        ]


class PathResolver:
    """Resolves and validates paths for harness operation.

    Detects:
    - Project root (containing harness/)
    - Git repository root
    - Ansible roles directory
    - Worktree base path
    - Database path
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize path resolver.

        Args:
            project_root: Override for project root detection
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()

    def check_all(self) -> PathCheckResult:
        """Check all required paths.

        Returns:
            PathCheckResult with status of all paths
        """
        results = []

        # Check project root
        results.append(self._check_project_root())

        # Check git repository
        results.append(self._check_git_root())

        # Check harness package
        results.append(self._check_harness_package())

        # Check database path
        results.append(self._check_database_path())

        # Check worktree base path
        results.append(self._check_worktree_base())

        # Check .claude directory
        results.append(self._check_claude_dir())

        all_valid = all(
            r.status in (PathStatus.VALID, PathStatus.WRITABLE)
            for r in results
        )

        return PathCheckResult(all_valid=all_valid, paths=results)

    def _check_project_root(self) -> PathResult:
        """Check project root directory."""
        harness_dir = self.project_root / "harness"
        if harness_dir.exists() and (harness_dir / "pyproject.toml").exists():
            return PathResult(
                name="project_root",
                path=self.project_root,
                status=PathStatus.VALID,
                description="Project root directory",
                auto_detected=True
            )

        # Try parent directories
        for parent in self.project_root.parents:
            harness_dir = parent / "harness"
            if harness_dir.exists() and (harness_dir / "pyproject.toml").exists():
                self.project_root = parent
                return PathResult(
                    name="project_root",
                    path=parent,
                    status=PathStatus.VALID,
                    description="Project root directory",
                    auto_detected=True
                )

        return PathResult(
            name="project_root",
            path=self.project_root,
            status=PathStatus.INVALID,
            description="Project root directory",
            error="Could not find harness/ with pyproject.toml"
        )

    def _check_git_root(self) -> PathResult:
        """Check git repository root."""
        git_dir = self.project_root / ".git"

        if git_dir.exists():
            return PathResult(
                name="git_root",
                path=self.project_root,
                status=PathStatus.VALID,
                description="Git repository root",
                auto_detected=True
            )

        # Check parent directories for git root
        for parent in self.project_root.parents:
            if (parent / ".git").exists():
                return PathResult(
                    name="git_root",
                    path=parent,
                    status=PathStatus.VALID,
                    description="Git repository root",
                    auto_detected=True
                )

        return PathResult(
            name="git_root",
            path=None,
            status=PathStatus.NOT_FOUND,
            description="Git repository root",
            error="Not inside a git repository"
        )

    def _check_harness_package(self) -> PathResult:
        """Check harness Python package."""
        harness_pkg = self.project_root / "harness" / "harness"

        if harness_pkg.exists() and (harness_pkg / "__init__.py").exists():
            return PathResult(
                name="harness_package",
                path=harness_pkg,
                status=PathStatus.VALID,
                description="Harness Python package",
                auto_detected=True
            )

        return PathResult(
            name="harness_package",
            path=harness_pkg,
            status=PathStatus.NOT_FOUND,
            description="Harness Python package",
            error="harness/harness/__init__.py not found"
        )

    def _check_database_path(self) -> PathResult:
        """Check database path is writable."""
        db_path = Path(os.environ.get("HARNESS_DB_PATH", ""))

        if not db_path or str(db_path) == "":
            # Default to harness directory
            db_path = self.project_root / "harness" / "harness.db"

        db_path = db_path.resolve()
        db_dir = db_path.parent

        # Check if parent directory exists and is writable
        if db_dir.exists():
            if os.access(db_dir, os.W_OK):
                return PathResult(
                    name="database",
                    path=db_path,
                    status=PathStatus.WRITABLE,
                    description="SQLite database file",
                    auto_detected=not os.environ.get("HARNESS_DB_PATH")
                )
            else:
                return PathResult(
                    name="database",
                    path=db_path,
                    status=PathStatus.NOT_WRITABLE,
                    description="SQLite database file",
                    error=f"Directory {db_dir} is not writable"
                )

        return PathResult(
            name="database",
            path=db_path,
            status=PathStatus.NOT_FOUND,
            description="SQLite database file",
            error=f"Directory {db_dir} does not exist"
        )

    def _check_worktree_base(self) -> PathResult:
        """Check worktree base path."""
        # Default to ../worktrees relative to git root
        git_root = self._find_git_root()

        if git_root:
            default_base = git_root.parent / "worktrees"
        else:
            default_base = self.project_root.parent / "worktrees"

        worktree_base = Path(os.environ.get("WORKTREE_BASE", str(default_base)))

        if worktree_base.exists():
            if os.access(worktree_base, os.W_OK):
                return PathResult(
                    name="worktree_base",
                    path=worktree_base,
                    status=PathStatus.WRITABLE,
                    description="Git worktree base directory",
                    auto_detected=not os.environ.get("WORKTREE_BASE")
                )
            else:
                return PathResult(
                    name="worktree_base",
                    path=worktree_base,
                    status=PathStatus.NOT_WRITABLE,
                    description="Git worktree base directory",
                    error=f"Directory {worktree_base} is not writable"
                )

        # Directory doesn't exist but parent might be writable
        parent = worktree_base.parent
        if parent.exists() and os.access(parent, os.W_OK):
            return PathResult(
                name="worktree_base",
                path=worktree_base,
                status=PathStatus.VALID,
                description="Git worktree base directory (will be created)",
                auto_detected=not os.environ.get("WORKTREE_BASE")
            )

        return PathResult(
            name="worktree_base",
            path=worktree_base,
            status=PathStatus.NOT_WRITABLE,
            description="Git worktree base directory",
            error=f"Cannot create directory at {worktree_base}"
        )

    def _check_claude_dir(self) -> PathResult:
        """Check .claude directory."""
        claude_dir = self.project_root / ".claude"

        if claude_dir.exists():
            has_settings = (claude_dir / "settings.json").exists()
            has_hooks = (claude_dir / "hooks").exists()

            if has_settings or has_hooks:
                return PathResult(
                    name="claude_dir",
                    path=claude_dir,
                    status=PathStatus.VALID,
                    description="MCP client integration directory",
                    auto_detected=True
                )
            else:
                return PathResult(
                    name="claude_dir",
                    path=claude_dir,
                    status=PathStatus.INVALID,
                    description="MCP client integration directory",
                    error="Missing settings.json or hooks/"
                )

        # Will be created during install
        return PathResult(
            name="claude_dir",
            path=claude_dir,
            status=PathStatus.VALID,
            description="MCP client integration directory (will be created)",
            auto_detected=True
        )

    def _find_git_root(self) -> Optional[Path]:
        """Find git repository root."""
        if (self.project_root / ".git").exists():
            return self.project_root

        for parent in self.project_root.parents:
            if (parent / ".git").exists():
                return parent

        return None

    def detect_repo_root(self) -> Optional[Path]:
        """Detect the Ansible repository root (for role discovery).

        Returns:
            Path to repo containing ansible/roles, or None
        """
        # Check environment variable first
        env_root = os.environ.get("REPO_ROOT")
        if env_root:
            path = Path(env_root)
            if (path / "ansible" / "roles").exists():
                return path

        # Check current project
        if (self.project_root / "ansible" / "roles").exists():
            return self.project_root

        # Check parent directories
        for parent in self.project_root.parents:
            if (parent / "ansible" / "roles").exists():
                return parent

        return None

    def get_recommended_config(self) -> dict:
        """Get recommended path configuration.

        Returns:
            Dict with recommended path settings
        """
        check_result = self.check_all()
        repo_root = self.detect_repo_root()

        config = {
            "db_path": str(check_result.get_path("database") or "./harness/harness.db"),
            "project_root": str(check_result.get_path("project_root") or "."),
        }

        if repo_root:
            config["repo_root"] = str(repo_root)

        worktree_base = check_result.get_path("worktree_base")
        if worktree_base:
            config["worktree_base"] = str(worktree_base)

        return config
