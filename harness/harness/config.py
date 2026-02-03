"""
Harness configuration management.

Loads configuration from:
1. Environment variables (HARNESS_* prefix)
2. Config files (harness.yml, .claude/box-up-role/config.yml)
3. Defaults
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def find_repo_python(repo_root: Path) -> str:
    """
    Find the Python interpreter for a repository.

    Search order:
    1. repo_root/.venv/bin/python
    2. repo_root/venv/bin/python
    3. UV_PROJECT_ENVIRONMENT if set
    4. sys.executable (fallback)

    Args:
        repo_root: Root directory of the target repository

    Returns:
        Path to Python interpreter
    """
    venv_paths = [
        repo_root / ".venv" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python3",
        repo_root / "venv" / "bin" / "python",
        repo_root / "venv" / "bin" / "python3",
    ]

    for venv_python in venv_paths:
        if venv_python.exists():
            return str(venv_python)

    # Check UV_PROJECT_ENVIRONMENT
    uv_env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if uv_env:
        uv_python = Path(uv_env) / "bin" / "python"
        if uv_python.exists():
            return str(uv_python)

    # Fallback to current interpreter
    return sys.executable


@dataclass
class GitLabConfig:
    """GitLab configuration."""

    project_path: str = "bates-ils/projects/ems/ems-mono"
    group_path: str = "bates-ils"
    default_assignee: str = "jsullivan2"
    default_labels: list[str] = field(default_factory=lambda: ["role", "ansible", "molecule"])
    default_iteration: str = "EMS Upgrade"


@dataclass
class WorktreeConfig:
    """Git worktree configuration."""

    base_path: str = ".."
    branch_prefix: str = "sid/"


@dataclass
class TestingConfig:
    """Testing configuration."""

    molecule_required: bool = True
    pytest_required: bool = True
    deploy_target: str = "vmnode852"
    molecule_timeout: int = 600
    pytest_timeout: int = 300


@dataclass
class NotificationConfig:
    """Notification configuration."""

    discord_webhook_url: str | None = None
    email_recipient: str | None = None
    email_from: str | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_username: str | None = None
    smtp_password: str | None = None
    enabled: bool = False


@dataclass
class ObservabilityConfig:
    """
    Observability configuration for LangSmith tracing.

    Environment variables:
    - LANGCHAIN_TRACING_V2: Set to "true" to enable LangSmith tracing
    - LANGCHAIN_PROJECT: Project name in LangSmith (default: "dag-harness")
    - LANGCHAIN_API_KEY: LangSmith API key (required if tracing enabled)
    - HARNESS_ANONYMIZE_SENSITIVE: Set to "false" to disable sensitive data anonymization
    """

    langsmith_enabled: bool = False  # Set from LANGCHAIN_TRACING_V2
    langsmith_project: str = "dag-harness"  # Set from LANGCHAIN_PROJECT
    anonymize_sensitive: bool = True  # Anonymize sensitive data before sending

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Load observability config from environment variables."""
        return cls(
            langsmith_enabled=os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true",
            langsmith_project=os.environ.get("LANGCHAIN_PROJECT", "dag-harness"),
            anonymize_sensitive=os.environ.get("HARNESS_ANONYMIZE_SENSITIVE", "true").lower()
            != "false",
        )


@dataclass
class HarnessConfig:
    """Main harness configuration."""

    db_path: str = "harness.db"
    repo_root: str = "."
    gitlab: GitLabConfig = field(default_factory=GitLabConfig)
    worktree: WorktreeConfig = field(default_factory=WorktreeConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # Wave definitions
    waves: dict[int, dict] = field(
        default_factory=lambda: {
            0: {"name": "Foundation", "roles": ["common"]},
            1: {
                "name": "Infrastructure Foundation",
                "roles": ["windows_prerequisites", "ems_registry_urls", "iis-config"],
            },
            2: {
                "name": "Core Platform",
                "roles": [
                    "ems_platform_services",
                    "ems_web_app",
                    "database_clone",
                    "ems_download_artifacts",
                ],
            },
            3: {
                "name": "Web Applications",
                "roles": [
                    "ems_master_calendar",
                    "ems_master_calendar_api",
                    "ems_campus_webservice",
                    "ems_desktop_deploy",
                ],
            },
            4: {
                "name": "Supporting Services",
                "roles": [
                    "grafana_alloy_windows",
                    "email_infrastructure",
                    "hrtk_protected_users",
                    "ems_environment_util",
                ],
            },
        }
    )

    # Cached repo_python path (set after load)
    _repo_python: str | None = field(default=None, repr=False)

    def __post_init__(self):
        """Resolve repo_root to absolute path after initialization."""
        self.repo_root = str(Path(self.repo_root).resolve())
        self._repo_python = None  # Will be computed lazily

    @property
    def repo_python(self) -> str:
        """Get the Python interpreter for the target repository."""
        if self._repo_python is None:
            self._repo_python = find_repo_python(Path(self.repo_root))
        return self._repo_python

    @classmethod
    def load(cls, config_path: str | None = None) -> "HarnessConfig":
        """Load configuration from file and environment."""
        config = cls()

        # Check HARNESS_CONFIG environment variable first
        if not config_path:
            config_path = os.environ.get("HARNESS_CONFIG")

        # Try to find config file
        paths_to_try = [
            config_path,
            "harness.yml",
            "harness.yaml",
            ".claude/skills/box-up-role/config.yml",
            ".claude/box-up-role/config.yml",
        ]

        # Also search up from CWD to find repo root
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents)[:5]:
            paths_to_try.append(str(parent / "harness.yml"))
            paths_to_try.append(str(parent / ".harness" / "config.yml"))

        for path in paths_to_try:
            if path and Path(path).exists():
                config = cls._load_from_file(path)
                break

        # Override with environment variables
        config._load_from_env()

        # Ensure repo_root is absolute
        config.repo_root = str(Path(config.repo_root).resolve())

        return config

    @classmethod
    def _load_from_file(cls, path: str) -> "HarnessConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        if "db_path" in data:
            config.db_path = data["db_path"]
        if "repo_root" in data:
            config.repo_root = data["repo_root"]

        if "gitlab" in data:
            gl = data["gitlab"]
            config.gitlab = GitLabConfig(
                project_path=gl.get("project_path", config.gitlab.project_path),
                group_path=gl.get("group_path", config.gitlab.group_path),
                default_assignee=gl.get("default_assignee", config.gitlab.default_assignee),
                default_labels=gl.get("default_labels", config.gitlab.default_labels),
                default_iteration=gl.get("default_iteration", config.gitlab.default_iteration),
            )

        if "worktree" in data:
            wt = data["worktree"]
            config.worktree = WorktreeConfig(
                base_path=wt.get("base_path", config.worktree.base_path),
                branch_prefix=wt.get("branch_prefix", config.worktree.branch_prefix),
            )

        if "testing" in data:
            t = data["testing"]
            config.testing = TestingConfig(
                molecule_required=t.get("molecule_required", config.testing.molecule_required),
                pytest_required=t.get("pytest_required", config.testing.pytest_required),
                deploy_target=t.get("deploy_target", config.testing.deploy_target),
                molecule_timeout=t.get("molecule_timeout", config.testing.molecule_timeout),
                pytest_timeout=t.get("pytest_timeout", config.testing.pytest_timeout),
            )

        if "notifications" in data:
            n = data["notifications"]
            config.notifications = NotificationConfig(
                discord_webhook_url=n.get("discord_webhook_url"),
                email_recipient=n.get("email_recipient"),
                enabled=n.get("enabled", False),
            )

        if "waves" in data:
            config.waves = data["waves"]

        return config

    def _load_from_env(self) -> None:
        """Override configuration from environment variables."""
        if os.environ.get("HARNESS_DB_PATH"):
            self.db_path = os.environ["HARNESS_DB_PATH"]

        # NEW: Support HARNESS_REPO_ROOT environment variable
        if os.environ.get("HARNESS_REPO_ROOT"):
            self.repo_root = os.environ["HARNESS_REPO_ROOT"]

        if os.environ.get("GITLAB_PROJECT"):
            self.gitlab.project_path = os.environ["GITLAB_PROJECT"]

        if os.environ.get("GITLAB_GROUP"):
            self.gitlab.group_path = os.environ["GITLAB_GROUP"]

        if os.environ.get("DISCORD_WEBHOOK_URL"):
            self.notifications.discord_webhook_url = os.environ["DISCORD_WEBHOOK_URL"]
            self.notifications.enabled = True

        if os.environ.get("EMAIL_RECIPIENT"):
            self.notifications.email_recipient = os.environ["EMAIL_RECIPIENT"]
            self.notifications.enabled = True

        # Load observability config from environment
        self.observability = ObservabilityConfig.from_env()

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        repo_root = Path(self.repo_root)
        if not repo_root.exists():
            errors.append(f"repo_root does not exist: {self.repo_root}")
        elif not (repo_root / "ansible" / "roles").exists():
            errors.append(f"No ansible/roles directory in repo_root: {self.repo_root}")

        # Validate db_path is writable
        db_path = Path(self.db_path)
        if not db_path.is_absolute():
            db_path = repo_root / db_path
        db_parent = db_path.parent
        if not db_parent.exists():
            try:
                db_parent.mkdir(parents=True)
            except OSError as e:
                errors.append(f"Cannot create database directory {db_parent}: {e}")

        return errors

    def get_wave_for_role(self, role_name: str) -> tuple[int, str]:
        """Get wave number and name for a role."""
        for wave_num, wave_info in self.waves.items():
            if role_name in wave_info.get("roles", []):
                return wave_num, wave_info.get("name", f"Wave {wave_num}")
        return 0, "Unassigned"

    def is_foundation_role(self, role_name: str) -> bool:
        """Check if a role is a foundation role (Wave 0 with no dependencies)."""
        wave_num, _ = self.get_wave_for_role(role_name)
        return wave_num == 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "db_path": self.db_path,
            "repo_root": self.repo_root,
            "gitlab": {
                "project_path": self.gitlab.project_path,
                "group_path": self.gitlab.group_path,
                "default_assignee": self.gitlab.default_assignee,
                "default_labels": self.gitlab.default_labels,
                "default_iteration": self.gitlab.default_iteration,
            },
            "worktree": {
                "base_path": self.worktree.base_path,
                "branch_prefix": self.worktree.branch_prefix,
            },
            "testing": {
                "molecule_required": self.testing.molecule_required,
                "pytest_required": self.testing.pytest_required,
                "deploy_target": self.testing.deploy_target,
                "molecule_timeout": self.testing.molecule_timeout,
                "pytest_timeout": self.testing.pytest_timeout,
            },
            "notifications": {
                "discord_webhook_url": self.notifications.discord_webhook_url,
                "email_recipient": self.notifications.email_recipient,
                "enabled": self.notifications.enabled,
            },
            "observability": {
                "langsmith_enabled": self.observability.langsmith_enabled,
                "langsmith_project": self.observability.langsmith_project,
                "anonymize_sensitive": self.observability.anonymize_sensitive,
            },
            "waves": self.waves,
        }

    def save(self, path: str = "harness.yml") -> None:
        """Save configuration to file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
