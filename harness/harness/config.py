"""
Harness configuration management.

Loads configuration from:
1. Environment variables
2. Config files (harness.yml, .claude/box-up-role/config.yml)
3. Defaults
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


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
    discord_webhook_url: Optional[str] = None
    email_recipient: Optional[str] = None
    email_from: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    enabled: bool = False


@dataclass
class HarnessConfig:
    """Main harness configuration."""
    db_path: str = "harness.db"
    repo_root: str = "."
    gitlab: GitLabConfig = field(default_factory=GitLabConfig)
    worktree: WorktreeConfig = field(default_factory=WorktreeConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    # Wave definitions
    waves: dict[int, dict] = field(default_factory=lambda: {
        0: {"name": "Foundation", "roles": ["common"]},
        1: {
            "name": "Infrastructure Foundation",
            "roles": ["windows_prerequisites", "ems_registry_urls", "iis-config"]
        },
        2: {
            "name": "Core Platform",
            "roles": ["ems_platform_services", "ems_web_app", "database_clone", "ems_download_artifacts"]
        },
        3: {
            "name": "Web Applications",
            "roles": ["ems_master_calendar", "ems_master_calendar_api", "ems_campus_webservice", "ems_desktop_deploy"]
        },
        4: {
            "name": "Supporting Services",
            "roles": ["grafana_alloy_windows", "email_infrastructure", "hrtk_protected_users", "ems_environment_util"]
        }
    })

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "HarnessConfig":
        """Load configuration from file and environment."""
        config = cls()

        # Try to find config file
        paths_to_try = [
            config_path,
            "harness.yml",
            "harness.yaml",
            ".claude/skills/box-up-role/config.yml",
            ".claude/box-up-role/config.yml"
        ]

        for path in paths_to_try:
            if path and Path(path).exists():
                config = cls._load_from_file(path)
                break

        # Override with environment variables
        config._load_from_env()

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
                default_iteration=gl.get("default_iteration", config.gitlab.default_iteration)
            )

        if "worktree" in data:
            wt = data["worktree"]
            config.worktree = WorktreeConfig(
                base_path=wt.get("base_path", config.worktree.base_path),
                branch_prefix=wt.get("branch_prefix", config.worktree.branch_prefix)
            )

        if "testing" in data:
            t = data["testing"]
            config.testing = TestingConfig(
                molecule_required=t.get("molecule_required", config.testing.molecule_required),
                pytest_required=t.get("pytest_required", config.testing.pytest_required),
                deploy_target=t.get("deploy_target", config.testing.deploy_target),
                molecule_timeout=t.get("molecule_timeout", config.testing.molecule_timeout),
                pytest_timeout=t.get("pytest_timeout", config.testing.pytest_timeout)
            )

        if "notifications" in data:
            n = data["notifications"]
            config.notifications = NotificationConfig(
                discord_webhook_url=n.get("discord_webhook_url"),
                email_recipient=n.get("email_recipient"),
                enabled=n.get("enabled", False)
            )

        if "waves" in data:
            config.waves = data["waves"]

        return config

    def _load_from_env(self) -> None:
        """Override configuration from environment variables."""
        if os.environ.get("HARNESS_DB_PATH"):
            self.db_path = os.environ["HARNESS_DB_PATH"]

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

    def get_wave_for_role(self, role_name: str) -> tuple[int, str]:
        """Get wave number and name for a role."""
        for wave_num, wave_info in self.waves.items():
            if role_name in wave_info.get("roles", []):
                return wave_num, wave_info.get("name", f"Wave {wave_num}")
        return 0, "Unassigned"

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
                "default_iteration": self.gitlab.default_iteration
            },
            "worktree": {
                "base_path": self.worktree.base_path,
                "branch_prefix": self.worktree.branch_prefix
            },
            "testing": {
                "molecule_required": self.testing.molecule_required,
                "pytest_required": self.testing.pytest_required,
                "deploy_target": self.testing.deploy_target,
                "molecule_timeout": self.testing.molecule_timeout,
                "pytest_timeout": self.testing.pytest_timeout
            },
            "notifications": {
                "discord_webhook_url": self.notifications.discord_webhook_url,
                "email_recipient": self.notifications.email_recipient,
                "enabled": self.notifications.enabled
            },
            "waves": self.waves
        }

    def save(self, path: str = "harness.yml") -> None:
        """Save configuration to file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
