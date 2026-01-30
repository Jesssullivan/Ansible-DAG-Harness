"""Pytest fixtures for harness tests."""

import pytest
import tempfile
from pathlib import Path
from typing import Generator

from harness.db.state import StateDB
from harness.db.models import (
    Role, RoleDependency, DependencyType, Credential,
    Worktree, WorktreeStatus, TestType
)
from harness.config import HarnessConfig


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_harness.db"


@pytest.fixture
def db(temp_db_path: Path) -> Generator[StateDB, None, None]:
    """Create a fresh database for each test."""
    database = StateDB(temp_db_path)
    yield database
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def in_memory_db() -> Generator[StateDB, None, None]:
    """Create an in-memory database for fast tests."""
    database = StateDB(":memory:")
    yield database


@pytest.fixture
def db_with_roles(db: StateDB) -> StateDB:
    """Database pre-populated with test roles."""
    roles = [
        Role(name="common", wave=0, has_molecule_tests=True),
        Role(name="sql_server_2022", wave=2, has_molecule_tests=True),
        Role(name="sql_management_studio", wave=2, has_molecule_tests=False),
        Role(name="ems_web_app", wave=2, has_molecule_tests=True),
        Role(name="ems_platform_services", wave=3, has_molecule_tests=True),
    ]
    for role in roles:
        db.upsert_role(role)

    # Add dependencies: sql_server_2022 -> common
    common = db.get_role("common")
    sql_server = db.get_role("sql_server_2022")
    sql_mgmt = db.get_role("sql_management_studio")
    ems_web = db.get_role("ems_web_app")
    ems_platform = db.get_role("ems_platform_services")

    # sql_server_2022 depends on common
    db.add_dependency(RoleDependency(
        role_id=sql_server.id,
        depends_on_id=common.id,
        dependency_type=DependencyType.EXPLICIT
    ))

    # sql_management_studio depends on common and sql_server_2022
    db.add_dependency(RoleDependency(
        role_id=sql_mgmt.id,
        depends_on_id=common.id,
        dependency_type=DependencyType.EXPLICIT
    ))
    db.add_dependency(RoleDependency(
        role_id=sql_mgmt.id,
        depends_on_id=sql_server.id,
        dependency_type=DependencyType.EXPLICIT
    ))

    # ems_web_app depends on common
    db.add_dependency(RoleDependency(
        role_id=ems_web.id,
        depends_on_id=common.id,
        dependency_type=DependencyType.EXPLICIT
    ))

    # ems_platform_services depends on ems_web_app
    db.add_dependency(RoleDependency(
        role_id=ems_platform.id,
        depends_on_id=ems_web.id,
        dependency_type=DependencyType.EXPLICIT
    ))

    return db


@pytest.fixture
def db_with_credentials(db_with_roles: StateDB) -> StateDB:
    """Database with roles and credentials."""
    common = db_with_roles.get_role("common")
    sql_server = db_with_roles.get_role("sql_server_2022")

    db_with_roles.add_credential(Credential(
        role_id=common.id,
        entry_name="ansible-self",
        purpose="WinRM authentication",
        is_base58=True
    ))

    db_with_roles.add_credential(Credential(
        role_id=sql_server.id,
        entry_name="dev-sql-sa",
        purpose="SQL Server SA password",
        is_base58=False
    ))

    db_with_roles.add_credential(Credential(
        role_id=sql_server.id,
        entry_name="test-windows-admin",
        purpose="Admin access",
        is_base58=False
    ))

    return db_with_roles


@pytest.fixture
def db_with_worktrees(db_with_roles: StateDB) -> StateDB:
    """Database with roles and worktrees."""
    common = db_with_roles.get_role("common")
    sql_server = db_with_roles.get_role("sql_server_2022")

    db_with_roles.upsert_worktree(Worktree(
        role_id=common.id,
        path="../.worktrees/sid-common",
        branch="sid/common",
        commits_ahead=2,
        commits_behind=0,
        uncommitted_changes=0,
        status=WorktreeStatus.ACTIVE
    ))

    db_with_roles.upsert_worktree(Worktree(
        role_id=sql_server.id,
        path="../.worktrees/sid-sql_server_2022",
        branch="sid/sql_server_2022",
        commits_ahead=0,
        commits_behind=5,
        uncommitted_changes=3,
        status=WorktreeStatus.DIRTY
    ))

    return db_with_roles


@pytest.fixture
def config(temp_db_path: Path) -> HarnessConfig:
    """Create test configuration."""
    return HarnessConfig(
        db_path=str(temp_db_path),
    )


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "pbt: mark test as property-based test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow-running")
