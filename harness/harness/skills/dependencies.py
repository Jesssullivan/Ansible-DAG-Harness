"""Dependency analysis skill for analyzing and managing project dependencies.

This skill provides capabilities for:
- Analyzing project dependencies
- Checking for available updates
- Identifying security vulnerabilities
- Generating dependency graphs
- Finding unused dependencies

Supports Python (pip/uv) and Ansible (collections/roles).
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.skills.base import (
    Skill,
    SkillAction,
    SkillContext,
    SkillResult,
    SkillStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class DependencyInfo:
    """Information about a dependency.

    Attributes:
        name: Package/dependency name
        version: Current version
        required_version: Version constraint from requirements
        source: Where dependency is defined
        is_direct: Whether this is a direct dependency
        dependencies: List of transitive dependencies
        metadata: Additional metadata
    """

    name: str
    version: str | None = None
    required_version: str | None = None
    source: str | None = None
    is_direct: bool = True
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "required_version": self.required_version,
            "source": self.source,
            "is_direct": self.is_direct,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }


@dataclass
class DependencyUpdate:
    """Information about an available update.

    Attributes:
        name: Package name
        current_version: Currently installed version
        latest_version: Latest available version
        update_type: Type of update (major, minor, patch)
        breaking: Whether update may be breaking
        changelog_url: URL to changelog if available
    """

    name: str
    current_version: str
    latest_version: str
    update_type: str = "unknown"
    breaking: bool = False
    changelog_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "update_type": self.update_type,
            "breaking": self.breaking,
            "changelog_url": self.changelog_url,
        }


class DependencySkill(Skill):
    """Skill for dependency analysis and management.

    Provides actions for:
    - analyze_deps: Analyze project dependencies
    - check_updates: Check for available updates
    - find_unused: Find unused dependencies
    - security_check: Check for known vulnerabilities

    Configuration:
        - package_manager: Default package manager (pip, uv)
        - include_transitive: Include transitive dependencies
        - check_security: Enable security checks
    """

    def __init__(
        self,
        enabled: bool = True,
        config: dict | None = None,
    ):
        """Initialize dependency skill.

        Args:
            enabled: Whether skill is active
            config: Optional configuration
        """
        config = config or {}
        config.setdefault("package_manager", "pip")
        config.setdefault("include_transitive", True)
        config.setdefault("check_security", True)
        super().__init__(enabled=enabled, config=config)

    @property
    def name(self) -> str:
        return "dependencies"

    @property
    def description(self) -> str:
        return "Analyze dependencies, check updates, and security"

    def _register_actions(self) -> None:
        """Register dependency actions."""
        self.register_action(
            SkillAction(
                name="analyze_deps",
                description="Analyze project dependencies",
                handler=self._analyze_deps,
                parameters=["path", "include_transitive", "format"],
                required_params=[],
            )
        )

        self.register_action(
            SkillAction(
                name="check_updates",
                description="Check for available updates",
                handler=self._check_updates,
                parameters=["path", "include_prerelease"],
                required_params=[],
            )
        )

        self.register_action(
            SkillAction(
                name="find_unused",
                description="Find potentially unused dependencies",
                handler=self._find_unused,
                parameters=["path", "source_dir"],
                required_params=[],
            )
        )

        self.register_action(
            SkillAction(
                name="security_check",
                description="Check for known security vulnerabilities",
                handler=self._security_check,
                parameters=["path"],
                required_params=[],
            )
        )

        self.register_action(
            SkillAction(
                name="analyze_ansible",
                description="Analyze Ansible role dependencies",
                handler=self._analyze_ansible,
                parameters=["role_path"],
                required_params=["role_path"],
            )
        )

    async def _analyze_deps(
        self,
        context: SkillContext,
        params: dict,
    ) -> SkillResult:
        """Analyze project dependencies.

        Args:
            context: Skill context
            params: Parameters (path, include_transitive, format)

        Returns:
            SkillResult with dependency information
        """
        path = context.working_dir / params.get("path", ".")
        include_transitive = params.get(
            "include_transitive",
            self.config["include_transitive"],
        )

        dependencies: list[DependencyInfo] = []

        # Try to detect project type and parse dependencies
        pyproject_path = path / "pyproject.toml"
        requirements_path = path / "requirements.txt"

        if pyproject_path.exists():
            deps = await self._parse_pyproject(pyproject_path)
            dependencies.extend(deps)

        if requirements_path.exists():
            deps = await self._parse_requirements(requirements_path)
            dependencies.extend(deps)

        # Get installed versions
        installed = await self._get_installed_packages(context)

        # Update dependency info with installed versions
        for dep in dependencies:
            if dep.name in installed:
                dep.version = installed[dep.name]

        # Get transitive dependencies if requested
        if include_transitive:
            transitive = await self._get_transitive_deps(context, dependencies)
            dependencies.extend(transitive)

        return SkillResult.success(
            f"Found {len(dependencies)} dependencies",
            data={
                "dependencies": [d.to_dict() for d in dependencies],
                "direct_count": sum(1 for d in dependencies if d.is_direct),
                "transitive_count": sum(1 for d in dependencies if not d.is_direct),
            },
        )

    async def _parse_pyproject(
        self,
        pyproject_path: Path,
    ) -> list[DependencyInfo]:
        """Parse dependencies from pyproject.toml.

        Args:
            pyproject_path: Path to pyproject.toml

        Returns:
            List of dependencies
        """
        dependencies = []

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)

            # PEP 621 format
            project_deps = data.get("project", {}).get("dependencies", [])
            for dep in project_deps:
                name, version = self._parse_dependency_spec(dep)
                dependencies.append(
                    DependencyInfo(
                        name=name,
                        required_version=version,
                        source="pyproject.toml",
                        is_direct=True,
                    )
                )

            # Optional dependencies
            optional = data.get("project", {}).get("optional-dependencies", {})
            for group, deps in optional.items():
                for dep in deps:
                    name, version = self._parse_dependency_spec(dep)
                    dependencies.append(
                        DependencyInfo(
                            name=name,
                            required_version=version,
                            source=f"pyproject.toml[{group}]",
                            is_direct=True,
                            metadata={"optional_group": group},
                        )
                    )

        except Exception as e:
            logger.warning(f"Failed to parse pyproject.toml: {e}")

        return dependencies

    async def _parse_requirements(
        self,
        requirements_path: Path,
    ) -> list[DependencyInfo]:
        """Parse dependencies from requirements.txt.

        Args:
            requirements_path: Path to requirements.txt

        Returns:
            List of dependencies
        """
        dependencies = []

        try:
            with open(requirements_path) as f:
                for line in f:
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    # Skip options
                    if line.startswith("-"):
                        continue

                    name, version = self._parse_dependency_spec(line)
                    if name:
                        dependencies.append(
                            DependencyInfo(
                                name=name,
                                required_version=version,
                                source="requirements.txt",
                                is_direct=True,
                            )
                        )

        except Exception as e:
            logger.warning(f"Failed to parse requirements.txt: {e}")

        return dependencies

    def _parse_dependency_spec(
        self,
        spec: str,
    ) -> tuple[str, str | None]:
        """Parse a dependency specification.

        Args:
            spec: Dependency spec like "requests>=2.0"

        Returns:
            Tuple of (name, version_constraint)
        """
        # Remove extras
        spec = re.sub(r"\[.*?\]", "", spec)

        # Split on version operators
        match = re.match(r"^([a-zA-Z0-9_-]+)\s*([><=!~].*)?", spec)
        if match:
            return match.group(1).lower(), match.group(2)
        return spec.strip().lower(), None

    async def _get_installed_packages(
        self,
        context: SkillContext,
    ) -> dict[str, str]:
        """Get installed package versions.

        Args:
            context: Skill context

        Returns:
            Dict mapping package name to version
        """
        installed = {}

        try:
            proc = subprocess.run(
                ["pip", "list", "--format=json"],
                cwd=str(context.working_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc.returncode == 0:
                packages = json.loads(proc.stdout)
                for pkg in packages:
                    installed[pkg["name"].lower()] = pkg["version"]

        except Exception as e:
            logger.warning(f"Failed to get installed packages: {e}")

        return installed

    async def _get_transitive_deps(
        self,
        context: SkillContext,
        direct_deps: list[DependencyInfo],
    ) -> list[DependencyInfo]:
        """Get transitive dependencies.

        Args:
            context: Skill context
            direct_deps: List of direct dependencies

        Returns:
            List of transitive dependencies
        """
        transitive = []
        direct_names = {d.name for d in direct_deps}

        try:
            # Use pipdeptree if available
            proc = subprocess.run(
                ["pip", "show"] + list(direct_names),
                cwd=str(context.working_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if proc.returncode == 0:
                current_pkg = None
                for line in proc.stdout.split("\n"):
                    if line.startswith("Name:"):
                        current_pkg = line.split(":", 1)[1].strip().lower()
                    elif line.startswith("Requires:") and current_pkg:
                        requires = line.split(":", 1)[1].strip()
                        if requires:
                            for req in requires.split(","):
                                req_name = req.strip().lower()
                                if req_name and req_name not in direct_names:
                                    transitive.append(
                                        DependencyInfo(
                                            name=req_name,
                                            source=f"transitive from {current_pkg}",
                                            is_direct=False,
                                        )
                                    )

        except Exception as e:
            logger.warning(f"Failed to get transitive deps: {e}")

        return transitive

    async def _check_updates(
        self,
        context: SkillContext,
        params: dict,
    ) -> SkillResult:
        """Check for available dependency updates.

        Args:
            context: Skill context
            params: Parameters (path, include_prerelease)

        Returns:
            SkillResult with update information
        """
        # Note: include_prerelease is parsed for future use
        _ = params.get("include_prerelease", False)

        updates: list[DependencyUpdate] = []

        try:
            cmd = ["pip", "list", "--outdated", "--format=json"]
            proc = subprocess.run(
                cmd,
                cwd=str(context.working_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if proc.returncode == 0:
                packages = json.loads(proc.stdout)
                for pkg in packages:
                    current = pkg["version"]
                    latest = pkg["latest_version"]

                    # Determine update type
                    update_type = self._determine_update_type(current, latest)

                    updates.append(
                        DependencyUpdate(
                            name=pkg["name"],
                            current_version=current,
                            latest_version=latest,
                            update_type=update_type,
                            breaking=update_type == "major",
                        )
                    )

        except Exception as e:
            logger.warning(f"Failed to check updates: {e}")
            return SkillResult.error(f"Failed to check updates: {e}")

        if not updates:
            return SkillResult.success(
                "All dependencies are up to date",
                data={"updates": []},
            )

        return SkillResult(
            status=SkillStatus.SUCCESS,
            message=f"Found {len(updates)} available updates",
            data={
                "updates": [u.to_dict() for u in updates],
                "major_updates": sum(1 for u in updates if u.update_type == "major"),
                "minor_updates": sum(1 for u in updates if u.update_type == "minor"),
                "patch_updates": sum(1 for u in updates if u.update_type == "patch"),
            },
            warnings=[
                f"{u.name}: {u.current_version} -> {u.latest_version} (MAJOR)"
                for u in updates
                if u.update_type == "major"
            ],
        )

    def _determine_update_type(
        self,
        current: str,
        latest: str,
    ) -> str:
        """Determine the type of version update.

        Args:
            current: Current version
            latest: Latest version

        Returns:
            Update type (major, minor, patch, unknown)
        """
        try:
            current_parts = current.split(".")
            latest_parts = latest.split(".")

            if current_parts[0] != latest_parts[0]:
                return "major"
            elif len(current_parts) > 1 and len(latest_parts) > 1:
                if current_parts[1] != latest_parts[1]:
                    return "minor"
            if len(current_parts) > 2 and len(latest_parts) > 2:
                if current_parts[2] != latest_parts[2]:
                    return "patch"
            return "unknown"
        except Exception:
            return "unknown"

    async def _find_unused(
        self,
        context: SkillContext,
        params: dict,
    ) -> SkillResult:
        """Find potentially unused dependencies.

        Args:
            context: Skill context
            params: Parameters (path, source_dir)

        Returns:
            SkillResult with unused dependency information
        """
        source_dir = context.working_dir / params.get("source_dir", ".")

        # Get direct dependencies
        deps_result = await self._analyze_deps(
            context,
            {"include_transitive": False},
        )

        if deps_result.status != SkillStatus.SUCCESS:
            return deps_result

        dependencies = deps_result.data.get("dependencies", [])
        dep_names = {d["name"] for d in dependencies}

        # Scan source files for import statements
        imported = set()

        for py_file in source_dir.rglob("*.py"):
            try:
                content = py_file.read_text()

                # Find import statements
                import_pattern = r"^\s*(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
                for match in re.finditer(import_pattern, content, re.MULTILINE):
                    module = match.group(1).lower()
                    imported.add(module)

            except Exception as e:
                logger.warning(f"Failed to scan {py_file}: {e}")

        # Find deps not imported
        # Note: Package name may differ from import name
        potentially_unused = dep_names - imported

        return SkillResult.success(
            f"Found {len(potentially_unused)} potentially unused dependencies",
            data={
                "potentially_unused": list(potentially_unused),
                "total_dependencies": len(dep_names),
                "imported_modules": list(imported),
            },
            warnings=[f"'{dep}' may be unused (not found in imports)" for dep in potentially_unused]
            if potentially_unused
            else [],
            metadata={
                "note": "Package names may differ from import names - verify before removing",
            },
        )

    async def _security_check(
        self,
        context: SkillContext,
        params: dict,
    ) -> SkillResult:
        """Check for known security vulnerabilities.

        Args:
            context: Skill context
            params: Parameters (path)

        Returns:
            SkillResult with vulnerability information
        """
        vulnerabilities = []

        # Try pip-audit if available
        try:
            proc = subprocess.run(
                ["pip-audit", "--format=json"],
                cwd=str(context.working_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )

            if proc.returncode == 0 or proc.stdout:
                try:
                    audit_results = json.loads(proc.stdout)
                    for vuln in audit_results:
                        vulnerabilities.append(
                            {
                                "package": vuln.get("name"),
                                "version": vuln.get("version"),
                                "vulnerability_id": vuln.get("id"),
                                "description": vuln.get("description"),
                                "fix_version": vuln.get("fix_versions"),
                            }
                        )
                except json.JSONDecodeError:
                    pass

        except FileNotFoundError:
            # pip-audit not installed, try safety
            try:
                proc = subprocess.run(
                    ["safety", "check", "--json"],
                    cwd=str(context.working_dir),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if proc.stdout:
                    try:
                        safety_results = json.loads(proc.stdout)
                        for vuln in safety_results.get("vulnerabilities", []):
                            vulnerabilities.append(
                                {
                                    "package": vuln.get("package_name"),
                                    "version": vuln.get("installed_version"),
                                    "vulnerability_id": vuln.get("vulnerability_id"),
                                    "description": vuln.get("advisory"),
                                }
                            )
                    except json.JSONDecodeError:
                        pass

            except FileNotFoundError:
                return SkillResult(
                    status=SkillStatus.SKIPPED,
                    message="No security scanner available (install pip-audit or safety)",
                    warnings=["Consider installing pip-audit: pip install pip-audit"],
                )

        except Exception as e:
            logger.warning(f"Security check error: {e}")
            return SkillResult.error(f"Security check failed: {e}")

        if not vulnerabilities:
            return SkillResult.success(
                "No known vulnerabilities found",
                data={"vulnerabilities": []},
            )

        return SkillResult(
            status=SkillStatus.FAILURE,
            message=f"Found {len(vulnerabilities)} vulnerabilities",
            data={
                "vulnerabilities": vulnerabilities,
                "affected_packages": list(set(v["package"] for v in vulnerabilities)),
            },
            errors=[f"{v['package']}: {v['vulnerability_id']}" for v in vulnerabilities],
        )

    async def _analyze_ansible(
        self,
        context: SkillContext,
        params: dict,
    ) -> SkillResult:
        """Analyze Ansible role dependencies.

        Args:
            context: Skill context
            params: Parameters (role_path)

        Returns:
            SkillResult with role dependency information
        """
        role_path = context.working_dir / params["role_path"]

        dependencies = []

        # Check meta/main.yml for dependencies
        meta_path = role_path / "meta" / "main.yml"
        if meta_path.exists():
            try:
                import yaml

                with open(meta_path) as f:
                    meta = yaml.safe_load(f)

                role_deps = meta.get("dependencies", [])
                for dep in role_deps:
                    if isinstance(dep, str):
                        dependencies.append(
                            DependencyInfo(
                                name=dep,
                                source="meta/main.yml",
                                is_direct=True,
                            )
                        )
                    elif isinstance(dep, dict):
                        dependencies.append(
                            DependencyInfo(
                                name=dep.get("role", dep.get("name", "unknown")),
                                version=dep.get("version"),
                                source="meta/main.yml",
                                is_direct=True,
                                metadata=dep,
                            )
                        )

            except Exception as e:
                logger.warning(f"Failed to parse meta/main.yml: {e}")

        # Check requirements.yml for collections
        requirements_path = role_path / "requirements.yml"
        if requirements_path.exists():
            try:
                import yaml

                with open(requirements_path) as f:
                    reqs = yaml.safe_load(f)

                collections = reqs.get("collections", [])
                for coll in collections:
                    if isinstance(coll, str):
                        dependencies.append(
                            DependencyInfo(
                                name=coll,
                                source="requirements.yml",
                                is_direct=True,
                                metadata={"type": "collection"},
                            )
                        )
                    elif isinstance(coll, dict):
                        dependencies.append(
                            DependencyInfo(
                                name=coll.get("name", "unknown"),
                                version=coll.get("version"),
                                source="requirements.yml",
                                is_direct=True,
                                metadata={"type": "collection", **coll},
                            )
                        )

            except Exception as e:
                logger.warning(f"Failed to parse requirements.yml: {e}")

        return SkillResult.success(
            f"Found {len(dependencies)} Ansible dependencies",
            data={
                "dependencies": [d.to_dict() for d in dependencies],
                "roles": [d for d in dependencies if d.metadata.get("type") != "collection"],
                "collections": [d for d in dependencies if d.metadata.get("type") == "collection"],
            },
        )
