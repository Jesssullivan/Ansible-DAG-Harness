"""
Parallel execution support for wave-based workflows.

Provides:
- Wave grouping of independent nodes
- Concurrent execution with asyncio
- Progress tracking across parallel tasks
- Error aggregation and reporting
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from harness.db.state import StateDB
from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    LangGraphWorkflowRunner,
    create_initial_state,
)


class WaveStatus(str, Enum):
    """Status of a wave execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some roles completed, some failed


@dataclass
class RoleExecutionResult:
    """Result of a single role execution."""
    role_name: str
    wave: int
    status: str
    execution_id: Optional[int] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    summary: Optional[dict] = None


@dataclass
class WaveExecutionResult:
    """Result of a wave execution."""
    wave: int
    wave_name: str
    status: WaveStatus
    roles: list[RoleExecutionResult]
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.roles if r.status == "completed")

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.roles if r.status in ("failed", "error"))


@dataclass
class WaveProgress:
    """Progress tracking for parallel wave execution."""
    wave: int
    total_roles: int
    completed_roles: int = 0
    failed_roles: int = 0
    current_roles: list[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        if self.total_roles == 0:
            return 100.0
        return (self.completed_roles + self.failed_roles) / self.total_roles * 100


class ParallelWaveExecutor:
    """
    Execute workflow for multiple roles in parallel within a wave.

    Supports:
    - Concurrent execution up to max_concurrent limit
    - Progress tracking and callbacks
    - Error aggregation without stopping other roles
    - Wave-level success/failure determination
    """

    # Wave definitions matching CLAUDE.md
    WAVES = {
        0: {
            "name": "Foundation",
            "roles": ["common"]
        },
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
        },
    }

    def __init__(
        self,
        db: StateDB,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable[[WaveProgress], None]] = None
    ):
        self.db = db
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback
        self._runner = LangGraphWorkflowRunner(db)

    async def _execute_role(
        self,
        role_name: str,
        wave: int,
        semaphore: asyncio.Semaphore,
        progress: WaveProgress
    ) -> RoleExecutionResult:
        """Execute a single role with semaphore control."""
        async with semaphore:
            progress.current_roles.append(role_name)
            if self.progress_callback:
                self.progress_callback(progress)

            start_time = datetime.utcnow()
            try:
                result = await self._runner.execute(role_name)

                duration = (datetime.utcnow() - start_time).total_seconds()

                if result.get("status") == "completed":
                    progress.completed_roles += 1
                else:
                    progress.failed_roles += 1

                return RoleExecutionResult(
                    role_name=role_name,
                    wave=wave,
                    status=result.get("status", "unknown"),
                    execution_id=result.get("execution_id"),
                    error=result.get("error"),
                    duration_seconds=duration,
                    summary=result.get("summary")
                )

            except Exception as e:
                progress.failed_roles += 1
                duration = (datetime.utcnow() - start_time).total_seconds()

                return RoleExecutionResult(
                    role_name=role_name,
                    wave=wave,
                    status="error",
                    error=str(e),
                    duration_seconds=duration
                )
            finally:
                progress.current_roles.remove(role_name)
                if self.progress_callback:
                    self.progress_callback(progress)

    async def execute_wave(
        self,
        wave: int,
        roles: Optional[list[str]] = None
    ) -> WaveExecutionResult:
        """
        Execute all roles in a wave concurrently.

        Args:
            wave: Wave number (0-4)
            roles: Optional override of roles to execute (uses WAVES if not provided)

        Returns:
            WaveExecutionResult with all role results
        """
        wave_config = self.WAVES.get(wave, {"name": f"Wave {wave}", "roles": []})
        wave_name = wave_config["name"]
        roles_to_execute = roles or wave_config["roles"]

        if not roles_to_execute:
            return WaveExecutionResult(
                wave=wave,
                wave_name=wave_name,
                status=WaveStatus.COMPLETED,
                roles=[],
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_seconds=0
            )

        progress = WaveProgress(
            wave=wave,
            total_roles=len(roles_to_execute)
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)
        started_at = datetime.utcnow()

        # Execute all roles concurrently
        tasks = [
            self._execute_role(role, wave, semaphore, progress)
            for role in roles_to_execute
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        # Process results (handle any exceptions that weren't caught)
        role_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                role_results.append(RoleExecutionResult(
                    role_name=roles_to_execute[i],
                    wave=wave,
                    status="error",
                    error=str(result)
                ))
            else:
                role_results.append(result)

        # Determine wave status
        success_count = sum(1 for r in role_results if r.status == "completed")
        failure_count = len(role_results) - success_count

        if failure_count == 0:
            status = WaveStatus.COMPLETED
        elif success_count == 0:
            status = WaveStatus.FAILED
        else:
            status = WaveStatus.PARTIAL

        return WaveExecutionResult(
            wave=wave,
            wave_name=wave_name,
            status=status,
            roles=role_results,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration
        )

    async def execute_all_waves(
        self,
        start_wave: int = 0,
        end_wave: int = 4,
        stop_on_wave_failure: bool = True
    ) -> list[WaveExecutionResult]:
        """
        Execute all waves sequentially, with roles within each wave in parallel.

        Args:
            start_wave: First wave to execute (default: 0)
            end_wave: Last wave to execute (default: 4)
            stop_on_wave_failure: If True, stop if any role in a wave fails

        Returns:
            List of WaveExecutionResult for all executed waves
        """
        results = []

        for wave in range(start_wave, end_wave + 1):
            wave_result = await self.execute_wave(wave)
            results.append(wave_result)

            # Check if we should stop
            if stop_on_wave_failure and wave_result.status in (WaveStatus.FAILED, WaveStatus.PARTIAL):
                break

        return results


async def execute_roles_parallel(
    db: StateDB,
    roles: list[str],
    max_concurrent: int = 3,
    progress_callback: Optional[Callable[[WaveProgress], None]] = None
) -> list[RoleExecutionResult]:
    """
    Convenience function to execute arbitrary roles in parallel.

    Args:
        db: StateDB instance
        roles: List of role names to execute
        max_concurrent: Maximum concurrent executions
        progress_callback: Optional progress callback

    Returns:
        List of RoleExecutionResult
    """
    executor = ParallelWaveExecutor(db, max_concurrent, progress_callback)

    # Execute as a single "wave"
    result = await executor.execute_wave(wave=-1, roles=roles)
    return result.roles
