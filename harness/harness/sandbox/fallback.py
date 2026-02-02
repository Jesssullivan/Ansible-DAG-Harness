"""
Local execution fallback when E2B is not available.

Provides subprocess-based execution with similar interface to
SandboxedExecution, but running locally without isolation.

WARNING: Local execution does not provide the same security
guarantees as E2B sandboxes. Use with caution for untrusted code.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """
    Resource limits for execution.

    These limits are best-effort for local execution - actual
    enforcement depends on the operating system capabilities.
    """

    timeout_seconds: int = 300  # 5 minutes default
    memory_mb: int = 512  # Not enforced locally
    cpu_count: int = 1  # Not enforced locally

    def __post_init__(self):
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.memory_mb <= 0:
            raise ValueError("memory_mb must be positive")
        if self.cpu_count <= 0:
            raise ValueError("cpu_count must be positive")


@dataclass
class ExecutionResult:
    """
    Result of code execution.

    Contains the output, timing, and metadata about the execution.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    sandbox_used: bool  # True if ran in E2B, False if local fallback

    # Additional metadata
    timed_out: bool = False
    error_message: str | None = None
    files_changed: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.exit_code == 0 and not self.timed_out

    def __str__(self) -> str:
        status = "success" if self.success else f"failed (exit={self.exit_code})"
        mode = "sandbox" if self.sandbox_used else "local"
        return f"ExecutionResult({status}, {self.duration_seconds:.2f}s, {mode})"


class LocalFallbackExecutor:
    """
    Local subprocess executor as fallback when E2B is unavailable.

    Provides a similar interface to E2B sandbox execution but runs
    code locally using subprocess. This is less isolated but allows
    the harness to function without E2B credentials.

    WARNING: This executor provides NO isolation. Code runs with
    the same permissions as the harness process. Do not use for
    untrusted code.

    Example:
        executor = LocalFallbackExecutor(timeout=60)

        # Run Python code
        result = await executor.run_python("print('hello')")

        # Run Bash command
        result = await executor.run_bash("ls -la")

        # Run a file
        result = await executor.run_file("/path/to/script.py")
    """

    def __init__(
        self,
        timeout: int = 300,
        memory_mb: int = 512,
        cpu_count: int = 1,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        """
        Initialize the local fallback executor.

        Args:
            timeout: Default timeout in seconds
            memory_mb: Memory limit (informational only, not enforced)
            cpu_count: CPU limit (informational only, not enforced)
            working_dir: Working directory for execution
            env_vars: Additional environment variables
        """
        self.limits = ResourceLimits(
            timeout_seconds=timeout,
            memory_mb=memory_mb,
            cpu_count=cpu_count,
        )
        self.working_dir = working_dir
        self.env_vars = env_vars or {}
        self._temp_dir: Path | None = None

        logger.warning(
            "Using local fallback executor - code runs without isolation. "
            "Install e2b for sandboxed execution: pip install 'dag-harness[sandbox]'"
        )

    def _get_env(self) -> dict[str, str]:
        """Get environment variables for execution."""
        env = os.environ.copy()
        env.update(self.env_vars)
        return env

    def _get_working_dir(self) -> str | None:
        """Get working directory for execution."""
        if self.working_dir:
            return self.working_dir
        if self._temp_dir:
            return str(self._temp_dir)
        return None

    async def run(
        self,
        command: str,
        timeout: int | None = None,
        shell: bool = True,
    ) -> ExecutionResult:
        """
        Run a command locally.

        Args:
            command: Command string to execute
            timeout: Override default timeout
            shell: Run through shell (default True)

        Returns:
            ExecutionResult with output and metadata
        """
        effective_timeout = timeout or self.limits.timeout_seconds
        start_time = time.time()

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    shell=shell,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                    cwd=self._get_working_dir(),
                    env=self._get_env(),
                ),
            )

            duration = time.time() - start_time

            return ExecutionResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
                sandbox_used=False,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            return ExecutionResult(
                exit_code=-1,
                stdout=e.stdout or "" if hasattr(e, "stdout") else "",
                stderr=e.stderr or "" if hasattr(e, "stderr") else "",
                duration_seconds=duration,
                sandbox_used=False,
                timed_out=True,
                error_message=f"Command timed out after {effective_timeout}s",
            )

        except Exception as e:
            duration = time.time() - start_time
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                sandbox_used=False,
                error_message=f"Execution error: {e}",
            )

    async def run_python(
        self,
        code: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Run Python code locally.

        Args:
            code: Python code to execute
            timeout: Override default timeout

        Returns:
            ExecutionResult with output and metadata
        """
        # Write code to temp file and execute
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            return await self.run(f"python {temp_path}", timeout=timeout)
        finally:
            os.unlink(temp_path)

    async def run_bash(
        self,
        code: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Run Bash code locally.

        Args:
            code: Bash code to execute
            timeout: Override default timeout

        Returns:
            ExecutionResult with output and metadata
        """
        return await self.run(code, timeout=timeout, shell=True)

    async def run_file(
        self,
        path: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Run a file locally based on its extension.

        Args:
            path: Path to file to execute
            timeout: Override default timeout

        Returns:
            ExecutionResult with output and metadata
        """
        file_path = Path(path)
        if not file_path.exists():
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"File not found: {path}",
                duration_seconds=0.0,
                sandbox_used=False,
                error_message=f"File not found: {path}",
            )

        extension = file_path.suffix.lower()

        if extension == ".py":
            return await self.run(f"python {path}", timeout=timeout)
        elif extension in (".sh", ".bash"):
            return await self.run(f"bash {path}", timeout=timeout)
        else:
            # Try to execute directly
            return await self.run(path, timeout=timeout, shell=False)

    def create_temp_dir(self) -> Path:
        """
        Create a temporary directory for file operations.

        Returns:
            Path to the temporary directory
        """
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="harness_sandbox_"))
        return self._temp_dir

    async def sync_files_to(
        self,
        local_dir: str,
        remote_dir: str | None = None,
    ) -> list[str]:
        """
        Sync local files to the working directory.

        For local fallback, this copies files to the temp directory.

        Args:
            local_dir: Local directory to sync from
            remote_dir: Remote path (uses temp dir root if None)

        Returns:
            List of synced file paths
        """
        local_path = Path(local_dir)
        if not local_path.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        target_dir = self.create_temp_dir()
        if remote_dir:
            target_dir = target_dir / remote_dir
            target_dir.mkdir(parents=True, exist_ok=True)

        synced_files = []

        for src_file in local_path.rglob("*"):
            if src_file.is_file():
                rel_path = src_file.relative_to(local_path)
                dst_file = target_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                synced_files.append(str(dst_file))

        return synced_files

    async def sync_files_from(
        self,
        remote_dir: str,
        local_dir: str,
    ) -> list[str]:
        """
        Sync files from the working directory to local.

        For local fallback, this copies files from the temp directory.

        Args:
            remote_dir: Remote directory to sync from
            local_dir: Local directory to sync to

        Returns:
            List of synced file paths
        """
        if self._temp_dir is None:
            return []

        remote_path = self._temp_dir / remote_dir
        if not remote_path.exists():
            return []

        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        synced_files = []

        for src_file in remote_path.rglob("*"):
            if src_file.is_file():
                rel_path = src_file.relative_to(remote_path)
                dst_file = local_path / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                synced_files.append(str(dst_file))

        return synced_files

    async def cleanup(self) -> None:
        """Clean up temporary resources."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    async def __aenter__(self) -> LocalFallbackExecutor:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.cleanup()
