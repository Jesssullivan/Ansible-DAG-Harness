"""
SandboxedExecution class for running code in E2B sandboxes.

Provides isolated execution environment for destructive operations
with automatic fallback to local execution when E2B is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path

from harness.sandbox.fallback import ExecutionResult, LocalFallbackExecutor, ResourceLimits
from harness.sandbox.templates import SandboxTemplate, get_template

logger = logging.getLogger(__name__)


# Re-export for convenience
__all__ = ["SandboxedExecution", "ExecutionResult", "ResourceLimits"]


class SandboxedExecution:
    """
    Execute code in an isolated E2B sandbox environment.

    Provides secure, isolated execution for potentially destructive
    operations like molecule tests, git operations, and shell scripts.
    Automatically falls back to local execution when E2B is unavailable.

    Features:
    - Isolated execution environment via E2B
    - Configurable resource limits (timeout, memory, CPU)
    - File synchronization between local and sandbox
    - Multiple execution templates (Python, Bash, Molecule)
    - Automatic fallback to local subprocess execution

    Example:
        # Basic usage
        async with SandboxedExecution(template="python") as sandbox:
            result = await sandbox.run("print('Hello from sandbox!')")
            print(result.stdout)

        # With custom limits
        sandbox = SandboxedExecution(
            template="molecule",
            timeout=600,
            memory_mb=1024,
        )
        result = await sandbox.run("molecule test")

        # File sync example
        async with SandboxedExecution() as sandbox:
            await sandbox.sync_files("./local_role", "/workspace/role")
            result = await sandbox.run("cd /workspace/role && molecule test")
            changed = await sandbox.get_changed_files("/workspace/role")
    """

    def __init__(
        self,
        template: str | SandboxTemplate = "python",
        timeout: int = 300,
        memory_mb: int = 512,
        cpu_count: int = 1,
        api_key: str | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        """
        Initialize the sandboxed execution environment.

        Args:
            template: Template name or SandboxTemplate instance
            timeout: Execution timeout in seconds (default 5 min)
            memory_mb: Memory limit in MB (default 512)
            cpu_count: CPU core limit (default 1)
            api_key: E2B API key (defaults to E2B_API_KEY env var)
            env_vars: Additional environment variables for sandbox
        """
        # Get template
        if isinstance(template, str):
            self.template = get_template(template)
        else:
            self.template = template

        # Resource limits
        self.limits = ResourceLimits(
            timeout_seconds=timeout,
            memory_mb=memory_mb,
            cpu_count=cpu_count,
        )

        # E2B configuration
        self.api_key = api_key or os.environ.get("E2B_API_KEY")
        self.env_vars = env_vars or {}

        # Merge template env vars
        self.env_vars = {**self.template.env_vars, **self.env_vars}

        # State
        self._sandbox = None
        self._fallback: LocalFallbackExecutor | None = None
        self._file_hashes: dict[str, str] = {}  # Track file changes

        # Check E2B availability
        self._e2b_available = self._check_e2b_available()

    def _check_e2b_available(self) -> bool:
        """Check if E2B is available and configured."""
        try:
            import e2b  # noqa: F401

            if not self.api_key:
                logger.warning(
                    "E2B package installed but E2B_API_KEY not set. "
                    "Falling back to local execution."
                )
                return False
            return True
        except ImportError:
            return False

    @property
    def using_sandbox(self) -> bool:
        """Check if we're using E2B sandbox (vs local fallback)."""
        return self._e2b_available and self._sandbox is not None

    async def _ensure_executor(self) -> None:
        """Ensure we have an executor ready (sandbox or fallback)."""
        if self._e2b_available and self._sandbox is None:
            await self._create_sandbox()
        elif not self._e2b_available and self._fallback is None:
            self._fallback = LocalFallbackExecutor(
                timeout=self.limits.timeout_seconds,
                memory_mb=self.limits.memory_mb,
                cpu_count=self.limits.cpu_count,
                env_vars=self.env_vars,
            )

    async def _create_sandbox(self) -> None:
        """Create the E2B sandbox."""
        try:
            from e2b_code_interpreter import Sandbox

            logger.info(
                f"Creating E2B sandbox with template '{self.template.name}' "
                f"(timeout={self.limits.timeout_seconds}s, "
                f"memory={self.limits.memory_mb}MB)"
            )

            # Create sandbox with configuration
            self._sandbox = Sandbox(
                template=self.template.e2b_template_id,
                api_key=self.api_key,
                timeout=self.limits.timeout_seconds,
            )

            # Run setup commands
            for cmd in self.template.setup_commands:
                logger.debug(f"Running setup command: {cmd}")
                self._sandbox.commands.run(cmd)

            logger.info("E2B sandbox created successfully")

        except ImportError:
            logger.warning("e2b_code_interpreter not available, falling back to local execution")
            self._e2b_available = False
            await self._ensure_executor()

        except Exception as e:
            logger.error(f"Failed to create E2B sandbox: {e}")
            self._e2b_available = False
            await self._ensure_executor()

    async def run(
        self,
        code: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Run code in the sandbox.

        Args:
            code: Code to execute (interpreted based on template)
            timeout: Override default timeout

        Returns:
            ExecutionResult with output and metadata

        Example:
            result = await sandbox.run("print('Hello!')")
            if result.success:
                print(result.stdout)
        """
        await self._ensure_executor()

        effective_timeout = timeout or self.limits.timeout_seconds
        start_time = time.time()

        if self._sandbox is not None:
            return await self._run_in_sandbox(code, effective_timeout, start_time)
        else:
            return await self._run_in_fallback(code, effective_timeout)

    async def _run_in_sandbox(
        self,
        code: str,
        timeout: int,
        start_time: float,
    ) -> ExecutionResult:
        """Run code in E2B sandbox."""
        try:
            command = self.template.get_execution_command(code, is_file=False)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._sandbox.commands.run(command, timeout=timeout),
            )

            duration = time.time() - start_time

            return ExecutionResult(
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
                sandbox_used=True,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            # Check for timeout
            if "timeout" in error_msg.lower():
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=error_msg,
                    duration_seconds=duration,
                    sandbox_used=True,
                    timed_out=True,
                    error_message=f"Sandbox execution timed out after {timeout}s",
                )

            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=error_msg,
                duration_seconds=duration,
                sandbox_used=True,
                error_message=f"Sandbox execution error: {e}",
            )

    async def _run_in_fallback(
        self,
        code: str,
        timeout: int,
    ) -> ExecutionResult:
        """Run code using local fallback."""
        if self.template.name in ("python", "molecule"):
            return await self._fallback.run_python(code, timeout=timeout)
        else:
            return await self._fallback.run_bash(code, timeout=timeout)

    async def run_file(
        self,
        path: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Run a file in the sandbox.

        For E2B, the file must first be synced to the sandbox.
        For local fallback, the file is executed directly.

        Args:
            path: Path to file (local path for fallback, remote for sandbox)
            timeout: Override default timeout

        Returns:
            ExecutionResult with output and metadata
        """
        await self._ensure_executor()

        effective_timeout = timeout or self.limits.timeout_seconds
        start_time = time.time()

        if self._sandbox is not None:
            try:
                command = self.template.get_execution_command(path, is_file=True)

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._sandbox.commands.run(command, timeout=effective_timeout),
                )

                duration = time.time() - start_time

                return ExecutionResult(
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration_seconds=duration,
                    sandbox_used=True,
                )

            except Exception as e:
                duration = time.time() - start_time
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    duration_seconds=duration,
                    sandbox_used=True,
                    error_message=f"Sandbox file execution error: {e}",
                )
        else:
            return await self._fallback.run_file(path, timeout=effective_timeout)

    async def sync_files(
        self,
        local_dir: str,
        remote_dir: str,
    ) -> list[str]:
        """
        Sync files from local directory to sandbox.

        Also records file hashes for later diff detection.

        Args:
            local_dir: Local directory path
            remote_dir: Remote directory path in sandbox

        Returns:
            List of synced file paths
        """
        await self._ensure_executor()

        local_path = Path(local_dir)
        if not local_path.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        synced_files = []

        if self._sandbox is not None:
            # Upload to E2B sandbox
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(local_path)
                    remote_path = f"{remote_dir}/{rel_path}"

                    # Read and upload file
                    content = file_path.read_bytes()
                    self._sandbox.files.write(remote_path, content)
                    synced_files.append(remote_path)

                    # Record hash for diff detection
                    self._file_hashes[remote_path] = hashlib.sha256(content).hexdigest()

            logger.info(f"Synced {len(synced_files)} files to sandbox: {remote_dir}")
        else:
            # Use fallback sync
            synced_files = await self._fallback.sync_files_to(local_dir, remote_dir)

            # Record hashes
            for file_path in synced_files:
                content = Path(file_path).read_bytes()
                self._file_hashes[file_path] = hashlib.sha256(content).hexdigest()

        return synced_files

    async def sync_files_back(
        self,
        remote_dir: str,
        local_dir: str,
    ) -> list[str]:
        """
        Sync files from sandbox back to local directory.

        Args:
            remote_dir: Remote directory path in sandbox
            local_dir: Local directory path

        Returns:
            List of synced file paths
        """
        await self._ensure_executor()

        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        synced_files = []

        if self._sandbox is not None:
            # List and download from E2B sandbox
            try:
                files = self._sandbox.files.list(remote_dir)

                for file_info in files:
                    if file_info.is_file:
                        remote_path = f"{remote_dir}/{file_info.name}"
                        content = self._sandbox.files.read(remote_path)

                        local_file = local_path / file_info.name
                        local_file.write_bytes(content)
                        synced_files.append(str(local_file))

            except Exception as e:
                logger.error(f"Failed to sync files from sandbox: {e}")

        else:
            synced_files = await self._fallback.sync_files_from(remote_dir, local_dir)

        return synced_files

    async def get_changed_files(
        self,
        remote_dir: str,
    ) -> list[str]:
        """
        Detect files that changed since sync_files was called.

        Args:
            remote_dir: Remote directory to check

        Returns:
            List of file paths that were modified
        """
        await self._ensure_executor()

        changed_files = []

        if self._sandbox is not None:
            try:
                # List files in remote directory
                files = self._sandbox.files.list(remote_dir)

                for file_info in files:
                    if file_info.is_file:
                        remote_path = f"{remote_dir}/{file_info.name}"
                        content = self._sandbox.files.read(remote_path)
                        current_hash = hashlib.sha256(content).hexdigest()

                        original_hash = self._file_hashes.get(remote_path)
                        if original_hash is None or current_hash != original_hash:
                            changed_files.append(remote_path)

            except Exception as e:
                logger.error(f"Failed to check changed files: {e}")

        else:
            # Check fallback temp directory
            if self._fallback and self._fallback._temp_dir:
                remote_path = self._fallback._temp_dir / remote_dir
                if remote_path.exists():
                    for file_path in remote_path.rglob("*"):
                        if file_path.is_file():
                            content = file_path.read_bytes()
                            current_hash = hashlib.sha256(content).hexdigest()

                            original_hash = self._file_hashes.get(str(file_path))
                            if original_hash is None or current_hash != original_hash:
                                changed_files.append(str(file_path))

        return changed_files

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception as e:
                logger.warning(f"Error killing sandbox: {e}")
            self._sandbox = None

        if self._fallback is not None:
            await self._fallback.cleanup()
            self._fallback = None

        self._file_hashes.clear()
        logger.debug("Sandbox resources cleaned up")

    async def __aenter__(self) -> SandboxedExecution:
        """Async context manager entry."""
        await self._ensure_executor()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.cleanup()
