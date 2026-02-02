"""
Tests for the E2B sandbox integration module.

Tests cover:
- SandboxedExecution class
- Template configuration
- Local fallback behavior
- Resource limits
- File synchronization
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.sandbox import (
    E2B_AVAILABLE,
    ExecutionResult,
    LocalFallbackExecutor,
    ResourceLimits,
    SandboxedExecution,
    e2b_available,
)
from harness.sandbox.templates import (
    BashTemplate,
    MoleculeTemplate,
    PythonTemplate,
    SandboxTemplate,
    get_template,
    list_templates,
    register_template,
)

# ============================================================================
# TEMPLATE TESTS
# ============================================================================


class TestTemplates:
    """Test sandbox template definitions."""

    def test_list_templates(self):
        """Test listing available templates."""
        templates = list_templates()
        assert "python" in templates
        assert "bash" in templates
        assert "molecule" in templates

    def test_get_python_template(self):
        """Test getting Python template."""
        template = get_template("python")
        assert isinstance(template, PythonTemplate)
        assert template.name == "python"
        assert template.e2b_template_id == "python"
        assert template.default_timeout == 300
        assert template.default_memory_mb == 512

    def test_get_bash_template(self):
        """Test getting Bash template."""
        template = get_template("bash")
        assert isinstance(template, BashTemplate)
        assert template.name == "bash"
        assert template.e2b_template_id == "base"
        assert template.shell == "/bin/bash"

    def test_get_molecule_template(self):
        """Test getting Molecule template."""
        template = get_template("molecule")
        assert isinstance(template, MoleculeTemplate)
        assert template.name == "molecule"
        assert template.default_timeout == 600  # Extended for molecule

    def test_get_template_with_override(self):
        """Test getting template with custom values."""
        template = get_template("python", default_timeout=600)
        assert template.default_timeout == 600

    def test_get_unknown_template_raises(self):
        """Test that unknown template raises ValueError."""
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent")

    def test_python_execution_command_code(self):
        """Test Python template execution command for code."""
        template = get_template("python")
        cmd = template.get_execution_command("print('hello')", is_file=False)
        assert "python -c" in cmd
        assert "print" in cmd

    def test_python_execution_command_file(self):
        """Test Python template execution command for file."""
        template = get_template("python")
        cmd = template.get_execution_command("/path/to/script.py", is_file=True)
        assert cmd == "python /path/to/script.py"

    def test_bash_execution_command_code(self):
        """Test Bash template execution command for code."""
        template = get_template("bash")
        cmd = template.get_execution_command("echo hello", is_file=False)
        assert "/bin/bash -c" in cmd

    def test_bash_execution_command_file(self):
        """Test Bash template execution command for file."""
        template = get_template("bash")
        cmd = template.get_execution_command("/path/to/script.sh", is_file=True)
        assert cmd == "/bin/bash /path/to/script.sh"

    def test_python_file_extension(self):
        """Test Python template file extension."""
        template = get_template("python")
        assert template.get_file_extension() == ".py"

    def test_bash_file_extension(self):
        """Test Bash template file extension."""
        template = get_template("bash")
        assert template.get_file_extension() == ".sh"

    def test_molecule_execution_command(self):
        """Test Molecule template execution command."""
        template = get_template("molecule")

        # Molecule command string
        cmd = template.get_execution_command("molecule test", is_file=False)
        assert cmd == "molecule test"

        # Non-molecule command
        cmd = template.get_execution_command("test", is_file=False)
        assert cmd == "molecule test"

    def test_register_custom_template(self):
        """Test registering a custom template."""
        from dataclasses import dataclass

        @dataclass
        class CustomTemplate(SandboxTemplate):
            name: str = "custom"
            description: str = "Custom template"
            e2b_template_id: str = "custom"

            def get_execution_command(self, code_or_path, is_file=False):
                return f"custom {code_or_path}"

            def get_file_extension(self):
                return ".custom"

        register_template("custom", CustomTemplate)
        assert "custom" in list_templates()

        template = get_template("custom")
        assert template.name == "custom"


# ============================================================================
# RESOURCE LIMITS TESTS
# ============================================================================


class TestResourceLimits:
    """Test resource limit validation."""

    def test_valid_limits(self):
        """Test creating valid resource limits."""
        limits = ResourceLimits(
            timeout_seconds=60,
            memory_mb=256,
            cpu_count=2,
        )
        assert limits.timeout_seconds == 60
        assert limits.memory_mb == 256
        assert limits.cpu_count == 2

    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        assert limits.timeout_seconds == 300
        assert limits.memory_mb == 512
        assert limits.cpu_count == 1

    def test_invalid_timeout_raises(self):
        """Test that invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ResourceLimits(timeout_seconds=0)

    def test_invalid_memory_raises(self):
        """Test that invalid memory raises ValueError."""
        with pytest.raises(ValueError, match="memory_mb must be positive"):
            ResourceLimits(memory_mb=-100)

    def test_invalid_cpu_raises(self):
        """Test that invalid CPU count raises ValueError."""
        with pytest.raises(ValueError, match="cpu_count must be positive"):
            ResourceLimits(cpu_count=0)


# ============================================================================
# EXECUTION RESULT TESTS
# ============================================================================


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_successful_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            exit_code=0,
            stdout="Hello, World!",
            stderr="",
            duration_seconds=0.5,
            sandbox_used=True,
        )
        assert result.success
        assert result.exit_code == 0
        assert result.stdout == "Hello, World!"
        assert not result.timed_out

    def test_failed_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="Error occurred",
            duration_seconds=0.5,
            sandbox_used=False,
        )
        assert not result.success
        assert result.exit_code == 1
        assert result.stderr == "Error occurred"

    def test_timeout_result(self):
        """Test timeout execution result."""
        result = ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="",
            duration_seconds=300.0,
            sandbox_used=True,
            timed_out=True,
            error_message="Command timed out",
        )
        assert not result.success
        assert result.timed_out
        assert result.error_message == "Command timed out"

    def test_result_str(self):
        """Test ExecutionResult string representation."""
        result = ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.5,
            sandbox_used=True,
        )
        result_str = str(result)
        assert "success" in result_str
        assert "1.5" in result_str
        assert "sandbox" in result_str

    def test_result_str_failed(self):
        """Test ExecutionResult string representation for failure."""
        result = ExecutionResult(
            exit_code=42,
            stdout="",
            stderr="",
            duration_seconds=1.0,
            sandbox_used=False,
        )
        result_str = str(result)
        assert "failed" in result_str
        assert "exit=42" in result_str
        assert "local" in result_str


# ============================================================================
# LOCAL FALLBACK TESTS
# ============================================================================


class TestLocalFallbackExecutor:
    """Test local fallback execution."""

    @pytest.fixture
    def executor(self):
        """Create a local fallback executor."""
        return LocalFallbackExecutor(timeout=30)

    @pytest.mark.asyncio
    async def test_run_simple_command(self, executor):
        """Test running a simple command."""
        result = await executor.run("echo hello")
        assert result.success
        assert "hello" in result.stdout
        assert not result.sandbox_used

    @pytest.mark.asyncio
    async def test_run_python_code(self, executor):
        """Test running Python code."""
        result = await executor.run_python("print('hello from python')")
        assert result.success
        assert "hello from python" in result.stdout

    @pytest.mark.asyncio
    async def test_run_bash_code(self, executor):
        """Test running Bash code."""
        result = await executor.run_bash("echo 'bash works'")
        assert result.success
        assert "bash works" in result.stdout

    @pytest.mark.asyncio
    async def test_run_file_python(self, executor):
        """Test running a Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('file executed')")
            f.flush()
            temp_path = f.name

        try:
            result = await executor.run_file(temp_path)
            assert result.success
            assert "file executed" in result.stdout
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_run_file_not_found(self, executor):
        """Test running a non-existent file."""
        result = await executor.run_file("/nonexistent/path.py")
        assert not result.success
        assert "not found" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_command_timeout(self, executor):
        """Test command timeout handling."""
        result = await executor.run("sleep 10", timeout=1)
        assert not result.success
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_command_failure(self, executor):
        """Test handling command failure."""
        result = await executor.run("exit 42")
        assert not result.success
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_create_temp_dir(self, executor):
        """Test temporary directory creation."""
        temp_dir = executor.create_temp_dir()
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Cleanup
        await executor.cleanup()
        assert not temp_dir.exists()

    @pytest.mark.asyncio
    async def test_sync_files_to(self, executor):
        """Test syncing files to working directory."""
        with tempfile.TemporaryDirectory() as src_dir:
            # Create test files
            src_path = Path(src_dir)
            (src_path / "test.txt").write_text("hello")
            (src_path / "subdir").mkdir()
            (src_path / "subdir" / "nested.txt").write_text("nested")

            synced = await executor.sync_files_to(src_dir, "workspace")
            assert len(synced) == 2

            # Verify files exist in temp dir
            for file_path in synced:
                assert Path(file_path).exists()

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_sync_files_from(self, executor):
        """Test syncing files from working directory."""
        # Create files in temp dir
        temp_dir = executor.create_temp_dir()
        remote_dir = temp_dir / "remote"
        remote_dir.mkdir()
        (remote_dir / "output.txt").write_text("output content")

        with tempfile.TemporaryDirectory() as local_dir:
            synced = await executor.sync_files_from("remote", local_dir)
            assert len(synced) == 1
            assert Path(synced[0]).read_text() == "output content"

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with LocalFallbackExecutor(timeout=30) as executor:
            result = await executor.run("echo test")
            assert result.success

    @pytest.mark.asyncio
    async def test_env_vars(self):
        """Test custom environment variables."""
        executor = LocalFallbackExecutor(env_vars={"MY_VAR": "my_value"})
        result = await executor.run("echo $MY_VAR")
        assert "my_value" in result.stdout
        await executor.cleanup()


# ============================================================================
# SANDBOXED EXECUTION TESTS (with mocked E2B)
# ============================================================================


class TestSandboxedExecution:
    """Test SandboxedExecution class."""

    @pytest.fixture
    def mock_e2b_sandbox(self):
        """Create a mock E2B sandbox."""
        with patch(
            "harness.sandbox.execution.SandboxedExecution._check_e2b_available"
        ) as mock_check:
            mock_check.return_value = False  # Force fallback
            yield mock_check

    @pytest.mark.asyncio
    async def test_init_default_template(self, mock_e2b_sandbox):
        """Test initialization with default template."""
        executor = SandboxedExecution()
        assert executor.template.name == "python"
        assert executor.limits.timeout_seconds == 300
        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_init_custom_template(self, mock_e2b_sandbox):
        """Test initialization with custom template."""
        executor = SandboxedExecution(template="bash", timeout=60)
        assert executor.template.name == "bash"
        assert executor.limits.timeout_seconds == 60
        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_init_with_template_instance(self, mock_e2b_sandbox):
        """Test initialization with SandboxTemplate instance."""
        template = get_template("molecule")
        executor = SandboxedExecution(template=template)
        assert executor.template.name == "molecule"
        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_fallback_when_e2b_unavailable(self, mock_e2b_sandbox):
        """Test fallback to local execution when E2B unavailable."""
        async with SandboxedExecution() as executor:
            result = await executor.run("print('fallback test')")
            assert result.success
            assert not result.sandbox_used  # Using fallback
            assert "fallback test" in result.stdout

    @pytest.mark.asyncio
    async def test_using_sandbox_property(self, mock_e2b_sandbox):
        """Test using_sandbox property."""
        async with SandboxedExecution() as executor:
            assert not executor.using_sandbox  # Fallback mode

    @pytest.mark.asyncio
    async def test_run_python_code(self, mock_e2b_sandbox):
        """Test running Python code."""
        async with SandboxedExecution(template="python") as executor:
            result = await executor.run("x = 1 + 1\nprint(f'Result: {x}')")
            assert result.success
            assert "Result: 2" in result.stdout

    @pytest.mark.asyncio
    async def test_run_bash_code(self, mock_e2b_sandbox):
        """Test running Bash code."""
        async with SandboxedExecution(template="bash") as executor:
            result = await executor.run("echo 'Hello from bash'")
            assert result.success
            assert "Hello from bash" in result.stdout

    @pytest.mark.asyncio
    async def test_run_with_timeout(self, mock_e2b_sandbox):
        """Test running with custom timeout."""
        async with SandboxedExecution(timeout=30) as executor:
            result = await executor.run("print('quick')", timeout=5)
            assert result.success

    @pytest.mark.asyncio
    async def test_run_file(self, mock_e2b_sandbox):
        """Test running a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('running file')")
            f.flush()
            temp_path = f.name

        try:
            async with SandboxedExecution() as executor:
                result = await executor.run_file(temp_path)
                assert result.success
                assert "running file" in result.stdout
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_sync_files(self, mock_e2b_sandbox):
        """Test file synchronization."""
        with tempfile.TemporaryDirectory() as src_dir:
            # Create test files
            src_path = Path(src_dir)
            (src_path / "test.py").write_text("print('synced')")

            async with SandboxedExecution() as executor:
                synced = await executor.sync_files(src_dir, "workspace")
                assert len(synced) >= 1

    @pytest.mark.asyncio
    async def test_sync_files_not_found(self, mock_e2b_sandbox):
        """Test sync_files with non-existent directory."""
        async with SandboxedExecution() as executor:
            with pytest.raises(FileNotFoundError):
                await executor.sync_files("/nonexistent/path", "workspace")

    @pytest.mark.asyncio
    async def test_env_vars_passed(self, mock_e2b_sandbox):
        """Test environment variables are passed."""
        async with SandboxedExecution(
            template="bash", env_vars={"TEST_VAR": "test_value"}
        ) as executor:
            result = await executor.run("echo $TEST_VAR")
            assert "test_value" in result.stdout

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_e2b_sandbox):
        """Test cleanup releases resources."""
        executor = SandboxedExecution()
        await executor._ensure_executor()
        assert executor._fallback is not None

        await executor.cleanup()
        assert executor._fallback is None


# ============================================================================
# E2B MOCKED INTEGRATION TESTS
# ============================================================================


class TestSandboxedExecutionWithMockedE2B:
    """Test SandboxedExecution with mocked E2B SDK."""

    @pytest.fixture
    def mock_e2b_full(self):
        """Create fully mocked E2B environment."""
        mock_sandbox = MagicMock()
        mock_sandbox.commands = MagicMock()
        mock_sandbox.files = MagicMock()

        # Mock command execution
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.stdout = "mocked output"
        mock_result.stderr = ""
        mock_sandbox.commands.run.return_value = mock_result

        # Mock file operations
        mock_sandbox.files.write = MagicMock()
        mock_sandbox.files.read = MagicMock(return_value=b"file content")
        mock_file_info = MagicMock()
        mock_file_info.is_file = True
        mock_file_info.name = "test.txt"
        mock_sandbox.files.list.return_value = [mock_file_info]

        with patch.dict(
            "sys.modules",
            {
                "e2b": MagicMock(),
                "e2b_code_interpreter": MagicMock(),
            },
        ):
            with patch(
                "harness.sandbox.execution.SandboxedExecution._check_e2b_available"
            ) as mock_check:
                mock_check.return_value = True

                with patch(
                    "harness.sandbox.execution.SandboxedExecution._create_sandbox"
                ) as mock_create:

                    async def create_sandbox_coro(self):
                        self._sandbox = mock_sandbox

                    mock_create.side_effect = lambda self: asyncio.coroutine(
                        lambda: setattr(self, "_sandbox", mock_sandbox)
                    )()

                    yield mock_sandbox

    @pytest.mark.asyncio
    async def test_sandbox_run_mocked(self, mock_e2b_full):
        """Test running code in mocked E2B sandbox."""
        executor = SandboxedExecution(api_key="test_key")
        executor._sandbox = mock_e2b_full  # Directly set mock

        result = await executor._run_in_sandbox(
            "print('test')",
            timeout=30,
            start_time=0,
        )

        assert result.sandbox_used
        assert result.stdout == "mocked output"
        mock_e2b_full.commands.run.assert_called_once()

        await executor.cleanup()


# ============================================================================
# E2B AVAILABILITY TESTS
# ============================================================================


class TestE2BAvailability:
    """Test E2B availability detection."""

    def test_e2b_available_function(self):
        """Test e2b_available function."""
        result = e2b_available()
        assert isinstance(result, bool)
        assert result == E2B_AVAILABLE

    def test_e2b_available_constant(self):
        """Test E2B_AVAILABLE constant matches function."""
        assert e2b_available() == E2B_AVAILABLE


# ============================================================================
# INTEGRATION STYLE TESTS (using fallback)
# ============================================================================


class TestIntegration:
    """Integration-style tests using local fallback."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete workflow with file sync and execution."""
        with tempfile.TemporaryDirectory() as work_dir:
            src_dir = Path(work_dir) / "source"
            src_dir.mkdir()

            # Create a test script
            script = src_dir / "test_script.py"
            script.write_text("""
import sys
print("Test passed!")
sys.exit(0)
""")

            async with SandboxedExecution(
                template="python",
                timeout=60,
            ) as executor:
                # Sync files
                synced = await executor.sync_files(str(src_dir), "workspace")
                assert len(synced) >= 1

                # Run the script (in fallback, use original path)
                result = await executor.run_file(str(script))
                assert result.success
                assert "Test passed!" in result.stdout

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in execution."""
        async with SandboxedExecution(template="python") as executor:
            result = await executor.run("raise ValueError('intentional error')")
            assert not result.success
            assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_multiline_code(self):
        """Test multiline code execution."""
        code = """
def greet(name):
    return f"Hello, {name}!"

result = greet("World")
print(result)
"""
        async with SandboxedExecution(template="python") as executor:
            result = await executor.run(code)
            assert result.success
            assert "Hello, World!" in result.stdout

    @pytest.mark.asyncio
    async def test_shell_pipeline(self):
        """Test shell pipeline execution."""
        async with SandboxedExecution(template="bash") as executor:
            result = await executor.run("echo 'line1\nline2\nline3' | wc -l")
            assert result.success
            assert "3" in result.stdout
