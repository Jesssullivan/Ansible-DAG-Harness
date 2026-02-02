"""
Sandbox template definitions for E2B execution environments.

Provides pre-configured templates for:
- Python execution (molecule tests, pytest, general Python scripts)
- Bash execution (git operations, shell scripts)

Each template defines the base environment, installed packages,
and setup scripts needed for that type of execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SandboxTemplate(ABC):
    """
    Abstract base class for sandbox templates.

    Templates define the execution environment configuration
    for E2B sandboxes, including language runtime, packages,
    and initialization scripts.
    """

    name: str
    description: str
    e2b_template_id: str  # E2B template identifier (e.g., "python", "base")

    # Default resource limits
    default_timeout: int = 300  # 5 minutes
    default_memory_mb: int = 512
    default_cpu_count: int = 1

    # Environment setup
    env_vars: dict[str, str] = field(default_factory=dict)
    setup_commands: list[str] = field(default_factory=list)

    @abstractmethod
    def get_execution_command(self, code_or_path: str, is_file: bool = False) -> str:
        """
        Get the command to execute the given code or file.

        Args:
            code_or_path: Code string or path to file to execute
            is_file: If True, code_or_path is a file path

        Returns:
            Command string to execute in the sandbox
        """
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the file extension for this template's language."""
        pass


@dataclass
class PythonTemplate(SandboxTemplate):
    """
    Template for Python code execution.

    Suitable for:
    - Molecule test execution
    - Pytest runs
    - General Python scripts
    - Ansible-related Python tools

    The template includes common testing packages pre-installed.
    """

    name: str = "python"
    description: str = "Python 3.11+ execution environment with testing tools"
    e2b_template_id: str = "python"

    # Python-specific settings
    python_version: str = "3.11"
    pip_packages: list[str] = field(
        default_factory=lambda: [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "molecule>=6.0.0",
            "ansible-core>=2.15.0",
            "pyyaml>=6.0.0",
        ]
    )

    def __post_init__(self):
        # Add pip install to setup commands
        if self.pip_packages:
            packages = " ".join(f'"{pkg}"' for pkg in self.pip_packages)
            self.setup_commands = [
                f"pip install --quiet {packages}",
            ]

    def get_execution_command(self, code_or_path: str, is_file: bool = False) -> str:
        """Get Python execution command."""
        if is_file:
            return f"python {code_or_path}"
        else:
            # Escape the code for shell execution
            escaped_code = code_or_path.replace("'", "'\"'\"'")
            return f"python -c '{escaped_code}'"

    def get_file_extension(self) -> str:
        return ".py"


@dataclass
class BashTemplate(SandboxTemplate):
    """
    Template for Bash script execution.

    Suitable for:
    - Git operations
    - Shell scripts
    - System commands
    - File operations

    Includes common command-line tools and git.
    """

    name: str = "bash"
    description: str = "Bash execution environment with git and common tools"
    e2b_template_id: str = "base"  # E2B base template with bash

    # Bash-specific settings
    shell: str = "/bin/bash"
    apt_packages: list[str] = field(
        default_factory=lambda: [
            "git",
            "curl",
            "jq",
            "tree",
        ]
    )

    def __post_init__(self):
        # Add apt install to setup commands
        if self.apt_packages:
            packages = " ".join(self.apt_packages)
            self.setup_commands = [
                f"apt-get update -qq && apt-get install -qq -y {packages}",
            ]

    def get_execution_command(self, code_or_path: str, is_file: bool = False) -> str:
        """Get Bash execution command."""
        if is_file:
            return f"{self.shell} {code_or_path}"
        else:
            # Escape the code for shell execution
            escaped_code = code_or_path.replace("'", "'\"'\"'")
            return f"{self.shell} -c '{escaped_code}'"

    def get_file_extension(self) -> str:
        return ".sh"


@dataclass
class MoleculeTemplate(PythonTemplate):
    """
    Specialized template for Molecule test execution.

    Extends PythonTemplate with Molecule-specific configuration
    and additional packages commonly needed for Ansible testing.
    """

    name: str = "molecule"
    description: str = "Molecule test execution environment"
    e2b_template_id: str = "python"

    # Extended timeout for molecule tests
    default_timeout: int = 600  # 10 minutes

    pip_packages: list[str] = field(
        default_factory=lambda: [
            "molecule>=6.0.0",
            "molecule-plugins[docker]>=23.0.0",
            "ansible-core>=2.15.0",
            "ansible-lint>=24.0.0",
            "pytest>=8.0.0",
            "pytest-testinfra>=10.0.0",
            "docker>=6.0.0",
        ]
    )

    def get_execution_command(self, code_or_path: str, is_file: bool = False) -> str:
        """Get Molecule execution command."""
        if is_file:
            # If it's a file, run it with Python
            return f"python {code_or_path}"
        else:
            # If it's a molecule command string, run it directly
            if code_or_path.startswith("molecule "):
                return code_or_path
            return f"molecule {code_or_path}"


# Template registry
_TEMPLATES: dict[str, type[SandboxTemplate]] = {
    "python": PythonTemplate,
    "bash": BashTemplate,
    "molecule": MoleculeTemplate,
}


def get_template(name: str, **kwargs) -> SandboxTemplate:
    """
    Get a sandbox template by name.

    Args:
        name: Template name ("python", "bash", "molecule")
        **kwargs: Override template default values

    Returns:
        Configured SandboxTemplate instance

    Raises:
        ValueError: If template name is not recognized

    Example:
        # Get default Python template
        template = get_template("python")

        # Get Python template with custom timeout
        template = get_template("python", default_timeout=600)
    """
    template_class = _TEMPLATES.get(name.lower())
    if template_class is None:
        available = ", ".join(_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available: {available}")

    return template_class(**kwargs)


def register_template(name: str, template_class: type[SandboxTemplate]) -> None:
    """
    Register a custom sandbox template.

    Args:
        name: Name to register the template under
        template_class: Template class (must be subclass of SandboxTemplate)

    Example:
        @dataclass
        class CustomTemplate(SandboxTemplate):
            name: str = "custom"
            # ... custom configuration

        register_template("custom", CustomTemplate)
    """
    if not issubclass(template_class, SandboxTemplate):
        raise TypeError(f"{template_class} must be a subclass of SandboxTemplate")
    _TEMPLATES[name.lower()] = template_class


def list_templates() -> list[str]:
    """
    List all available template names.

    Returns:
        List of registered template names
    """
    return list(_TEMPLATES.keys())
