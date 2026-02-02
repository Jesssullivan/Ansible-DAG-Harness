"""Testing skill for autonomous test execution and analysis.

This skill provides capabilities for:
- Running tests with various frameworks
- Analyzing test failures
- Generating test reports
- Identifying flaky tests
- Suggesting fixes for failures

Supports pytest, molecule, and other test frameworks.
"""

import json
import logging
import re
import subprocess
import xml.etree.ElementTree as ET
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
class TestFailure:
    """Information about a test failure.

    Attributes:
        test_name: Name of the failing test
        test_file: Path to test file
        error_message: Error message
        traceback: Full traceback if available
        failure_type: Type of failure (assertion, error, etc.)
        line_number: Line number where failure occurred
        duration_s: Test duration in seconds
    """

    test_name: str
    test_file: str | None = None
    error_message: str = ""
    traceback: str | None = None
    failure_type: str = "failure"
    line_number: int | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "test_file": self.test_file,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "failure_type": self.failure_type,
            "line_number": self.line_number,
            "duration_s": self.duration_s,
        }


@dataclass
class TestResult:
    """Result of a test run.

    Attributes:
        passed: Number of passed tests
        failed: Number of failed tests
        skipped: Number of skipped tests
        errors: Number of error tests
        total: Total number of tests
        duration_s: Total duration in seconds
        failures: List of failure details
        output: Raw test output
        xml_report: Path to XML report if generated
    """

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0
    duration_s: float = 0.0
    failures: list[TestFailure] = field(default_factory=list)
    output: str = ""
    xml_report: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total": self.total,
            "duration_s": self.duration_s,
            "failures": [f.to_dict() for f in self.failures],
            "has_output": len(self.output) > 0,
            "xml_report": self.xml_report,
        }

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100


class TestingSkill(Skill):
    """Skill for running and analyzing tests.

    Provides actions for:
    - run_tests: Execute tests with specified framework
    - analyze_failures: Analyze test failures and suggest fixes
    - get_coverage: Get test coverage information
    - find_flaky: Identify flaky tests through multiple runs

    Configuration:
        - default_framework: Default test framework (pytest, molecule)
        - coverage_enabled: Enable coverage collection
        - xml_output: Generate XML reports
        - max_output_lines: Maximum output lines to capture
    """

    def __init__(
        self,
        enabled: bool = True,
        config: dict | None = None,
    ):
        """Initialize testing skill.

        Args:
            enabled: Whether skill is active
            config: Optional configuration
        """
        config = config or {}
        config.setdefault("default_framework", "pytest")
        config.setdefault("coverage_enabled", False)
        config.setdefault("xml_output", True)
        config.setdefault("max_output_lines", 500)
        super().__init__(enabled=enabled, config=config)

    @property
    def name(self) -> str:
        return "testing"

    @property
    def description(self) -> str:
        return "Run tests, analyze failures, and generate reports"

    def _register_actions(self) -> None:
        """Register testing actions."""
        self.register_action(
            SkillAction(
                name="run_tests",
                description="Run tests with specified framework",
                handler=self._run_tests,
                parameters=["path", "framework", "pattern", "markers", "verbose"],
                required_params=[],
            )
        )

        self.register_action(
            SkillAction(
                name="analyze_failures",
                description="Analyze test failures and suggest fixes",
                handler=self._analyze_failures,
                parameters=["test_result", "include_suggestions"],
                required_params=["test_result"],
            )
        )

        self.register_action(
            SkillAction(
                name="get_coverage",
                description="Get test coverage report",
                handler=self._get_coverage,
                parameters=["path", "source"],
                required_params=[],
            )
        )

        self.register_action(
            SkillAction(
                name="find_flaky",
                description="Find flaky tests by running multiple times",
                handler=self._find_flaky,
                parameters=["path", "runs", "pattern"],
                required_params=[],
            )
        )

    async def _run_tests(
        self,
        context: SkillContext,
        params: dict,
    ) -> SkillResult:
        """Run tests and return results.

        Args:
            context: Skill context
            params: Parameters (path, framework, pattern, markers, verbose)

        Returns:
            SkillResult with TestResult data
        """
        path = params.get("path", ".")
        framework = params.get("framework", self.config["default_framework"])
        pattern = params.get("pattern")
        markers = params.get("markers")
        verbose = params.get("verbose", True)

        full_path = context.working_dir / path

        if framework == "pytest":
            test_result = await self._run_pytest(full_path, pattern, markers, verbose, context)
        elif framework == "molecule":
            test_result = await self._run_molecule(full_path, context)
        else:
            return SkillResult.error(f"Unknown test framework: {framework}")

        # Determine result status
        if test_result.failed > 0 or test_result.errors > 0:
            status = SkillStatus.FAILURE
            message = f"Tests failed: {test_result.failed} failures, {test_result.errors} errors"
        elif test_result.total == 0:
            status = SkillStatus.SKIPPED
            message = "No tests found"
        else:
            status = SkillStatus.SUCCESS
            message = f"All {test_result.passed} tests passed"

        return SkillResult(
            status=status,
            message=message,
            data=test_result.to_dict(),
            errors=[f.test_name for f in test_result.failures],
            metadata={
                "framework": framework,
                "path": str(path),
                "success_rate": test_result.success_rate,
            },
        )

    async def _run_pytest(
        self,
        path: Path,
        pattern: str | None,
        markers: str | None,
        verbose: bool,
        context: SkillContext,
    ) -> TestResult:
        """Run pytest tests.

        Args:
            path: Path to tests
            pattern: Test name pattern
            markers: Pytest markers
            verbose: Verbose output
            context: Skill context

        Returns:
            TestResult from pytest run
        """
        cmd = ["python", "-m", "pytest"]

        if verbose:
            cmd.append("-v")

        if pattern:
            cmd.extend(["-k", pattern])

        if markers:
            cmd.extend(["-m", markers])

        # XML output for parsing
        xml_path = context.working_dir / ".pytest_result.xml"
        if self.config["xml_output"]:
            cmd.extend(["--junitxml", str(xml_path)])

        # Coverage if enabled
        if self.config["coverage_enabled"]:
            cmd.extend(["--cov", str(path), "--cov-report", "term-missing"])

        cmd.append(str(path))

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(context.working_dir),
                capture_output=True,
                text=True,
                timeout=context.timeout,
                env={**context.env, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            # Parse output
            test_result = self._parse_pytest_output(proc.stdout, proc.stderr)
            test_result.output = proc.stdout[-50000:]  # Limit output size

            # Parse XML if available
            if xml_path.exists():
                test_result.xml_report = str(xml_path)
                self._parse_junit_xml(xml_path, test_result)

            return test_result

        except subprocess.TimeoutExpired:
            return TestResult(
                errors=1,
                total=1,
                output="Test execution timed out",
            )
        except Exception as e:
            logger.error(f"Pytest error: {e}")
            return TestResult(errors=1, total=1, output=str(e))

    async def _run_molecule(
        self,
        path: Path,
        context: SkillContext,
    ) -> TestResult:
        """Run molecule tests.

        Args:
            path: Path to role
            context: Skill context

        Returns:
            TestResult from molecule run
        """
        cmd = ["molecule", "test"]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=context.timeout,
                env=context.env,
            )

            # Parse molecule output
            test_result = self._parse_molecule_output(proc.stdout, proc.stderr)
            test_result.output = proc.stdout[-50000:]

            return test_result

        except subprocess.TimeoutExpired:
            return TestResult(
                errors=1,
                total=1,
                output="Molecule execution timed out",
            )
        except FileNotFoundError:
            return TestResult(
                errors=1,
                total=1,
                output="Molecule not installed",
            )
        except Exception as e:
            logger.error(f"Molecule error: {e}")
            return TestResult(errors=1, total=1, output=str(e))

    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
    ) -> TestResult:
        """Parse pytest console output.

        Args:
            stdout: Standard output
            stderr: Standard error

        Returns:
            TestResult with parsed data
        """
        result = TestResult()

        # Parse summary line: "X passed, Y failed, Z skipped in N.NNs"
        summary_pattern = r"(\d+) passed"
        match = re.search(summary_pattern, stdout)
        if match:
            result.passed = int(match.group(1))

        fail_pattern = r"(\d+) failed"
        match = re.search(fail_pattern, stdout)
        if match:
            result.failed = int(match.group(1))

        skip_pattern = r"(\d+) skipped"
        match = re.search(skip_pattern, stdout)
        if match:
            result.skipped = int(match.group(1))

        error_pattern = r"(\d+) error"
        match = re.search(error_pattern, stdout)
        if match:
            result.errors = int(match.group(1))

        duration_pattern = r"in (\d+\.?\d*)s"
        match = re.search(duration_pattern, stdout)
        if match:
            result.duration_s = float(match.group(1))

        result.total = result.passed + result.failed + result.skipped + result.errors

        return result

    def _parse_molecule_output(
        self,
        stdout: str,
        stderr: str,
    ) -> TestResult:
        """Parse molecule console output.

        Args:
            stdout: Standard output
            stderr: Standard error

        Returns:
            TestResult with parsed data
        """
        result = TestResult()

        # Molecule success/failure is based on return code
        # Look for success indicators
        if "PLAY RECAP" in stdout:
            # Count successful and failed hosts
            ok_pattern = r"ok=(\d+)"
            failed_pattern = r"failed=(\d+)"

            ok_matches = re.findall(ok_pattern, stdout)
            failed_matches = re.findall(failed_pattern, stdout)

            if ok_matches:
                result.passed = sum(int(m) for m in ok_matches)
            if failed_matches:
                total_failed = sum(int(m) for m in failed_matches)
                if total_failed > 0:
                    result.failed = 1
                    result.failures.append(
                        TestFailure(
                            test_name="molecule_test",
                            error_message=f"Ansible failed with {total_failed} failures",
                        )
                    )

        result.total = max(1, result.passed + result.failed)
        return result

    def _parse_junit_xml(
        self,
        xml_path: Path,
        result: TestResult,
    ) -> None:
        """Parse JUnit XML report for detailed failure info.

        Args:
            xml_path: Path to XML file
            result: TestResult to update
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for testcase in root.iter("testcase"):
                name = testcase.get("name", "")
                classname = testcase.get("classname", "")
                time_s = float(testcase.get("time", 0))

                # Check for failure
                failure = testcase.find("failure")
                error = testcase.find("error")

                if failure is not None:
                    result.failures.append(
                        TestFailure(
                            test_name=f"{classname}::{name}",
                            error_message=failure.get("message", ""),
                            traceback=failure.text,
                            failure_type="failure",
                            duration_s=time_s,
                        )
                    )
                elif error is not None:
                    result.failures.append(
                        TestFailure(
                            test_name=f"{classname}::{name}",
                            error_message=error.get("message", ""),
                            traceback=error.text,
                            failure_type="error",
                            duration_s=time_s,
                        )
                    )

        except ET.ParseError as e:
            logger.warning(f"Failed to parse JUnit XML: {e}")
        except Exception as e:
            logger.warning(f"Error reading JUnit XML: {e}")

    async def _analyze_failures(
        self,
        context: SkillContext,
        params: dict,
    ) -> SkillResult:
        """Analyze test failures and provide insights.

        Args:
            context: Skill context
            params: Parameters (test_result, include_suggestions)

        Returns:
            SkillResult with analysis
        """
        test_result_data = params["test_result"]
        include_suggestions = params.get("include_suggestions", True)

        # Parse failures from data
        failures = []
        for f_data in test_result_data.get("failures", []):
            failures.append(TestFailure(**f_data))

        if not failures:
            return SkillResult.success(
                "No failures to analyze",
                data={"analysis": [], "suggestions": []},
            )

        analysis = []
        suggestions = []

        for failure in failures:
            failure_analysis = {
                "test": failure.test_name,
                "type": failure.failure_type,
                "message": failure.error_message,
            }

            # Analyze error patterns
            if "AssertionError" in failure.error_message:
                failure_analysis["category"] = "assertion"
                if include_suggestions:
                    suggestions.append(
                        {
                            "test": failure.test_name,
                            "suggestion": "Check expected vs actual values in assertion",
                        }
                    )

            elif "ImportError" in failure.error_message:
                failure_analysis["category"] = "import"
                if include_suggestions:
                    suggestions.append(
                        {
                            "test": failure.test_name,
                            "suggestion": "Check for missing dependencies or incorrect imports",
                        }
                    )

            elif "AttributeError" in failure.error_message:
                failure_analysis["category"] = "attribute"
                if include_suggestions:
                    suggestions.append(
                        {
                            "test": failure.test_name,
                            "suggestion": "Verify object has expected attributes/methods",
                        }
                    )

            elif "TypeError" in failure.error_message:
                failure_analysis["category"] = "type"
                if include_suggestions:
                    suggestions.append(
                        {
                            "test": failure.test_name,
                            "suggestion": "Check argument types and function signatures",
                        }
                    )

            elif "FileNotFoundError" in failure.error_message:
                failure_analysis["category"] = "file"
                if include_suggestions:
                    suggestions.append(
                        {
                            "test": failure.test_name,
                            "suggestion": "Verify file paths and test fixtures",
                        }
                    )

            else:
                failure_analysis["category"] = "other"

            analysis.append(failure_analysis)

        return SkillResult.success(
            f"Analyzed {len(failures)} failures",
            data={
                "analysis": analysis,
                "suggestions": suggestions,
                "categories": {
                    cat: sum(1 for a in analysis if a.get("category") == cat)
                    for cat in set(a.get("category") for a in analysis)
                },
            },
        )

    async def _get_coverage(
        self,
        context: SkillContext,
        params: dict,
    ) -> SkillResult:
        """Get test coverage report.

        Args:
            context: Skill context
            params: Parameters (path, source)

        Returns:
            SkillResult with coverage data
        """
        # Note: path and source are parsed for future use with coverage configuration
        _ = params.get("path", ".")
        _ = params.get("source", ".")

        cmd = [
            "python",
            "-m",
            "coverage",
            "report",
            "--format=json",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(context.working_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if proc.returncode != 0:
                return SkillResult.failure(
                    "Coverage report failed",
                    errors=[proc.stderr],
                )

            # Parse coverage JSON
            try:
                coverage_data = json.loads(proc.stdout)
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)

                return SkillResult.success(
                    f"Coverage: {total_coverage:.1f}%",
                    data={
                        "total_coverage": total_coverage,
                        "files": coverage_data.get("files", {}),
                    },
                )
            except json.JSONDecodeError:
                # Fall back to text parsing
                return SkillResult.success(
                    "Coverage report generated",
                    data={"raw_output": proc.stdout},
                )

        except FileNotFoundError:
            return SkillResult.error("Coverage tool not installed")
        except subprocess.TimeoutExpired:
            return SkillResult.error("Coverage report timed out")

    async def _find_flaky(
        self,
        context: SkillContext,
        params: dict,
    ) -> SkillResult:
        """Find flaky tests by running multiple times.

        Args:
            context: Skill context
            params: Parameters (path, runs, pattern)

        Returns:
            SkillResult with flaky test information
        """
        path = params.get("path", ".")
        runs = params.get("runs", 3)
        pattern = params.get("pattern")

        # Run tests multiple times
        all_failures: dict[str, int] = {}

        for i in range(runs):
            logger.info(f"Flaky detection run {i + 1}/{runs}")

            result = await self._run_tests(
                context,
                {"path": path, "pattern": pattern},
            )

            if result.data:
                # Track failures
                for failure in result.data.get("failures", []):
                    test_name = failure.get("test_name", "unknown")
                    all_failures[test_name] = all_failures.get(test_name, 0) + 1

                # Track passes (total - failures)
                # This is approximate since we don't have individual pass info

        # Identify flaky tests (failed some but not all runs)
        flaky_tests = []
        for test_name, fail_count in all_failures.items():
            if 0 < fail_count < runs:
                flaky_tests.append(
                    {
                        "test": test_name,
                        "failures": fail_count,
                        "runs": runs,
                        "failure_rate": (fail_count / runs) * 100,
                    }
                )

        if not flaky_tests:
            return SkillResult.success(
                f"No flaky tests found in {runs} runs",
                data={"flaky_tests": [], "total_runs": runs},
            )

        return SkillResult(
            status=SkillStatus.PARTIAL,
            message=f"Found {len(flaky_tests)} potentially flaky tests",
            data={
                "flaky_tests": flaky_tests,
                "total_runs": runs,
            },
            warnings=[f"{t['test']}: {t['failure_rate']:.0f}% failure rate" for t in flaky_tests],
        )
