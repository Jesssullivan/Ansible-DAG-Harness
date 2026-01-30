"""
Node definitions for DAG workflow execution.

Follows LangGraph patterns:
- Nodes are functions that take state and return updates
- Edges define transitions between nodes
- Conditional edges allow branching based on state
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel


class NodeResult(str, Enum):
    """Possible outcomes from node execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    SKIP = "skip"
    RETRY = "retry"
    HUMAN_NEEDED = "human_needed"


@dataclass
class NodeContext:
    """
    Context passed to each node during execution.

    Contains:
    - Current state (mutable, accumulated across nodes)
    - Role being processed
    - Execution metadata
    - Access to state database
    """
    role_name: str
    execution_id: int
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self.state[key] = value

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple state values."""
        self.state.update(updates)


class Node(ABC):
    """
    Abstract base class for workflow nodes.

    Each node represents a discrete step in the workflow.
    Nodes are:
    - Idempotent where possible
    - Checkpointable (state can be saved/restored)
    - Observable (emit events for monitoring)
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.retries = 3
        self.timeout_seconds = 300

    @abstractmethod
    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        """
        Execute the node logic.

        Args:
            ctx: The node context with current state

        Returns:
            Tuple of (result status, state updates)
        """
        pass

    async def rollback(self, ctx: NodeContext) -> None:
        """
        Rollback any changes made by this node.

        Called when the workflow fails after this node completed.
        Override in subclasses where rollback is possible.
        """
        pass

    def can_skip(self, ctx: NodeContext) -> bool:
        """
        Check if this node can be skipped based on current state.

        Override to implement skip logic (e.g., already completed).
        """
        return False


class FunctionNode(Node):
    """Node that wraps a simple function."""

    def __init__(self, name: str, func: Callable[[NodeContext], tuple[NodeResult, dict[str, Any]]],
                 description: str = ""):
        super().__init__(name, description)
        self._func = func

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(ctx)
        return self._func(ctx)


class ConditionalEdge:
    """
    Conditional edge that routes to different nodes based on state.

    Example:
        ConditionalEdge(
            condition=lambda ctx: ctx.get("tests_passed"),
            if_true="create_commit",
            if_false="fix_tests"
        )
    """

    def __init__(self, condition: Callable[[NodeContext], bool],
                 if_true: str, if_false: str):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def evaluate(self, ctx: NodeContext) -> str:
        """Evaluate condition and return next node name."""
        return self.if_true if self.condition(ctx) else self.if_false


class RouterEdge:
    """
    Router edge that can route to multiple possible nodes.

    Example:
        RouterEdge(
            router=lambda ctx: "deploy" if ctx.get("ready") else "wait",
            possible_targets=["deploy", "wait", "abort"]
        )
    """

    def __init__(self, router: Callable[[NodeContext], str],
                 possible_targets: list[str]):
        self.router = router
        self.possible_targets = possible_targets

    def evaluate(self, ctx: NodeContext) -> str:
        """Evaluate router and return next node name."""
        target = self.router(ctx)
        if target not in self.possible_targets:
            raise ValueError(f"Router returned invalid target: {target}")
        return target


# Type alias for edge definitions
Edge = str | ConditionalEdge | RouterEdge


@dataclass
class NodeDefinition:
    """Definition of a node in the workflow graph."""
    name: str
    node: Node
    edges: dict[NodeResult, Edge] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage."""
        edges_dict = {}
        for result, edge in self.edges.items():
            if isinstance(edge, str):
                edges_dict[result.value] = {"type": "direct", "target": edge}
            elif isinstance(edge, ConditionalEdge):
                edges_dict[result.value] = {
                    "type": "conditional",
                    "if_true": edge.if_true,
                    "if_false": edge.if_false
                }
            elif isinstance(edge, RouterEdge):
                edges_dict[result.value] = {
                    "type": "router",
                    "targets": edge.possible_targets
                }
        return {
            "name": self.name,
            "description": self.node.description,
            "retries": self.node.retries,
            "timeout_seconds": self.node.timeout_seconds,
            "edges": edges_dict
        }


# ============================================================================
# BUILT-IN NODES FOR BOX-UP-ROLE WORKFLOW
# ============================================================================

class ValidateRoleNode(Node):
    """Validate that the role exists and extract metadata."""

    def __init__(self):
        super().__init__("validate_role", "Validate role exists and extract metadata")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        from pathlib import Path

        role_path = Path(f"ansible/roles/{ctx.role_name}")
        if not role_path.exists():
            return NodeResult.FAILURE, {"error": f"Role not found: {ctx.role_name}"}

        # Check for molecule tests
        has_molecule = (role_path / "molecule").exists()

        # Check for meta/main.yml
        meta_path = role_path / "meta" / "main.yml"
        has_meta = meta_path.exists()

        return NodeResult.SUCCESS, {
            "role_path": str(role_path),
            "has_molecule_tests": has_molecule,
            "has_meta": has_meta
        }


class AnalyzeDependenciesNode(Node):
    """Analyze role dependencies using analyze-role-deps.py."""

    def __init__(self):
        super().__init__("analyze_dependencies", "Analyze role dependencies and credentials")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import json
        import subprocess

        try:
            result = subprocess.run(
                ["python", "scripts/analyze-role-deps.py", ctx.role_name, "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return NodeResult.FAILURE, {"error": result.stderr}

            analysis = json.loads(result.stdout)
            return NodeResult.SUCCESS, {
                "wave": analysis.get("wave", 0),
                "wave_name": analysis.get("wave_name", ""),
                "explicit_deps": analysis.get("explicit_deps", []),
                "implicit_deps": analysis.get("implicit_deps", []),
                "credentials": analysis.get("credentials", []),
                "reverse_deps": analysis.get("reverse_deps", []),
                "tags": analysis.get("tags", [])
            }

        except subprocess.TimeoutExpired:
            return NodeResult.RETRY, {"error": "Analysis timed out"}
        except json.JSONDecodeError as e:
            return NodeResult.FAILURE, {"error": f"Invalid JSON output: {e}"}
        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}


class CheckReverseDepsNode(Node):
    """Check if reverse dependencies are already boxed up."""

    def __init__(self):
        super().__init__("check_reverse_deps", "Verify reverse dependencies are boxed up first")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess

        reverse_deps = ctx.get("reverse_deps", [])
        if not reverse_deps:
            return NodeResult.SUCCESS, {"blocking_deps": []}

        blocking = []
        for dep in reverse_deps:
            # Check if branch exists on origin
            result = subprocess.run(
                ["git", "ls-remote", "--heads", "origin", f"sid/{dep}"],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                blocking.append(dep)

        if blocking:
            return NodeResult.FAILURE, {
                "blocking_deps": blocking,
                "error": f"Must box up first: {', '.join(blocking)}"
            }

        return NodeResult.SUCCESS, {"blocking_deps": []}


class CreateWorktreeNode(Node):
    """Create git worktree for isolated development."""

    def __init__(self):
        super().__init__("create_worktree", "Create isolated git worktree")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess

        try:
            result = subprocess.run(
                ["scripts/create-role-worktree.sh", ctx.role_name],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return NodeResult.FAILURE, {"error": result.stderr}

            worktree_path = f"../sid-{ctx.role_name}"
            return NodeResult.SUCCESS, {
                "worktree_path": worktree_path,
                "branch": f"sid/{ctx.role_name}"
            }

        except subprocess.TimeoutExpired:
            return NodeResult.RETRY, {"error": "Worktree creation timed out"}
        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}

    async def rollback(self, ctx: NodeContext) -> None:
        import subprocess
        worktree_path = ctx.get("worktree_path")
        if worktree_path:
            subprocess.run(
                ["git", "worktree", "remove", worktree_path, "--force"],
                capture_output=True
            )


class RunMoleculeTestsNode(Node):
    """Run molecule tests for the role."""

    def __init__(self):
        super().__init__("run_molecule_tests", "Execute molecule tests (blocking)")
        self.timeout_seconds = 600  # 10 minutes

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess
        import time

        if not ctx.get("has_molecule_tests", False):
            return NodeResult.SKIP, {"molecule_skipped": True, "reason": "No molecule tests"}

        start_time = time.time()
        try:
            result = subprocess.run(
                ["npm", "run", "molecule:role", f"--role={ctx.role_name}"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=ctx.get("worktree_path", ".")
            )

            duration = int(time.time() - start_time)

            if result.returncode != 0:
                return NodeResult.FAILURE, {
                    "molecule_passed": False,
                    "molecule_duration": duration,
                    "molecule_output": result.stdout[-5000:],  # Last 5KB
                    "error": "Molecule tests failed"
                }

            return NodeResult.SUCCESS, {
                "molecule_passed": True,
                "molecule_duration": duration
            }

        except subprocess.TimeoutExpired:
            return NodeResult.FAILURE, {
                "molecule_passed": False,
                "error": "Molecule tests timed out"
            }


class CreateCommitNode(Node):
    """Create commit with semantic message."""

    def __init__(self):
        super().__init__("create_commit", "Create signed commit")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess

        worktree_path = ctx.get("worktree_path", ".")
        wave = ctx.get("wave", 0)
        wave_name = ctx.get("wave_name", "")

        commit_msg = f"""feat({ctx.role_name}): Add {ctx.role_name} Ansible role

Wave {wave}: {wave_name}

- Molecule tests passing
- Ready for merge train
"""

        try:
            # Stage all changes
            subprocess.run(["git", "add", "-A"], cwd=worktree_path, check=True)

            # Check if there are changes to commit
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=worktree_path,
                capture_output=True,
                text=True
            )

            if not status.stdout.strip():
                return NodeResult.SKIP, {"commit_skipped": True, "reason": "No changes to commit"}

            # Create commit
            result = subprocess.run(
                ["git", "commit",
                 "--author=Jess Sullivan <jsullivan2@bates.edu>",
                 "-m", commit_msg],
                cwd=worktree_path,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return NodeResult.FAILURE, {"error": result.stderr}

            # Get commit SHA
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=worktree_path,
                capture_output=True,
                text=True
            )

            return NodeResult.SUCCESS, {
                "commit_sha": sha_result.stdout.strip(),
                "commit_message": commit_msg
            }

        except subprocess.CalledProcessError as e:
            return NodeResult.FAILURE, {"error": str(e)}


class PushBranchNode(Node):
    """Push branch to origin."""

    def __init__(self):
        super().__init__("push_branch", "Push branch to origin with tracking")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess

        worktree_path = ctx.get("worktree_path", ".")
        branch = ctx.get("branch", f"sid/{ctx.role_name}")

        try:
            result = subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=worktree_path,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return NodeResult.FAILURE, {"error": result.stderr}

            return NodeResult.SUCCESS, {"pushed": True}

        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}


class CreateGitLabIssueNode(Node):
    """Create GitLab issue with iteration assignment."""

    def __init__(self):
        super().__init__("create_gitlab_issue", "Create GitLab issue")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import json
        import subprocess

        try:
            result = subprocess.run(
                ["scripts/create-gitlab-issues.sh", ctx.role_name, "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return NodeResult.FAILURE, {"error": result.stderr}

            issue_data = json.loads(result.stdout)
            return NodeResult.SUCCESS, {
                "issue_url": issue_data.get("issue_url"),
                "issue_iid": issue_data.get("issue_iid"),
                "iteration_assigned": issue_data.get("iteration_id") is not None
            }

        except subprocess.TimeoutExpired:
            return NodeResult.RETRY, {"error": "Issue creation timed out"}
        except json.JSONDecodeError:
            # Try to extract URL from non-JSON output
            return NodeResult.FAILURE, {"error": "Failed to parse issue creation output"}
        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}


class CreateMergeRequestNode(Node):
    """Create GitLab merge request."""

    def __init__(self):
        super().__init__("create_merge_request", "Create GitLab merge request")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import json
        import subprocess

        issue_iid = ctx.get("issue_iid")
        if not issue_iid:
            return NodeResult.FAILURE, {"error": "No issue IID available"}

        try:
            result = subprocess.run(
                ["scripts/create-gitlab-mr.sh", ctx.role_name,
                 "--issue", str(issue_iid), "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return NodeResult.FAILURE, {"error": result.stderr}

            mr_data = json.loads(result.stdout)
            return NodeResult.SUCCESS, {
                "mr_url": mr_data.get("mr_url"),
                "mr_iid": mr_data.get("mr_iid")
            }

        except subprocess.TimeoutExpired:
            return NodeResult.RETRY, {"error": "MR creation timed out"}
        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}


class ReportSummaryNode(Node):
    """Generate and report final summary."""

    def __init__(self):
        super().__init__("report_summary", "Generate workflow summary")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        summary = {
            "role": ctx.role_name,
            "wave": ctx.get("wave"),
            "wave_name": ctx.get("wave_name"),
            "worktree_path": ctx.get("worktree_path"),
            "branch": ctx.get("branch"),
            "commit_sha": ctx.get("commit_sha"),
            "issue_url": ctx.get("issue_url"),
            "mr_url": ctx.get("mr_url"),
            "molecule_passed": ctx.get("molecule_passed"),
            "credentials": ctx.get("credentials", []),
            "dependencies": ctx.get("explicit_deps", [])
        }

        return NodeResult.SUCCESS, {"summary": summary}
