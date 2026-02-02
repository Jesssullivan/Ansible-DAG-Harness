"""
SQLite state management with graph-queryable patterns.

Uses adjacency list pattern with recursive CTEs for dependency traversal.
"""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.db.models import (
    ActiveRegressionView,
    ContextCapability,
    Credential,
    CyclicDependencyError,
    ExecutionContext,
    Issue,
    Iteration,
    MergeRequest,
    MergeTrainEntry,
    MergeTrainStatus,
    NodeStatus,
    RegressionStatus,
    Role,
    RoleDependency,
    RoleStatusView,
    TestRegression,
    TestRun,
    TestStatus,
    TestType,
    WorkflowStatus,
    Worktree,
    WorktreeStatus,
)


class StateDB:
    """
    SQLite state management with graph-queryable patterns.

    Provides:
    - CRUD operations for all entities
    - Graph traversal via recursive CTEs
    - Checkpoint/restore for workflow execution
    - Audit logging for all mutations
    """

    def __init__(self, db_path: str | Path = "harness.db"):
        self.db_path = Path(db_path)
        # For in-memory databases, keep a persistent connection to maintain
        # the database across multiple operations
        self._is_memory = str(db_path) == ":memory:"
        self._persistent_conn = None
        if self._is_memory:
            # Create persistent connection that keeps the in-memory db alive
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.execute("PRAGMA foreign_keys = ON")
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database with schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        with self.connection() as conn:
            conn.executescript(schema_path.read_text())

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        if self._is_memory and self._persistent_conn:
            # For in-memory databases, use the persistent connection
            # We don't close it, just commit
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # =========================================================================
    # ROLE OPERATIONS
    # =========================================================================

    def upsert_role(self, role: Role) -> int:
        """Insert or update a role."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO roles (name, wave, wave_name, description, molecule_path, has_molecule_tests)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    wave = excluded.wave,
                    wave_name = excluded.wave_name,
                    description = excluded.description,
                    molecule_path = excluded.molecule_path,
                    has_molecule_tests = excluded.has_molecule_tests
                RETURNING id
                """,
                (
                    role.name,
                    role.wave,
                    role.wave_name,
                    role.description,
                    role.molecule_path,
                    role.has_molecule_tests,
                ),
            )
            return cursor.fetchone()[0]

    def get_role(self, name: str) -> Role | None:
        """Get role by name."""
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM roles WHERE name = ?", (name,)).fetchone()
            return Role(**dict(row)) if row else None

    def get_role_by_id(self, role_id: int) -> Role | None:
        """Get role by ID."""
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM roles WHERE id = ?", (role_id,)).fetchone()
            return Role(**dict(row)) if row else None

    def list_roles(self, wave: int | None = None) -> list[Role]:
        """List all roles, optionally filtered by wave."""
        with self.connection() as conn:
            if wave is not None:
                rows = conn.execute(
                    "SELECT * FROM roles WHERE wave = ? ORDER BY name", (wave,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM roles ORDER BY wave, name").fetchall()
            return [Role(**dict(row)) for row in rows]

    def get_role_status(self, name: str) -> RoleStatusView | None:
        """Get aggregated role status from view."""
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM v_role_status WHERE name = ?", (name,)).fetchone()
            return RoleStatusView(**dict(row)) if row else None

    def list_role_statuses(self) -> list[RoleStatusView]:
        """List all role statuses."""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM v_role_status ORDER BY wave, name").fetchall()
            return [RoleStatusView(**dict(row)) for row in rows]

    # =========================================================================
    # DEPENDENCY GRAPH OPERATIONS (Recursive CTEs)
    # =========================================================================

    def add_dependency(self, dep: RoleDependency) -> int:
        """Add a role dependency."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO role_dependencies (role_id, depends_on_id, dependency_type, source_file)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(role_id, depends_on_id, dependency_type) DO UPDATE SET
                    source_file = excluded.source_file
                RETURNING id
                """,
                (dep.role_id, dep.depends_on_id, dep.dependency_type.value, dep.source_file),
            )
            return cursor.fetchone()[0]

    def get_dependencies(self, role_name: str, transitive: bool = False) -> list[tuple[str, int]]:
        """
        Get dependencies of a role.

        Args:
            role_name: Name of the role
            transitive: If True, return all transitive dependencies using recursive CTE

        Returns:
            List of (role_name, depth) tuples
        """
        with self.connection() as conn:
            role = self.get_role(role_name)
            if not role or not role.id:
                return []

            if transitive:
                # Recursive CTE for transitive dependencies
                rows = conn.execute(
                    """
                    WITH RECURSIVE deps(role_id, depth) AS (
                        SELECT depends_on_id, 1
                        FROM role_dependencies
                        WHERE role_id = ?
                        UNION ALL
                        SELECT rd.depends_on_id, d.depth + 1
                        FROM role_dependencies rd
                        JOIN deps d ON rd.role_id = d.role_id
                        WHERE d.depth < 10
                    )
                    SELECT DISTINCT r.name, MIN(d.depth) as depth
                    FROM deps d
                    JOIN roles r ON d.role_id = r.id
                    GROUP BY r.name
                    ORDER BY depth, r.name
                    """,
                    (role.id,),
                ).fetchall()
            else:
                # Direct dependencies only
                rows = conn.execute(
                    """
                    SELECT r.name, 1 as depth
                    FROM role_dependencies rd
                    JOIN roles r ON rd.depends_on_id = r.id
                    WHERE rd.role_id = ?
                    ORDER BY r.name
                    """,
                    (role.id,),
                ).fetchall()

            return [(row["name"], row["depth"]) for row in rows]

    def get_reverse_dependencies(
        self, role_name: str, transitive: bool = False
    ) -> list[tuple[str, int]]:
        """
        Get roles that depend on the given role.

        Args:
            role_name: Name of the role
            transitive: If True, return all transitive reverse dependencies

        Returns:
            List of (role_name, depth) tuples
        """
        with self.connection() as conn:
            role = self.get_role(role_name)
            if not role or not role.id:
                return []

            if transitive:
                # Recursive CTE for transitive reverse dependencies
                rows = conn.execute(
                    """
                    WITH RECURSIVE reverse_deps(role_id, depth) AS (
                        SELECT role_id, 1
                        FROM role_dependencies
                        WHERE depends_on_id = ?
                        UNION ALL
                        SELECT rd.role_id, rd2.depth + 1
                        FROM role_dependencies rd
                        JOIN reverse_deps rd2 ON rd.depends_on_id = rd2.role_id
                        WHERE rd2.depth < 10
                    )
                    SELECT DISTINCT r.name, MIN(rd.depth) as depth
                    FROM reverse_deps rd
                    JOIN roles r ON rd.role_id = r.id
                    GROUP BY r.name
                    ORDER BY depth, r.name
                    """,
                    (role.id,),
                ).fetchall()
            else:
                # Direct reverse dependencies only
                rows = conn.execute(
                    """
                    SELECT r.name, 1 as depth
                    FROM role_dependencies rd
                    JOIN roles r ON rd.role_id = r.id
                    WHERE rd.depends_on_id = ?
                    ORDER BY r.name
                    """,
                    (role.id,),
                ).fetchall()

            return [(row["name"], row["depth"]) for row in rows]

    def get_dependency_graph(self) -> list[dict[str, Any]]:
        """Get full dependency graph for visualization."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM v_dependency_graph ORDER BY from_wave, from_role"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_deployment_order(self, raise_on_cycle: bool = True) -> list[str]:
        """
        Get topologically sorted deployment order.

        Returns roles in order such that dependencies come before dependents.
        Uses Kahn's algorithm over the dependency graph.

        Args:
            raise_on_cycle: If True, raises CyclicDependencyError when a cycle is detected.
                           If False, returns partial order (nodes not in cycle).

        Raises:
            CyclicDependencyError: If a cycle is detected and raise_on_cycle is True.
        """
        with self.connection() as conn:
            # Get all roles
            roles = {
                row["name"]: row["id"]
                for row in conn.execute("SELECT id, name FROM roles").fetchall()
            }

            if not roles:
                return []

            # Build adjacency list and in-degree count
            in_degree: dict[str, int] = {name: 0 for name in roles}
            graph: dict[str, list[str]] = {name: [] for name in roles}

            for row in conn.execute(
                """
                SELECT r1.name as from_role, r2.name as to_role
                FROM role_dependencies rd
                JOIN roles r1 ON rd.role_id = r1.id
                JOIN roles r2 ON rd.depends_on_id = r2.id
                """
            ).fetchall():
                from_role, to_role = row["from_role"], row["to_role"]
                graph[to_role].append(from_role)
                in_degree[from_role] += 1

            # Kahn's algorithm
            queue = [name for name, degree in in_degree.items() if degree == 0]
            result = []

            while queue:
                # Sort by wave for consistent ordering within same dependency level
                queue.sort(
                    key=lambda n: (
                        conn.execute("SELECT wave FROM roles WHERE name = ?", (n,)).fetchone()[
                            "wave"
                        ],
                        n,
                    )
                )
                node = queue.pop(0)
                result.append(node)

                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            # Check for cycles: if not all nodes processed, there's a cycle
            if len(result) != len(roles):
                # Find nodes involved in cycles (those with remaining in-degree)
                cycle_nodes = [name for name, degree in in_degree.items() if degree > 0]

                if raise_on_cycle:
                    # Find the actual cycle path for a better error message
                    cycle_path = self._find_cycle_path(graph, cycle_nodes)
                    raise CyclicDependencyError(
                        f"Cyclic dependency detected: {' -> '.join(cycle_path)}",
                        cycle_path=cycle_path,
                    )

            return result

    def _find_cycle_path(self, graph: dict[str, list[str]], cycle_nodes: list[str]) -> list[str]:
        """
        Find the actual path of a cycle using DFS.

        Args:
            graph: Adjacency list (to_role -> [from_role, ...])
            cycle_nodes: Nodes known to be in a cycle

        Returns:
            List of node names forming a cycle path
        """
        # Build reverse graph for DFS from cycle nodes
        reverse_graph: dict[str, list[str]] = {n: [] for n in cycle_nodes}
        for to_role, from_roles in graph.items():
            if to_role in cycle_nodes:
                for from_role in from_roles:
                    if from_role in cycle_nodes:
                        reverse_graph[from_role].append(to_role)

        # DFS to find cycle
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in reverse_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle - trim path to just the cycle
                    cycle_start = path.index(neighbor)
                    path.append(neighbor)  # Close the cycle
                    del path[:cycle_start]
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        # Try DFS from each cycle node
        for node in cycle_nodes:
            if node not in visited:
                if dfs(node):
                    return path

        # Fallback: just return the cycle nodes
        return cycle_nodes + [cycle_nodes[0]] if cycle_nodes else []

    def detect_cycles(self) -> list[list[str]]:
        """
        Detect all cycles in the dependency graph.

        Returns:
            List of cycle paths. Each cycle path is a list of role names
            where the last element connects back to the first.
            Empty list if no cycles exist.
        """
        with self.connection() as conn:
            # Get all roles
            roles = list(row["name"] for row in conn.execute("SELECT name FROM roles").fetchall())

            # Build adjacency list: role -> [roles it depends on]
            deps_graph: dict[str, list[str]] = {name: [] for name in roles}

            for row in conn.execute(
                """
                SELECT r1.name as role, r2.name as depends_on
                FROM role_dependencies rd
                JOIN roles r1 ON rd.role_id = r1.id
                JOIN roles r2 ON rd.depends_on_id = r2.id
                """
            ).fetchall():
                deps_graph[row["role"]].append(row["depends_on"])

            # Find all cycles using Tarjan's algorithm for SCCs
            cycles = []
            index_counter = [0]
            stack: list[str] = []
            lowlinks: dict[str, int] = {}
            index: dict[str, int] = {}
            on_stack: set[str] = set()

            def strongconnect(node: str):
                index[node] = index_counter[0]
                lowlinks[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack.add(node)

                for neighbor in deps_graph.get(node, []):
                    if neighbor not in index:
                        strongconnect(neighbor)
                        lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                    elif neighbor in on_stack:
                        lowlinks[node] = min(lowlinks[node], index[neighbor])

                # If node is a root node, pop the stack and generate an SCC
                if lowlinks[node] == index[node]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on_stack.remove(w)
                        scc.append(w)
                        if w == node:
                            break

                    # Only report SCCs with more than one node (actual cycles)
                    # or self-loops
                    if len(scc) > 1:
                        scc.reverse()
                        scc.append(scc[0])  # Close the cycle
                        cycles.append(scc)
                    elif len(scc) == 1 and scc[0] in deps_graph.get(scc[0], []):
                        # Self-loop
                        cycles.append([scc[0], scc[0]])

            for node in roles:
                if node not in index:
                    strongconnect(node)

            return cycles

    def validate_dependencies(self) -> dict[str, Any]:
        """
        Validate the dependency graph for issues.

        Returns:
            Dict with validation results:
            - 'valid': True if no issues found
            - 'cycles': List of cycle paths if any
            - 'missing_deps': List of dependencies referencing non-existent roles
            - 'self_deps': List of roles that depend on themselves
        """
        cycles = self.detect_cycles()

        with self.connection() as conn:
            # Check for dependencies on non-existent roles
            missing = conn.execute(
                """
                SELECT r1.name as role, rd.depends_on_id
                FROM role_dependencies rd
                JOIN roles r1 ON rd.role_id = r1.id
                LEFT JOIN roles r2 ON rd.depends_on_id = r2.id
                WHERE r2.id IS NULL
                """
            ).fetchall()
            missing_deps = [dict(row) for row in missing]

            # Check for self-dependencies
            self_deps = conn.execute(
                """
                SELECT r.name
                FROM role_dependencies rd
                JOIN roles r ON rd.role_id = r.id
                WHERE rd.role_id = rd.depends_on_id
                """
            ).fetchall()
            self_dep_roles = [row["name"] for row in self_deps]

        return {
            "valid": len(cycles) == 0 and len(missing_deps) == 0 and len(self_dep_roles) == 0,
            "cycles": cycles,
            "missing_deps": missing_deps,
            "self_deps": self_dep_roles,
        }

    # =========================================================================
    # CREDENTIAL OPERATIONS
    # =========================================================================

    def add_credential(self, cred: Credential) -> int:
        """Add a credential requirement."""
        # Use empty string instead of NULL for attribute to enable proper upsert
        # SQLite treats NULLs as distinct in unique constraints
        attr = cred.attribute if cred.attribute is not None else ""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO credentials (role_id, entry_name, purpose, is_base58, attribute, source_file, source_line)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(role_id, entry_name, attribute) DO UPDATE SET
                    purpose = excluded.purpose,
                    is_base58 = excluded.is_base58,
                    source_file = excluded.source_file,
                    source_line = excluded.source_line
                RETURNING id
                """,
                (
                    cred.role_id,
                    cred.entry_name,
                    cred.purpose,
                    cred.is_base58,
                    attr,
                    cred.source_file,
                    cred.source_line,
                ),
            )
            return cursor.fetchone()[0]

    def get_credentials(self, role_name: str) -> list[Credential]:
        """Get credentials for a role."""
        with self.connection() as conn:
            role = self.get_role(role_name)
            if not role or not role.id:
                return []
            rows = conn.execute(
                "SELECT * FROM credentials WHERE role_id = ?", (role.id,)
            ).fetchall()
            return [Credential(**dict(row)) for row in rows]

    # =========================================================================
    # WORKTREE OPERATIONS
    # =========================================================================

    def upsert_worktree(self, worktree: Worktree) -> int:
        """Insert or update a worktree."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO worktrees (role_id, path, branch, base_commit, current_commit,
                                       commits_ahead, commits_behind, uncommitted_changes, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(role_id) DO UPDATE SET
                    path = excluded.path,
                    branch = excluded.branch,
                    base_commit = excluded.base_commit,
                    current_commit = excluded.current_commit,
                    commits_ahead = excluded.commits_ahead,
                    commits_behind = excluded.commits_behind,
                    uncommitted_changes = excluded.uncommitted_changes,
                    status = excluded.status
                RETURNING id
                """,
                (
                    worktree.role_id,
                    worktree.path,
                    worktree.branch,
                    worktree.base_commit,
                    worktree.current_commit,
                    worktree.commits_ahead,
                    worktree.commits_behind,
                    worktree.uncommitted_changes,
                    worktree.status.value,
                ),
            )
            return cursor.fetchone()[0]

    def get_worktree(self, role_name: str) -> Worktree | None:
        """Get worktree for a role."""
        with self.connection() as conn:
            role = self.get_role(role_name)
            if not role or not role.id:
                return None
            row = conn.execute("SELECT * FROM worktrees WHERE role_id = ?", (role.id,)).fetchone()
            if row:
                data = dict(row)
                data["status"] = WorktreeStatus(data["status"])
                return Worktree(**data)
            return None

    def list_worktrees(self, status: WorktreeStatus | None = None) -> list[Worktree]:
        """List all worktrees."""
        with self.connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM worktrees WHERE status = ?", (status.value,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM worktrees").fetchall()
            result = []
            for row in rows:
                data = dict(row)
                data["status"] = WorktreeStatus(data["status"])
                result.append(Worktree(**data))
            return result

    # =========================================================================
    # GITLAB OPERATIONS
    # =========================================================================

    def upsert_iteration(self, iteration: Iteration) -> int:
        """Insert or update an iteration."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO iterations (id, title, state, start_date, due_date, group_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    state = excluded.state,
                    start_date = excluded.start_date,
                    due_date = excluded.due_date,
                    group_id = excluded.group_id
                """,
                (
                    iteration.id,
                    iteration.title,
                    iteration.state,
                    iteration.start_date,
                    iteration.due_date,
                    iteration.group_id,
                ),
            )
            return iteration.id

    def upsert_issue(self, issue: Issue) -> int:
        """Insert or update an issue."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO issues (id, iid, role_id, iteration_id, title, state, web_url, labels, assignee, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    iid = excluded.iid,
                    role_id = excluded.role_id,
                    iteration_id = excluded.iteration_id,
                    title = excluded.title,
                    state = excluded.state,
                    web_url = excluded.web_url,
                    labels = excluded.labels,
                    assignee = excluded.assignee,
                    weight = excluded.weight
                """,
                (
                    issue.id,
                    issue.iid,
                    issue.role_id,
                    issue.iteration_id,
                    issue.title,
                    issue.state,
                    issue.web_url,
                    issue.labels,
                    issue.assignee,
                    issue.weight,
                ),
            )
            return issue.id

    def upsert_merge_request(self, mr: MergeRequest) -> int:
        """Insert or update a merge request."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO merge_requests (id, iid, role_id, issue_id, source_branch, target_branch,
                                           title, state, web_url, merge_status, squash_on_merge, remove_source_branch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    iid = excluded.iid,
                    role_id = excluded.role_id,
                    issue_id = excluded.issue_id,
                    source_branch = excluded.source_branch,
                    target_branch = excluded.target_branch,
                    title = excluded.title,
                    state = excluded.state,
                    web_url = excluded.web_url,
                    merge_status = excluded.merge_status,
                    squash_on_merge = excluded.squash_on_merge,
                    remove_source_branch = excluded.remove_source_branch
                """,
                (
                    mr.id,
                    mr.iid,
                    mr.role_id,
                    mr.issue_id,
                    mr.source_branch,
                    mr.target_branch,
                    mr.title,
                    mr.state,
                    mr.web_url,
                    mr.merge_status,
                    mr.squash_on_merge,
                    mr.remove_source_branch,
                ),
            )
            return mr.id

    def get_issue(self, role_name: str) -> Issue | None:
        """Get issue for a role."""
        with self.connection() as conn:
            role = self.get_role(role_name)
            if not role or not role.id:
                return None
            row = conn.execute(
                "SELECT * FROM issues WHERE role_id = ? ORDER BY created_at DESC LIMIT 1",
                (role.id,),
            ).fetchone()
            return Issue(**dict(row)) if row else None

    def get_merge_request(self, role_name: str) -> MergeRequest | None:
        """Get merge request for a role."""
        with self.connection() as conn:
            role = self.get_role(role_name)
            if not role or not role.id:
                return None
            row = conn.execute(
                "SELECT * FROM merge_requests WHERE role_id = ? ORDER BY created_at DESC LIMIT 1",
                (role.id,),
            ).fetchone()
            return MergeRequest(**dict(row)) if row else None

    # =========================================================================
    # WORKFLOW EXECUTION OPERATIONS
    # =========================================================================

    def create_workflow_definition(
        self, name: str, description: str, nodes: list[dict], edges: list[dict]
    ) -> int:
        """Create a workflow definition."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO workflow_definitions (name, description, nodes_json, edges_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    nodes_json = excluded.nodes_json,
                    edges_json = excluded.edges_json
                RETURNING id
                """,
                (name, description, json.dumps(nodes), json.dumps(edges)),
            )
            return cursor.fetchone()[0]

    def create_execution(self, workflow_name: str, role_name: str) -> int:
        """Create a new workflow execution."""
        with self.connection() as conn:
            workflow = conn.execute(
                "SELECT id FROM workflow_definitions WHERE name = ?", (workflow_name,)
            ).fetchone()
            if not workflow:
                raise ValueError(f"Workflow '{workflow_name}' not found")

            role = self.get_role(role_name)
            if not role or not role.id:
                raise ValueError(f"Role '{role_name}' not found")

            cursor = conn.execute(
                """
                INSERT INTO workflow_executions (workflow_id, role_id, status)
                VALUES (?, ?, ?)
                RETURNING id
                """,
                (workflow["id"], role.id, WorkflowStatus.PENDING.value),
            )
            return cursor.fetchone()[0]

    def update_execution_status(
        self,
        execution_id: int,
        status: WorkflowStatus,
        current_node: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update workflow execution status."""
        with self.connection() as conn:
            now = datetime.utcnow().isoformat()
            updates = ["status = ?", "updated_at = ?"]
            params: list[Any] = [status.value, now]

            if current_node is not None:
                updates.append("current_node = ?")
                params.append(current_node)

            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)

            if status == WorkflowStatus.RUNNING:
                updates.append("started_at = COALESCE(started_at, ?)")
                params.append(now)
            elif status in (
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED,
            ):
                updates.append("completed_at = ?")
                params.append(now)

            params.append(execution_id)
            conn.execute(
                f"UPDATE workflow_executions SET {', '.join(updates)} WHERE id = ?", params
            )

    def checkpoint_execution(self, execution_id: int, checkpoint_data: dict) -> None:
        """Save checkpoint data for workflow resumption."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE workflow_executions SET checkpoint_data = ? WHERE id = ?",
                (json.dumps(checkpoint_data), execution_id),
            )

    def get_checkpoint(self, execution_id: int) -> dict | None:
        """Get checkpoint data for resumption."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT checkpoint_data FROM workflow_executions WHERE id = ?", (execution_id,)
            ).fetchone()
            if row and row["checkpoint_data"]:
                return json.loads(row["checkpoint_data"])
            return None

    def update_node_execution(
        self,
        execution_id: int,
        node_name: str,
        status: NodeStatus,
        input_data: dict | None = None,
        output_data: dict | None = None,
        error_message: str | None = None,
    ) -> int:
        """Update or create node execution record."""
        with self.connection() as conn:
            now = datetime.utcnow().isoformat()

            cursor = conn.execute(
                """
                INSERT INTO node_executions (execution_id, node_name, status, input_data, output_data, error_message, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id, node_name) DO UPDATE SET
                    status = excluded.status,
                    input_data = COALESCE(excluded.input_data, node_executions.input_data),
                    output_data = COALESCE(excluded.output_data, node_executions.output_data),
                    error_message = excluded.error_message,
                    completed_at = CASE WHEN excluded.status IN ('completed', 'failed', 'skipped') THEN ? ELSE NULL END,
                    retry_count = CASE WHEN excluded.status = 'running' AND node_executions.status = 'failed'
                                       THEN node_executions.retry_count + 1
                                       ELSE node_executions.retry_count END
                RETURNING id
                """,
                (
                    execution_id,
                    node_name,
                    status.value,
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    error_message,
                    now,
                    now,
                ),
            )
            return cursor.fetchone()[0]

    # =========================================================================
    # TEST OPERATIONS
    # =========================================================================

    def create_test_run(self, test_run: TestRun) -> int:
        """Create a test run record."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO test_runs (role_id, worktree_id, execution_id, test_type, status,
                                      duration_seconds, log_path, output_json, commit_sha, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    test_run.role_id,
                    test_run.worktree_id,
                    test_run.execution_id,
                    test_run.test_type.value,
                    test_run.status.value,
                    test_run.duration_seconds,
                    test_run.log_path,
                    test_run.output_json,
                    test_run.commit_sha,
                    test_run.started_at or datetime.utcnow().isoformat(),
                ),
            )
            return cursor.fetchone()[0]

    def update_test_run(
        self,
        test_run_id: int,
        status: TestStatus,
        duration_seconds: int | None = None,
        output_json: str | None = None,
    ) -> None:
        """Update test run status."""
        with self.connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                UPDATE test_runs SET
                    status = ?,
                    duration_seconds = COALESCE(?, duration_seconds),
                    output_json = COALESCE(?, output_json),
                    completed_at = ?
                WHERE id = ?
                """,
                (status.value, duration_seconds, output_json, now, test_run_id),
            )

    def get_recent_test_runs(self, role_name: str, limit: int = 10) -> list[TestRun]:
        """Get recent test runs for a role."""
        with self.connection() as conn:
            role = self.get_role(role_name)
            if not role or not role.id:
                return []
            rows = conn.execute(
                """
                SELECT * FROM test_runs
                WHERE role_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (role.id, limit),
            ).fetchall()
            return [TestRun(**dict(row)) for row in rows]

    # =========================================================================
    # AUDIT LOGGING
    # =========================================================================

    def log_audit(
        self,
        entity_type: str,
        entity_id: int,
        action: str,
        old_value: dict | None = None,
        new_value: dict | None = None,
        actor: str = "harness",
    ) -> None:
        """Log an audit entry."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (entity_type, entity_id, action, old_value, new_value, actor)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_type,
                    entity_id,
                    action,
                    json.dumps(old_value) if old_value else None,
                    json.dumps(new_value) if new_value else None,
                    actor,
                ),
            )

    # =========================================================================
    # EXECUTION CONTEXT OPERATIONS (SEE/ACP)
    # =========================================================================

    def create_context(
        self,
        session_id: str,
        user_id: str | None = None,
        request_id: str | None = None,
        capabilities: list[str] | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Create a new execution context for a MCP client session."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO execution_contexts (session_id, user_id, request_id, capabilities, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    request_id = excluded.request_id,
                    capabilities = excluded.capabilities,
                    metadata = excluded.metadata
                RETURNING id
                """,
                (
                    session_id,
                    user_id,
                    request_id,
                    json.dumps(capabilities) if capabilities else None,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            context_id = cursor.fetchone()[0]
        # Log audit after connection closes to avoid nested connection lock
        self.log_audit(
            "execution_context",
            context_id,
            "create",
            new_value={"session_id": session_id, "user_id": user_id},
        )
        return context_id

    def get_context(self, session_id: str) -> ExecutionContext | None:
        """Get execution context by session ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM execution_contexts WHERE session_id = ?", (session_id,)
            ).fetchone()
            return ExecutionContext(**dict(row)) if row else None

    def get_context_by_id(self, context_id: int) -> ExecutionContext | None:
        """Get execution context by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM execution_contexts WHERE id = ?", (context_id,)
            ).fetchone()
            return ExecutionContext(**dict(row)) if row else None

    def grant_capability(
        self,
        context_id: int,
        capability: str,
        scope: str | None = None,
        granted_by: str = "system",
        expires_at: datetime | None = None,
    ) -> int:
        """Grant a capability to an execution context."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO context_capabilities (context_id, capability, scope, granted_by, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(context_id, capability, scope) DO UPDATE SET
                    granted_by = excluded.granted_by,
                    expires_at = excluded.expires_at,
                    revoked_at = NULL
                RETURNING id
                """,
                (
                    context_id,
                    capability,
                    scope,
                    granted_by,
                    expires_at.isoformat() if expires_at else None,
                ),
            )
            cap_id = cursor.fetchone()[0]
        # Log audit after connection closes to avoid nested connection lock
        self.log_audit(
            "context_capability",
            cap_id,
            "grant",
            new_value={"capability": capability, "scope": scope},
        )
        return cap_id

    def revoke_capability(self, context_id: int, capability: str, scope: str | None = None) -> bool:
        """Revoke a capability from an execution context."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE context_capabilities
                SET revoked_at = CURRENT_TIMESTAMP
                WHERE context_id = ? AND capability = ? AND (scope = ? OR (scope IS NULL AND ? IS NULL))
                AND revoked_at IS NULL
                """,
                (context_id, capability, scope, scope),
            )
            revoked = cursor.rowcount > 0
        # Log audit after connection closes to avoid nested connection lock
        if revoked:
            self.log_audit(
                "context_capability",
                context_id,
                "revoke",
                old_value={"capability": capability, "scope": scope},
            )
        return revoked

    def check_capability(self, context_id: int, capability: str, scope: str | None = None) -> bool:
        """Check if a context has a specific capability."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT id FROM context_capabilities
                WHERE context_id = ?
                AND capability = ?
                AND (scope IS NULL OR scope = ?)
                AND revoked_at IS NULL
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """,
                (context_id, capability, scope),
            ).fetchone()
            return row is not None

    def get_context_capabilities(self, context_id: int) -> list[ContextCapability]:
        """Get all active capabilities for a context."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM context_capabilities
                WHERE context_id = ?
                AND revoked_at IS NULL
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """,
                (context_id,),
            ).fetchall()
            return [ContextCapability(**dict(row)) for row in rows]

    def log_tool_invocation(
        self, context_id: int | None, tool_name: str, arguments: dict | None = None
    ) -> int:
        """Log the start of a tool invocation."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tool_invocations (context_id, tool_name, arguments, status)
                VALUES (?, ?, ?, 'running')
                RETURNING id
                """,
                (context_id, tool_name, json.dumps(arguments) if arguments else None),
            )
            return cursor.fetchone()[0]

    def complete_tool_invocation(
        self,
        invocation_id: int,
        result: dict | None = None,
        status: str = "completed",
        duration_ms: int | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        """Complete a tool invocation with results."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE tool_invocations
                SET result = ?, status = ?, duration_ms = ?, blocked_reason = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    json.dumps(result) if result else None,
                    status,
                    duration_ms,
                    blocked_reason,
                    invocation_id,
                ),
            )

    # =========================================================================
    # TEST REGRESSION OPERATIONS
    # =========================================================================

    def record_test_failure(
        self,
        role_name: str,
        test_name: str,
        test_type: TestType,
        test_run_id: int,
        error_message: str | None = None,
    ) -> int:
        """Record or update a test regression."""
        role = self.get_role(role_name)
        if not role or not role.id:
            raise ValueError(f"Role '{role_name}' not found")

        with self.connection() as conn:
            # Check if regression already exists
            existing = conn.execute(
                """
                SELECT id, failure_count, consecutive_failures, status
                FROM test_regressions
                WHERE role_id = ? AND test_name = ? AND test_type = ?
                """,
                (role.id, test_name, test_type.value),
            ).fetchone()

            now = datetime.utcnow().isoformat()

            if existing:
                # Update existing regression
                new_count = existing["failure_count"] + 1
                new_consecutive = existing["consecutive_failures"] + 1
                # If was resolved, mark as active again
                new_status = "active" if existing["status"] == "resolved" else existing["status"]

                conn.execute(
                    """
                    UPDATE test_regressions
                    SET failure_count = ?, consecutive_failures = ?,
                        last_failure_at = ?, last_error_message = ?, status = ?
                    WHERE id = ?
                    """,
                    (new_count, new_consecutive, now, error_message, new_status, existing["id"]),
                )
                regression_id = existing["id"]
            else:
                # Create new regression
                cursor = conn.execute(
                    """
                    INSERT INTO test_regressions
                    (role_id, test_name, test_type, first_failure_run_id, failure_count,
                     consecutive_failures, last_failure_at, last_error_message, status)
                    VALUES (?, ?, ?, ?, 1, 1, ?, ?, 'active')
                    RETURNING id
                    """,
                    (role.id, test_name, test_type.value, test_run_id, now, error_message),
                )
                regression_id = cursor.fetchone()[0]
        # Log audit after connection closes to avoid nested connection lock
        self.log_audit(
            "test_regression",
            regression_id,
            "failure",
            new_value={"test_name": test_name, "role": role_name},
        )
        return regression_id

    def record_test_success(
        self, role_name: str, test_name: str, test_type: TestType, test_run_id: int
    ) -> int | None:
        """Record a test success, potentially resolving a regression."""
        role = self.get_role(role_name)
        if not role or not role.id:
            return None

        existing_id = None
        with self.connection() as conn:
            existing = conn.execute(
                """
                SELECT id, consecutive_failures, status
                FROM test_regressions
                WHERE role_id = ? AND test_name = ? AND test_type = ?
                AND status IN ('active', 'flaky')
                """,
                (role.id, test_name, test_type.value),
            ).fetchone()

            if not existing:
                return None

            existing_id = existing["id"]

            if existing["consecutive_failures"] <= 1:
                # Single failure then pass = flaky
                conn.execute(
                    """
                    UPDATE test_regressions
                    SET status = 'flaky', consecutive_failures = 0
                    WHERE id = ?
                    """,
                    (existing_id,),
                )
            else:
                # Multiple consecutive failures then pass = resolved
                conn.execute(
                    """
                    UPDATE test_regressions
                    SET status = 'resolved', resolved_run_id = ?, consecutive_failures = 0
                    WHERE id = ?
                    """,
                    (test_run_id, existing_id),
                )
        # Log audit after connection closes to avoid nested connection lock
        self.log_audit(
            "test_regression",
            existing_id,
            "success",
            new_value={"test_name": test_name, "status": "resolved"},
        )
        return existing_id

    def get_active_regressions(self, role_name: str | None = None) -> list[ActiveRegressionView]:
        """Get all active test regressions."""
        with self.connection() as conn:
            if role_name:
                rows = conn.execute(
                    "SELECT * FROM v_active_regressions WHERE role_name = ?", (role_name,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM v_active_regressions").fetchall()
            return [ActiveRegressionView(**dict(row)) for row in rows]

    def get_regression(
        self, role_name: str, test_name: str, test_type: TestType
    ) -> TestRegression | None:
        """Get a specific test regression."""
        role = self.get_role(role_name)
        if not role or not role.id:
            return None

        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM test_regressions
                WHERE role_id = ? AND test_name = ? AND test_type = ?
                """,
                (role.id, test_name, test_type.value),
            ).fetchone()
            if row:
                data = dict(row)
                data["test_type"] = TestType(data["test_type"]) if data["test_type"] else None
                data["status"] = RegressionStatus(data["status"])
                return TestRegression(**data)
            return None

    def mark_regression_known_issue(self, regression_id: int, notes: str) -> None:
        """Mark a regression as a known issue with notes."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE test_regressions
                SET status = 'known_issue', notes = ?
                WHERE id = ?
                """,
                (notes, regression_id),
            )
            self.log_audit(
                "test_regression", regression_id, "mark_known_issue", new_value={"notes": notes}
            )

    # =========================================================================
    # MERGE TRAIN OPERATIONS
    # =========================================================================

    def add_to_merge_train(
        self, mr_id: int, position: int | None = None, target_branch: str = "main"
    ) -> int:
        """Add a merge request to the merge train."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO merge_train_entries (mr_id, position, target_branch, status, queued_at)
                VALUES (?, ?, ?, 'queued', CURRENT_TIMESTAMP)
                RETURNING id
                """,
                (mr_id, position, target_branch),
            )
            entry_id = cursor.fetchone()[0]
            self.log_audit(
                "merge_train",
                entry_id,
                "queue",
                new_value={"mr_id": mr_id, "target_branch": target_branch},
            )
            return entry_id

    def update_merge_train_status(
        self,
        entry_id: int,
        status: MergeTrainStatus,
        pipeline_id: int | None = None,
        pipeline_status: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """Update merge train entry status."""
        with self.connection() as conn:
            now = datetime.utcnow().isoformat()
            updates = ["status = ?"]
            params: list[Any] = [status.value]

            if pipeline_id is not None:
                updates.append("pipeline_id = ?")
                params.append(pipeline_id)

            if pipeline_status is not None:
                updates.append("pipeline_status = ?")
                params.append(pipeline_status)

            if failure_reason is not None:
                updates.append("failure_reason = ?")
                params.append(failure_reason)

            if status == MergeTrainStatus.MERGING:
                updates.append("started_at = COALESCE(started_at, ?)")
                params.append(now)
            elif status in (
                MergeTrainStatus.MERGED,
                MergeTrainStatus.FAILED,
                MergeTrainStatus.CANCELLED,
            ):
                updates.append("completed_at = ?")
                params.append(now)

            params.append(entry_id)
            conn.execute(
                f"UPDATE merge_train_entries SET {', '.join(updates)} WHERE id = ?", params
            )

    def get_merge_train_entry(self, mr_id: int) -> MergeTrainEntry | None:
        """Get merge train entry for an MR."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM merge_train_entries
                WHERE mr_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (mr_id,),
            ).fetchone()
            if row:
                data = dict(row)
                data["status"] = MergeTrainStatus(data["status"])
                return MergeTrainEntry(**data)
            return None

    def list_merge_train(
        self, target_branch: str = "main", status: MergeTrainStatus | None = None
    ) -> list[MergeTrainEntry]:
        """List merge train entries."""
        with self.connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM merge_train_entries
                    WHERE target_branch = ? AND status = ?
                    ORDER BY position, queued_at
                    """,
                    (target_branch, status.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM merge_train_entries
                    WHERE target_branch = ?
                    ORDER BY position, queued_at
                    """,
                    (target_branch,),
                ).fetchall()
            result = []
            for row in rows:
                data = dict(row)
                data["status"] = MergeTrainStatus(data["status"])
                result.append(MergeTrainEntry(**data))
            return result

    # =========================================================================
    # SELF-CHECK AND VALIDATION OPERATIONS
    # =========================================================================

    def validate_schema(self) -> dict[str, Any]:
        """
        Validate database schema integrity.

        Returns:
            Dict with validation results:
            - 'valid': True if schema is correct
            - 'missing_tables': List of expected but missing tables
            - 'extra_tables': List of unexpected tables
            - 'missing_indexes': List of expected but missing indexes
        """
        expected_tables = {
            "roles",
            "role_dependencies",
            "credentials",
            "worktrees",
            "iterations",
            "issues",
            "merge_requests",
            "workflow_definitions",
            "workflow_executions",
            "node_executions",
            "test_runs",
            "test_cases",
            "audit_log",
            "execution_contexts",
            "context_capabilities",
            "tool_invocations",
            "test_regressions",
            "merge_train_entries",
        }

        expected_indexes = {
            "idx_role_deps_role",
            "idx_role_deps_depends",
            "idx_credentials_role",
            "idx_worktrees_role",
            "idx_worktrees_status",
            "idx_issues_role",
            "idx_issues_iteration",
            "idx_mrs_role",
            "idx_workflow_exec_status",
            "idx_workflow_exec_role",
            "idx_node_exec_status",
            "idx_test_runs_role",
            "idx_test_runs_status",
            "idx_audit_entity",
            "idx_exec_ctx_session",
            "idx_exec_ctx_user",
            "idx_ctx_caps_context",
            "idx_ctx_caps_capability",
            "idx_tool_inv_context",
            "idx_tool_inv_tool",
            "idx_tool_inv_status",
            "idx_regressions_role",
            "idx_regressions_status",
            "idx_regressions_test",
            "idx_merge_train_mr",
            "idx_merge_train_status",
        }

        with self.connection() as conn:
            # Get actual tables
            tables = set(
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            )

            # Get actual indexes
            indexes = set(
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            )

            missing_tables = expected_tables - tables
            extra_tables = tables - expected_tables - {"sqlite_sequence"}
            missing_indexes = expected_indexes - indexes

            return {
                "valid": len(missing_tables) == 0,
                "missing_tables": list(missing_tables),
                "extra_tables": list(extra_tables),
                "missing_indexes": list(missing_indexes),
                "table_count": len(tables),
                "index_count": len(indexes),
            }

    def validate_data_integrity(self) -> dict[str, Any]:
        """
        Validate data integrity constraints.

        Returns:
            Dict with validation results:
            - 'valid': True if all data is consistent
            - 'issues': List of issue dictionaries with type and count
        """
        issues = []

        with self.connection() as conn:
            # Check for orphaned dependencies (references to non-existent roles)
            orphaned_deps = conn.execute("""
                SELECT rd.id, rd.role_id, rd.depends_on_id
                FROM role_dependencies rd
                LEFT JOIN roles r1 ON rd.role_id = r1.id
                LEFT JOIN roles r2 ON rd.depends_on_id = r2.id
                WHERE r1.id IS NULL OR r2.id IS NULL
            """).fetchall()

            # Check for invalid status values in worktrees
            invalid_worktree_status = conn.execute("""
                SELECT id, status FROM worktrees
                WHERE status NOT IN ('active', 'stale', 'dirty', 'merged', 'pruned')
            """).fetchall()

            # Check for invalid wave values
            invalid_waves = conn.execute("""
                SELECT id, name, wave FROM roles WHERE wave < 0 OR wave > 4
            """).fetchall()

            # Check for orphaned workflow executions
            orphaned_executions = conn.execute("""
                SELECT we.id FROM workflow_executions we
                LEFT JOIN roles r ON we.role_id = r.id
                WHERE r.id IS NULL
            """).fetchall()

            # Check for orphaned node executions
            orphaned_nodes = conn.execute("""
                SELECT ne.id FROM node_executions ne
                LEFT JOIN workflow_executions we ON ne.execution_id = we.id
                WHERE we.id IS NULL
            """).fetchall()

            # Check for orphaned test runs
            orphaned_tests = conn.execute("""
                SELECT tr.id FROM test_runs tr
                LEFT JOIN roles r ON tr.role_id = r.id
                WHERE r.id IS NULL
            """).fetchall()

            if orphaned_deps:
                issues.append({"type": "orphaned_dependencies", "count": len(orphaned_deps)})
            if invalid_worktree_status:
                issues.append(
                    {"type": "invalid_worktree_status", "count": len(invalid_worktree_status)}
                )
            if invalid_waves:
                issues.append({"type": "invalid_waves", "count": len(invalid_waves)})
            if orphaned_executions:
                issues.append({"type": "orphaned_executions", "count": len(orphaned_executions)})
            if orphaned_nodes:
                issues.append({"type": "orphaned_node_executions", "count": len(orphaned_nodes)})
            if orphaned_tests:
                issues.append({"type": "orphaned_test_runs", "count": len(orphaned_tests)})

        return {"valid": len(issues) == 0, "issues": issues}

    def get_statistics(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with table row counts and other metrics
        """
        with self.connection() as conn:
            stats = {}

            tables = [
                "roles",
                "role_dependencies",
                "credentials",
                "worktrees",
                "iterations",
                "issues",
                "merge_requests",
                "workflow_executions",
                "node_executions",
                "test_runs",
                "test_cases",
                "test_regressions",
                "audit_log",
                "execution_contexts",
                "context_capabilities",
                "tool_invocations",
                "merge_train_entries",
            ]

            for table in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]
                    stats[table] = count
                except Exception:
                    stats[table] = -1  # Table might not exist

            # Get active regressions count
            try:
                stats["active_regressions"] = conn.execute(
                    "SELECT COUNT(*) as c FROM test_regressions WHERE status = 'active'"
                ).fetchone()["c"]
            except Exception:
                stats["active_regressions"] = 0

            # Get pending executions count
            try:
                stats["pending_executions"] = conn.execute(
                    "SELECT COUNT(*) as c FROM workflow_executions WHERE status = 'pending'"
                ).fetchone()["c"]
            except Exception:
                stats["pending_executions"] = 0

            # Get running executions count
            try:
                stats["running_executions"] = conn.execute(
                    "SELECT COUNT(*) as c FROM workflow_executions WHERE status = 'running'"
                ).fetchone()["c"]
            except Exception:
                stats["running_executions"] = 0

            # Database file size
            if str(self.db_path) != ":memory:":
                stats["db_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
            else:
                stats["db_size_bytes"] = 0

            return stats

    def reset_database(self, confirm: bool = False) -> bool:
        """
        Reset database to initial state.

        WARNING: This deletes ALL data.

        Args:
            confirm: Must be True to actually reset

        Returns:
            True if reset was performed
        """
        if not confirm:
            return False

        with self.connection() as conn:
            # Get all tables
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()

            # Disable foreign keys temporarily
            conn.execute("PRAGMA foreign_keys = OFF")

            # Delete all data from each table
            table_names = [row["name"] for row in tables]
            for table_name in table_names:
                conn.execute(f"DELETE FROM {table_name}")

            # Re-enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Vacuum to reclaim space
            conn.execute("VACUUM")

        self.log_audit("database", 0, "reset", new_value={"tables_cleared": len(table_names)})
        return True

    def clear_table(self, table_name: str, confirm: bool = False) -> int:
        """
        Clear a specific table.

        Only allows clearing of safe tables (audit_log, tool_invocations, test data).

        Args:
            table_name: Table to clear
            confirm: Must be True to actually clear

        Returns:
            Number of rows deleted
        """
        allowed_tables = {
            "audit_log",
            "tool_invocations",
            "test_runs",
            "test_cases",
            "node_executions",
            "workflow_executions",
            "test_regressions",
        }

        if table_name not in allowed_tables:
            raise ValueError(
                f"Cannot clear protected table: {table_name}. "
                f"Allowed tables: {', '.join(sorted(allowed_tables))}"
            )

        if not confirm:
            return 0

        with self.connection() as conn:
            count = conn.execute(f"SELECT COUNT(*) as c FROM {table_name}").fetchone()["c"]
            conn.execute(f"DELETE FROM {table_name}")

        self.log_audit("table", 0, "clear", new_value={"table": table_name, "rows_deleted": count})
        return count

    def vacuum(self) -> int:
        """
        Vacuum database to reclaim space.

        Returns:
            Space saved in bytes (0 for in-memory databases)
        """
        if str(self.db_path) == ":memory:":
            return 0

        size_before = self.db_path.stat().st_size if self.db_path.exists() else 0

        with self.connection() as conn:
            conn.execute("VACUUM")

        size_after = self.db_path.stat().st_size if self.db_path.exists() else 0
        return max(0, size_before - size_after)

    def backup(self, backup_path: str) -> bool:
        """
        Create a backup of the database.

        Args:
            backup_path: Path to save the backup

        Returns:
            True if backup was successful
        """
        import shutil

        if str(self.db_path) == ":memory:":
            return False

        shutil.copy2(self.db_path, backup_path)
        self.log_audit("database", 0, "backup", new_value={"backup_path": backup_path})
        return True

    # =========================================================================
    # TOKEN USAGE / COST TRACKING OPERATIONS
    # =========================================================================

    def record_token_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        context: dict | None = None,
    ) -> int:
        """
        Record token usage for cost tracking.

        Args:
            session_id: Session identifier
            model: Model used (e.g., "claude-opus-4-5")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Calculated cost in USD
            context: Optional context metadata

        Returns:
            ID of the created record
        """
        context_json = json.dumps(context) if context else None
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO token_usage (session_id, model, input_tokens, output_tokens, cost, context)
                VALUES (?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (session_id, model, input_tokens, output_tokens, cost, context_json),
            )
            return cursor.fetchone()[0]

    def get_session_costs(self, session_id: str) -> dict[str, Any]:
        """
        Get cost summary for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with total_cost, total_input_tokens, total_output_tokens,
            record_count, and by_model breakdown
        """
        with self.connection() as conn:
            # Get totals
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(cost), 0) as total_cost,
                    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COUNT(*) as record_count
                FROM token_usage
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            # Get breakdown by model
            model_rows = conn.execute(
                """
                SELECT model, SUM(cost) as model_cost,
                       SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens
                FROM token_usage
                WHERE session_id = ?
                GROUP BY model
                """,
                (session_id,),
            ).fetchall()

        by_model = {
            r["model"]: {
                "cost": r["model_cost"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
            }
            for r in model_rows
        }

        return {
            "session_id": session_id,
            "total_cost": row["total_cost"],
            "total_input_tokens": row["total_input_tokens"],
            "total_output_tokens": row["total_output_tokens"],
            "record_count": row["record_count"],
            "by_model": by_model,
        }

    def get_cost_summary(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Get cost summary over a date range.

        Args:
            start_date: Start date (ISO format, inclusive)
            end_date: End date (ISO format, inclusive)

        Returns:
            Dict with totals, by_model breakdown, and by_session breakdown
        """
        with self.connection() as conn:
            # Build query with optional date filters
            query = """
                SELECT
                    COALESCE(SUM(cost), 0) as total_cost,
                    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COUNT(*) as record_count
                FROM token_usage
                WHERE 1=1
            """
            params: list[Any] = []

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            row = conn.execute(query, params).fetchone()

            # Get breakdown by model
            model_query = """
                SELECT model, SUM(cost) as model_cost,
                       SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens,
                       COUNT(*) as count
                FROM token_usage
                WHERE 1=1
            """
            if start_date:
                model_query += " AND timestamp >= ?"
            if end_date:
                model_query += " AND timestamp <= ?"
            model_query += " GROUP BY model ORDER BY model_cost DESC"

            model_rows = conn.execute(model_query, params).fetchall()

            # Get breakdown by session (top 20)
            session_query = """
                SELECT session_id, SUM(cost) as session_cost,
                       SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens,
                       COUNT(*) as count
                FROM token_usage
                WHERE 1=1
            """
            if start_date:
                session_query += " AND timestamp >= ?"
            if end_date:
                session_query += " AND timestamp <= ?"
            session_query += " GROUP BY session_id ORDER BY session_cost DESC LIMIT 20"

            session_rows = conn.execute(session_query, params).fetchall()

        by_model = {
            r["model"]: {
                "cost": r["model_cost"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "count": r["count"],
            }
            for r in model_rows
        }

        by_session = {
            r["session_id"]: {
                "cost": r["session_cost"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "count": r["count"],
            }
            for r in session_rows
        }

        return {
            "total_cost": row["total_cost"],
            "total_input_tokens": row["total_input_tokens"],
            "total_output_tokens": row["total_output_tokens"],
            "record_count": row["record_count"],
            "by_model": by_model,
            "by_session": by_session,
            "start_date": start_date,
            "end_date": end_date,
        }
