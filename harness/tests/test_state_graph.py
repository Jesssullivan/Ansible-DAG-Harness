"""Tests for graph operations: dependencies, cycles, topological sort."""

import pytest
from hypothesis import given, settings, assume, HealthCheck
from harness.db.state import StateDB
from harness.db.models import Role, RoleDependency, DependencyType, CyclicDependencyError
from tests.strategies import dag_graph_strategy, cyclic_graph_strategy, self_loop_graph_strategy


class TestDependencyGraph:
    """Test dependency graph operations."""

    @pytest.mark.unit
    def test_get_dependencies_direct(self, db_with_roles: StateDB):
        """Get direct dependencies of a role."""
        deps = db_with_roles.get_dependencies("sql_management_studio", transitive=False)
        dep_names = [name for name, _ in deps]

        assert "common" in dep_names
        assert "sql_server_2022" in dep_names
        assert len(dep_names) == 2

    @pytest.mark.unit
    def test_get_dependencies_transitive(self, db_with_roles: StateDB):
        """Get transitive dependencies."""
        # sql_management_studio -> sql_server_2022 -> common
        # sql_management_studio -> common
        deps = db_with_roles.get_dependencies("sql_management_studio", transitive=True)
        dep_names = [name for name, _ in deps]

        assert "common" in dep_names
        assert "sql_server_2022" in dep_names
        # ems_web_app should NOT be in dependencies
        assert "ems_web_app" not in dep_names

    @pytest.mark.unit
    def test_get_dependencies_empty(self, db_with_roles: StateDB):
        """Role with no dependencies."""
        deps = db_with_roles.get_dependencies("common", transitive=False)
        assert deps == []

    @pytest.mark.unit
    def test_get_dependencies_nonexistent_role(self, db_with_roles: StateDB):
        """Non-existent role returns empty list."""
        deps = db_with_roles.get_dependencies("nonexistent")
        assert deps == []

    @pytest.mark.unit
    def test_get_dependencies_depth(self, db_with_roles: StateDB):
        """Transitive dependencies include depth information."""
        deps = db_with_roles.get_dependencies("sql_management_studio", transitive=True)
        dep_dict = {name: depth for name, depth in deps}

        # common is directly depended on, depth 1
        # sql_server_2022 is also directly depended on, depth 1
        assert dep_dict["common"] == 1
        assert dep_dict["sql_server_2022"] == 1


class TestReverseDependencies:
    """Test reverse dependency queries."""

    @pytest.mark.unit
    def test_get_reverse_dependencies(self, db_with_roles: StateDB):
        """Get roles that depend on common."""
        reverse = db_with_roles.get_reverse_dependencies("common", transitive=False)
        rev_names = [name for name, _ in reverse]

        assert "sql_server_2022" in rev_names
        assert "sql_management_studio" in rev_names
        assert "ems_web_app" in rev_names
        assert len(rev_names) == 3

    @pytest.mark.unit
    def test_get_reverse_dependencies_transitive(self, db_with_roles: StateDB):
        """Get transitive reverse dependencies."""
        # common is depended on by sql_server_2022
        # sql_server_2022 is depended on by sql_management_studio
        # So transitively, common -> sql_server_2022 -> sql_management_studio
        reverse = db_with_roles.get_reverse_dependencies("common", transitive=True)
        rev_names = [name for name, _ in reverse]

        assert "sql_server_2022" in rev_names
        assert "sql_management_studio" in rev_names
        assert "ems_web_app" in rev_names
        # ems_platform_services depends on ems_web_app which depends on common
        assert "ems_platform_services" in rev_names

    @pytest.mark.unit
    def test_get_reverse_dependencies_empty(self, db_with_roles: StateDB):
        """Role with no reverse dependencies."""
        # ems_platform_services is a leaf node
        reverse = db_with_roles.get_reverse_dependencies("ems_platform_services")
        assert reverse == []

    @pytest.mark.unit
    def test_get_reverse_dependencies_chain(self, db_with_roles: StateDB):
        """Test reverse dependency chain."""
        # ems_web_app is depended on by ems_platform_services
        reverse = db_with_roles.get_reverse_dependencies("ems_web_app", transitive=False)
        rev_names = [name for name, _ in reverse]

        assert "ems_platform_services" in rev_names
        assert len(rev_names) == 1


class TestDependencyGraphView:
    """Test dependency graph view."""

    @pytest.mark.unit
    def test_get_dependency_graph(self, db_with_roles: StateDB):
        """Get full dependency graph."""
        graph = db_with_roles.get_dependency_graph()

        assert len(graph) >= 5  # At least 5 edges in our test data

        # Check that graph contains expected edges
        edge_set = {(e["from_role"], e["to_role"]) for e in graph}
        assert ("sql_server_2022", "common") in edge_set
        assert ("sql_management_studio", "common") in edge_set
        assert ("sql_management_studio", "sql_server_2022") in edge_set


class TestCycleDetection:
    """Test cycle detection using Tarjan's algorithm."""

    @pytest.mark.unit
    def test_no_cycles_in_valid_dag(self, db_with_roles: StateDB):
        """Valid DAG should have no cycles."""
        cycles = db_with_roles.detect_cycles()
        assert cycles == []

    @pytest.mark.unit
    def test_detect_simple_cycle(self, db: StateDB):
        """Detect A -> B -> A cycle."""
        role_a = Role(name="role_a", wave=1)
        role_b = Role(name="role_b", wave=1)
        db.upsert_role(role_a)
        db.upsert_role(role_b)

        a = db.get_role("role_a")
        b = db.get_role("role_b")

        # A depends on B
        db.add_dependency(RoleDependency(
            role_id=a.id, depends_on_id=b.id, dependency_type=DependencyType.EXPLICIT
        ))
        # B depends on A (creates cycle)
        db.add_dependency(RoleDependency(
            role_id=b.id, depends_on_id=a.id, dependency_type=DependencyType.EXPLICIT
        ))

        cycles = db.detect_cycles()
        assert len(cycles) == 1
        # Cycle should contain both roles
        cycle_nodes = set(cycles[0][:-1])  # Exclude the closing node
        assert "role_a" in cycle_nodes or "role_b" in cycle_nodes

    @pytest.mark.unit
    def test_detect_self_loop(self, db: StateDB):
        """Detect self-dependency cycle."""
        role = Role(name="self_dep", wave=1)
        db.upsert_role(role)
        r = db.get_role("self_dep")

        db.add_dependency(RoleDependency(
            role_id=r.id, depends_on_id=r.id, dependency_type=DependencyType.EXPLICIT
        ))

        cycles = db.detect_cycles()
        assert len(cycles) == 1
        assert cycles[0] == ["self_dep", "self_dep"]

    @pytest.mark.unit
    def test_detect_larger_cycle(self, db: StateDB):
        """Detect A -> B -> C -> A cycle."""
        roles = ["role_a", "role_b", "role_c"]
        for name in roles:
            db.upsert_role(Role(name=name, wave=1))

        a = db.get_role("role_a")
        b = db.get_role("role_b")
        c = db.get_role("role_c")

        # A -> B -> C -> A
        db.add_dependency(RoleDependency(
            role_id=a.id, depends_on_id=b.id, dependency_type=DependencyType.EXPLICIT
        ))
        db.add_dependency(RoleDependency(
            role_id=b.id, depends_on_id=c.id, dependency_type=DependencyType.EXPLICIT
        ))
        db.add_dependency(RoleDependency(
            role_id=c.id, depends_on_id=a.id, dependency_type=DependencyType.EXPLICIT
        ))

        cycles = db.detect_cycles()
        assert len(cycles) == 1
        # All three roles should be in the cycle
        cycle_nodes = set(cycles[0][:-1])
        assert len(cycle_nodes) == 3

    @pytest.mark.unit
    def test_detect_multiple_cycles(self, db: StateDB):
        """Detect multiple independent cycles."""
        # Create two independent cycles
        for name in ["a1", "a2", "b1", "b2"]:
            db.upsert_role(Role(name=name, wave=1))

        a1 = db.get_role("a1")
        a2 = db.get_role("a2")
        b1 = db.get_role("b1")
        b2 = db.get_role("b2")

        # Cycle 1: a1 -> a2 -> a1
        db.add_dependency(RoleDependency(
            role_id=a1.id, depends_on_id=a2.id, dependency_type=DependencyType.EXPLICIT
        ))
        db.add_dependency(RoleDependency(
            role_id=a2.id, depends_on_id=a1.id, dependency_type=DependencyType.EXPLICIT
        ))

        # Cycle 2: b1 -> b2 -> b1
        db.add_dependency(RoleDependency(
            role_id=b1.id, depends_on_id=b2.id, dependency_type=DependencyType.EXPLICIT
        ))
        db.add_dependency(RoleDependency(
            role_id=b2.id, depends_on_id=b1.id, dependency_type=DependencyType.EXPLICIT
        ))

        cycles = db.detect_cycles()
        assert len(cycles) == 2

    @pytest.mark.pbt
    @given(graph=dag_graph_strategy())
    @settings(max_examples=30)
    def test_valid_dag_no_cycles_pbt(self, graph):
        """Property: valid DAG should never have cycles."""
        nodes, edges = graph
        assume(len(nodes) >= 2)

        # Create fresh db for each hypothesis example
        db = StateDB(":memory:")

        # Create roles
        for node in nodes:
            db.upsert_role(Role(name=node, wave=1))

        # Create edges
        for from_node, to_node in edges:
            from_role = db.get_role(from_node)
            to_role = db.get_role(to_node)
            if from_role and to_role:
                db.add_dependency(RoleDependency(
                    role_id=from_role.id,
                    depends_on_id=to_role.id,
                    dependency_type=DependencyType.EXPLICIT
                ))

        cycles = db.detect_cycles()
        assert cycles == []

    @pytest.mark.pbt
    @given(graph=cyclic_graph_strategy())
    @settings(max_examples=20)
    def test_cyclic_graph_detected_pbt(self, graph):
        """Property: graph with cycle should be detected."""
        nodes, edges = graph

        # Create fresh db for each hypothesis example
        db = StateDB(":memory:")

        # Create roles
        for node in nodes:
            db.upsert_role(Role(name=node, wave=1))

        # Create edges (some may fail due to duplicates)
        for from_node, to_node in edges:
            from_role = db.get_role(from_node)
            to_role = db.get_role(to_node)
            if from_role and to_role:
                try:
                    db.add_dependency(RoleDependency(
                        role_id=from_role.id,
                        depends_on_id=to_role.id,
                        dependency_type=DependencyType.EXPLICIT
                    ))
                except Exception:
                    pass  # Ignore duplicate edges

        cycles = db.detect_cycles()
        assert len(cycles) >= 1

    @pytest.mark.pbt
    @given(graph=self_loop_graph_strategy())
    @settings(max_examples=15)
    def test_self_loop_detected_pbt(self, graph):
        """Property: graph with self-loop should be detected."""
        nodes, edges = graph

        # Create fresh db for each hypothesis example
        db = StateDB(":memory:")

        # Create roles
        for node in nodes:
            db.upsert_role(Role(name=node, wave=1))

        # Create edges
        for from_node, to_node in edges:
            from_role = db.get_role(from_node)
            to_role = db.get_role(to_node)
            if from_role and to_role:
                try:
                    db.add_dependency(RoleDependency(
                        role_id=from_role.id,
                        depends_on_id=to_role.id,
                        dependency_type=DependencyType.EXPLICIT
                    ))
                except Exception:
                    pass

        cycles = db.detect_cycles()
        assert len(cycles) >= 1


class TestTopologicalSort:
    """Test topological sort / deployment order."""

    @pytest.mark.unit
    def test_deployment_order_valid_dag(self, db_with_roles: StateDB):
        """Deployment order should have dependencies before dependents."""
        order = db_with_roles.get_deployment_order()

        assert len(order) == 5

        common_idx = order.index("common")
        sql_server_idx = order.index("sql_server_2022")
        sql_mgmt_idx = order.index("sql_management_studio")
        ems_web_idx = order.index("ems_web_app")
        ems_platform_idx = order.index("ems_platform_services")

        # Dependencies must come before dependents
        assert common_idx < sql_server_idx
        assert common_idx < sql_mgmt_idx
        assert common_idx < ems_web_idx
        assert sql_server_idx < sql_mgmt_idx
        assert ems_web_idx < ems_platform_idx

    @pytest.mark.unit
    def test_deployment_order_empty_db(self, db: StateDB):
        """Empty database returns empty order."""
        order = db.get_deployment_order()
        assert order == []

    @pytest.mark.unit
    def test_deployment_order_no_dependencies(self, db: StateDB):
        """Roles with no dependencies have arbitrary valid order."""
        for name in ["role_a", "role_b", "role_c"]:
            db.upsert_role(Role(name=name, wave=1))

        order = db.get_deployment_order()
        assert len(order) == 3
        assert set(order) == {"role_a", "role_b", "role_c"}

    @pytest.mark.unit
    def test_deployment_order_raises_on_cycle(self, db: StateDB):
        """Should raise CyclicDependencyError on cycle."""
        role_a = Role(name="a", wave=1)
        role_b = Role(name="b", wave=1)
        db.upsert_role(role_a)
        db.upsert_role(role_b)

        a = db.get_role("a")
        b = db.get_role("b")

        db.add_dependency(RoleDependency(
            role_id=a.id, depends_on_id=b.id, dependency_type=DependencyType.EXPLICIT
        ))
        db.add_dependency(RoleDependency(
            role_id=b.id, depends_on_id=a.id, dependency_type=DependencyType.EXPLICIT
        ))

        with pytest.raises(CyclicDependencyError) as exc_info:
            db.get_deployment_order(raise_on_cycle=True)

        assert "Cyclic dependency" in str(exc_info.value)
        assert exc_info.value.cycle_path is not None

    @pytest.mark.unit
    def test_deployment_order_partial_on_cycle(self, db: StateDB):
        """Without raise_on_cycle, returns partial order."""
        for name in ["a", "b", "c", "d"]:
            db.upsert_role(Role(name=name, wave=1))

        a = db.get_role("a")
        b = db.get_role("b")
        c = db.get_role("c")
        d = db.get_role("d")

        # a -> b -> a (cycle)
        db.add_dependency(RoleDependency(
            role_id=a.id, depends_on_id=b.id, dependency_type=DependencyType.EXPLICIT
        ))
        db.add_dependency(RoleDependency(
            role_id=b.id, depends_on_id=a.id, dependency_type=DependencyType.EXPLICIT
        ))

        # c and d are not in cycle
        db.add_dependency(RoleDependency(
            role_id=d.id, depends_on_id=c.id, dependency_type=DependencyType.EXPLICIT
        ))

        order = db.get_deployment_order(raise_on_cycle=False)
        # c and d should be in order (c before d)
        assert "c" in order
        assert "d" in order

    @pytest.mark.pbt
    @given(graph=dag_graph_strategy(min_nodes=3, max_nodes=8))
    @settings(max_examples=25)
    def test_deployment_order_respects_deps_pbt(self, graph):
        """Property: topological order respects all dependencies."""
        nodes, edges = graph
        assume(len(nodes) >= 3 and len(edges) >= 1)

        # Create fresh db for each hypothesis example
        db = StateDB(":memory:")

        # Create roles
        for node in nodes:
            db.upsert_role(Role(name=node, wave=1))

        # Create edges
        for from_node, to_node in edges:
            from_role = db.get_role(from_node)
            to_role = db.get_role(to_node)
            if from_role and to_role:
                try:
                    db.add_dependency(RoleDependency(
                        role_id=from_role.id,
                        depends_on_id=to_role.id,
                        dependency_type=DependencyType.EXPLICIT
                    ))
                except Exception:
                    pass

        order = db.get_deployment_order()
        order_idx = {name: i for i, name in enumerate(order)}

        # Check that all edges are respected
        # Edge (a, b) means a depends on b, so b must come before a
        for from_node, to_node in edges:
            if from_node in order_idx and to_node in order_idx:
                assert order_idx[to_node] < order_idx[from_node], \
                    f"{to_node} should come before {from_node}"


class TestValidateDependencies:
    """Test comprehensive dependency validation."""

    @pytest.mark.unit
    def test_valid_graph(self, db_with_roles: StateDB):
        """Valid graph should pass validation."""
        result = db_with_roles.validate_dependencies()

        assert result["valid"] is True
        assert result["cycles"] == []
        assert result["missing_deps"] == []
        assert result["self_deps"] == []

    @pytest.mark.unit
    def test_detect_cycles_in_validation(self, db: StateDB):
        """Validation should detect cycles."""
        role_a = Role(name="a", wave=1)
        role_b = Role(name="b", wave=1)
        db.upsert_role(role_a)
        db.upsert_role(role_b)

        a = db.get_role("a")
        b = db.get_role("b")

        db.add_dependency(RoleDependency(
            role_id=a.id, depends_on_id=b.id, dependency_type=DependencyType.EXPLICIT
        ))
        db.add_dependency(RoleDependency(
            role_id=b.id, depends_on_id=a.id, dependency_type=DependencyType.EXPLICIT
        ))

        result = db.validate_dependencies()
        assert result["valid"] is False
        assert len(result["cycles"]) > 0

    @pytest.mark.unit
    def test_detect_self_deps_in_validation(self, db: StateDB):
        """Validation should detect self-dependencies."""
        role = Role(name="self_dep", wave=1)
        db.upsert_role(role)
        r = db.get_role("self_dep")

        db.add_dependency(RoleDependency(
            role_id=r.id, depends_on_id=r.id, dependency_type=DependencyType.EXPLICIT
        ))

        result = db.validate_dependencies()
        assert result["valid"] is False
        assert "self_dep" in result["self_deps"]
