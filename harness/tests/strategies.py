"""Hypothesis strategies for property-based testing."""

from hypothesis import strategies as st
from harness.db.models import (
    Role, RoleDependency, DependencyType, Credential,
    Worktree, WorktreeStatus, TestRun, TestType, TestStatus,
    WorkflowExecution, WorkflowStatus, NodeExecution, NodeStatus
)


# Base strategies
wave_strategy = st.integers(min_value=0, max_value=4)

role_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=50
).filter(lambda x: len(x) > 0 and x[0].isalpha() and not x.startswith("_"))


# Model strategies
@st.composite
def role_strategy(draw):
    """Generate valid Role instances."""
    name = draw(role_name_strategy)
    return Role(
        name=name,
        wave=draw(wave_strategy),
        wave_name=draw(st.one_of(st.none(), st.text(min_size=1, max_size=30))),
        description=draw(st.one_of(st.none(), st.text(max_size=200))),
        has_molecule_tests=draw(st.booleans())
    )


@st.composite
def dependency_type_strategy(draw):
    """Generate valid DependencyType."""
    return draw(st.sampled_from(list(DependencyType)))


@st.composite
def worktree_status_strategy(draw):
    """Generate valid WorktreeStatus."""
    return draw(st.sampled_from(list(WorktreeStatus)))


@st.composite
def test_type_strategy(draw):
    """Generate valid TestType."""
    return draw(st.sampled_from(list(TestType)))


@st.composite
def test_status_strategy(draw):
    """Generate valid TestStatus."""
    return draw(st.sampled_from(list(TestStatus)))


@st.composite
def credential_strategy(draw, role_id: int = 1):
    """Generate valid Credential instances."""
    return Credential(
        role_id=role_id,
        entry_name=draw(st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="_-"),
            min_size=1,
            max_size=50
        ).filter(lambda x: len(x) > 0 and x[0].isalpha())),
        purpose=draw(st.one_of(st.none(), st.text(max_size=100))),
        is_base58=draw(st.booleans()),
        attribute=draw(st.one_of(st.none(), st.text(min_size=1, max_size=30)))
    )


@st.composite
def worktree_strategy(draw, role_id: int = 1):
    """Generate valid Worktree instances."""
    role_name = draw(role_name_strategy)
    return Worktree(
        role_id=role_id,
        path=f"../.worktrees/sid-{role_name}",
        branch=f"sid/{role_name}",
        commits_ahead=draw(st.integers(min_value=0, max_value=100)),
        commits_behind=draw(st.integers(min_value=0, max_value=100)),
        uncommitted_changes=draw(st.integers(min_value=0, max_value=50)),
        status=draw(worktree_status_strategy())
    )


# Graph strategies for cycle detection testing
@st.composite
def dag_graph_strategy(draw, min_nodes: int = 2, max_nodes: int = 10):
    """
    Generate a valid DAG (no cycles).

    Creates nodes and edges where edges only go from lower to higher indices,
    which guarantees a DAG structure.
    """
    n_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = [f"role_{i}" for i in range(n_nodes)]
    edges = []

    # Only allow edges from lower to higher indices (guarantees DAG)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if draw(st.booleans()):
                edges.append((nodes[i], nodes[j]))

    return nodes, edges


@st.composite
def cyclic_graph_strategy(draw, min_nodes: int = 3, max_nodes: int = 8):
    """
    Generate a graph with at least one cycle.

    Creates a guaranteed cycle of at least 2 nodes, plus optional additional edges.
    """
    n_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = [f"role_{i}" for i in range(n_nodes)]

    # Create a cycle (at least length 2)
    cycle_length = draw(st.integers(min_value=2, max_value=min(n_nodes, 5)))
    edges = [(nodes[i], nodes[(i + 1) % cycle_length]) for i in range(cycle_length)]

    # Add some random non-cycle edges
    extra_edges = draw(st.integers(min_value=0, max_value=5))
    for _ in range(extra_edges):
        i = draw(st.integers(0, n_nodes - 1))
        j = draw(st.integers(0, n_nodes - 1))
        if i != j:
            edge = (nodes[i], nodes[j])
            if edge not in edges:
                edges.append(edge)

    return nodes, edges


@st.composite
def self_loop_graph_strategy(draw, min_nodes: int = 1, max_nodes: int = 5):
    """Generate a graph with at least one self-loop."""
    n_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = [f"role_{i}" for i in range(n_nodes)]
    edges = []

    # Add a self-loop
    loop_node = draw(st.integers(0, n_nodes - 1))
    edges.append((nodes[loop_node], nodes[loop_node]))

    # Add some random edges
    extra_edges = draw(st.integers(min_value=0, max_value=3))
    for _ in range(extra_edges):
        i = draw(st.integers(0, n_nodes - 1))
        j = draw(st.integers(0, n_nodes - 1))
        edge = (nodes[i], nodes[j])
        if edge not in edges:
            edges.append(edge)

    return nodes, edges


# Wave-based strategies for testing wave assignment
@st.composite
def roles_by_wave_strategy(draw):
    """Generate roles distributed across waves for dependency testing."""
    waves = {
        0: ["common", "windows_prerequisites"],
        1: ["iis_config", "ems_registry_urls"],
        2: ["sql_server_2022", "sql_management_studio", "database_clone"],
        3: ["ems_web_app", "ems_platform_services"],
        4: ["grafana_alloy", "monitoring"],
    }

    # Select subset of roles from each wave
    selected = []
    for wave, role_names in waves.items():
        count = draw(st.integers(min_value=0, max_value=len(role_names)))
        selected_names = draw(st.sampled_from(
            [list(combo) for combo in _combinations(role_names, count)]
        ) if count > 0 else st.just([]))
        for name in selected_names:
            selected.append(Role(name=name, wave=wave, has_molecule_tests=True))

    return selected


def _combinations(items, r):
    """Generate all combinations of r items."""
    if r == 0:
        return [()]
    if not items:
        return []
    first, rest = items[0], items[1:]
    with_first = [(first,) + combo for combo in _combinations(rest, r - 1)]
    without_first = _combinations(rest, r)
    return with_first + without_first


# Test data strategies
@st.composite
def test_run_strategy(draw, role_id: int = 1):
    """Generate valid TestRun instances."""
    return TestRun(
        role_id=role_id,
        test_type=draw(test_type_strategy()),
        status=draw(test_status_strategy()),
        duration_seconds=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=3600))),
        commit_sha=draw(st.one_of(st.none(), st.text(
            alphabet="0123456789abcdef",
            min_size=40,
            max_size=40
        )))
    )


# Execution context strategies
@st.composite
def session_id_strategy(draw):
    """Generate valid session IDs."""
    import uuid
    return str(uuid.uuid4())


@st.composite
def capability_strategy(draw):
    """Generate valid capability strings."""
    actions = ["read", "write", "execute", "admin"]
    resources = ["roles", "worktrees", "tests", "workflows", "credentials", "context"]
    action = draw(st.sampled_from(actions))
    resource = draw(st.sampled_from(resources))
    return f"{action}:{resource}"


@st.composite
def capabilities_list_strategy(draw, max_capabilities: int = 5):
    """Generate a list of unique capabilities."""
    count = draw(st.integers(min_value=0, max_value=max_capabilities))
    capabilities = set()
    for _ in range(count):
        capabilities.add(draw(capability_strategy()))
    return list(capabilities)
