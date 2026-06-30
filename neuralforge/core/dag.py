"""DAG builder and validator for pipeline definition."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class NodeStatus(enum.Enum):
    """Execution status of a DAG node."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class DAGNode:
    """Single node in the execution DAG.

    Attributes:
        node_id: Unique identifier for this node.
        name: Human-readable display name.
        agent: Agent type key used to look up the executor.
        config: Arbitrary configuration passed to the agent at runtime.
        depends_on: List of node_ids that must complete before this runs.
        retry_policy: Max retries and backoff multiplier.
        timeout: Maximum execution time in seconds (0 = unlimited).
        condition: Optional predicate evaluated against context; if False the node is skipped.
    """

    node_id: str
    name: str
    agent: str
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    retry_policy: dict[str, int] = field(
        default_factory=lambda: {"max_retries": 3, "backoff_base": 2}
    )
    timeout: float = 0.0
    condition: Optional[Callable[..., bool]] = None
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    attempt: int = 0


class CyclicDependencyError(Exception):
    """Raised when the DAG contains a cycle."""


class DAG:
    """Directed Acyclic Graph for pipeline task ordering.

    Supports:
        - Topological sorting with parallelism detection
        - Cycle detection
        - Conditional node execution
        - Dependency graph validation
    """

    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}
        self._adjacency: dict[str, set[str]] = {}  # node -> dependents
        self._reverse: dict[str, set[str]] = {}  # node -> dependencies

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_node(self, node: DAGNode) -> None:
        """Register a node and wire its declared dependencies."""
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")

        self._nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, set())
        self._reverse.setdefault(node.node_id, set())

        for dep in node.depends_on:
            if dep not in self._nodes:
                raise ValueError(
                    f"Node '{node.node_id}' depends on unknown node '{dep}'"
                )
            self._adjacency.setdefault(dep, set()).add(node.node_id)
            self._reverse[node.node_id].add(dep)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges connected to it."""
        self._nodes.pop(node_id, None)
        # Remove from adjacency lists
        self._adjacency.pop(node_id, None)
        for deps in self._adjacency.values():
            deps.discard(node_id)
        # Remove from reverse lists
        self._reverse.pop(node_id, None)
        for deps in self._reverse.values():
            deps.discard(node_id)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate the DAG has no cycles and all dependencies resolve."""
        if not self._nodes:
            return
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(node_id: str) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            for dep in self._adjacency.get(node_id, set()):
                if dep not in visited:
                    _dfs(dep)
                elif dep in rec_stack:
                    raise CyclicDependencyError(
                        f"Cycle detected: {node_id} -> {dep}"
                    )
            rec_stack.discard(node_id)

        for nid in self._nodes:
            if nid not in visited:
                _dfs(nid)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> DAGNode:
        return self._nodes[node_id]

    @property
    def nodes(self) -> list[DAGNode]:
        return list(self._nodes.values())

    @property
    def root_nodes(self) -> list[DAGNode]:
        """Nodes with no dependencies."""
        return [n for n in self._nodes.values() if not self._reverse.get(n.node_id)]

    def ready_nodes(self, completed: set[str]) -> list[DAGNode]:
        """Return nodes whose dependencies are all in *completed* and are still pending."""
        ready = []
        for node in self._nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            deps = self._reverse.get(node.node_id, set())
            if deps <= completed:
                ready.append(node)
        return ready

    def dependents(self, node_id: str) -> list[DAGNode]:
        """Nodes that directly depend on *node_id*."""
        return [self._nodes[d] for d in self._adjacency.get(node_id, set())]

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[DAGNode]:
        """Return a valid topological ordering of all nodes."""
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid, deps in self._reverse.items():
            in_degree[nid] = len(deps)

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[DAGNode] = []

        while queue:
            nid = queue.pop(0)
            order.append(self._nodes[nid])
            for dependent in self._adjacency.get(nid, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self._nodes):
            raise CyclicDependencyError("Topological sort failed — cycle exists")

        return order

    def parallel_levels(self) -> list[list[DAGNode]]:
        """Group nodes into levels that can execute in parallel.

        Level 0 = root nodes, level N = nodes whose deps are all in levels < N.
        """
        levels: list[list[DAGNode]] = []
        completed: set[str] = set()
        remaining = set(self._nodes.keys())

        while remaining:
            level = [
                self._nodes[nid]
                for nid in remaining
                if self._reverse.get(nid, set()) <= completed
            ]
            if not level:
                raise CyclicDependencyError(
                    "Cannot determine parallel levels — cycle detected"
                )
            levels.append(level)
            for node in level:
                completed.add(node.node_id)
                remaining.discard(node.node_id)

        return levels

    def to_mermaid(self) -> str:
        """Generate a Mermaid flowchart representation."""
        lines = ["graph TD"]
        for node in self._nodes.values():
            safe_id = node.node_id.replace("-", "_")
            lines.append(f'    {safe_id}["{node.name}"]')
        for node in self._nodes.values():
            safe_id = node.node_id.replace("-", "_")
            for dep in node.depends_on:
                safe_dep = dep.replace("-", "_")
                lines.append(f"    {safe_dep} --> {safe_id}")
        return "\n".join(lines)
