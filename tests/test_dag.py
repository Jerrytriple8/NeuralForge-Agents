"""Tests for the DAG builder and validator."""

import pytest

from neuralforge.core.dag import DAG, DAGNode, CyclicDependencyError, NodeStatus


class TestDAGNode:
    def test_create_node(self):
        node = DAGNode(node_id="test", name="Test Node", agent="researcher")
        assert node.node_id == "test"
        assert node.name == "Test Node"
        assert node.agent == "researcher"
        assert node.depends_on == []
        assert node.status == NodeStatus.PENDING

    def test_node_with_dependencies(self):
        node = DAGNode(node_id="gen", name="Generate", agent="coder", depends_on=["research"])
        assert node.depends_on == ["research"]

    def test_node_with_config(self):
        node = DAGNode(
            node_id="test",
            name="Test",
            agent="coder",
            config={"task": "generate code", "language": "python"},
        )
        assert node.config["language"] == "python"


class TestDAG:
    def test_empty_dag(self):
        dag = DAG()
        assert len(dag.nodes) == 0

    def test_add_node(self):
        dag = DAG()
        dag.add_node(DAGNode(node_id="a", name="Research", agent="researcher"))
        assert len(dag.nodes) == 1
        assert dag.get_node("a").agent == "researcher"

    def test_duplicate_node_raises(self):
        dag = DAG()
        dag.add_node(DAGNode(node_id="a", name="Research", agent="researcher"))
        with pytest.raises(ValueError):
            dag.add_node(DAGNode(node_id="a", name="Research2", agent="coder"))

    def test_cycle_detection(self):
        dag = DAG()
        dag.add_node(DAGNode(node_id="a", name="Research", agent="researcher"))
        dag.add_node(DAGNode(node_id="b", name="Generate", agent="coder"))
        dag.add_node(DAGNode(node_id="c", name="Review", agent="reviewer", depends_on=["b"]))
        # Add dependency b -> a
        dag.add_node(DAGNode(node_id="b", name="Generate", agent="coder", depends_on=["a"]))
        dag._reverse["b"] = {"a"}
        dag._adjacency.setdefault("a", set()).add("b")
        # Attempt to create cycle: a depends on c (which depends on b -> a)
        with pytest.raises((CyclicDependencyError, ValueError)):
            dag.add_node(DAGNode(node_id="a", name="Research", agent="researcher", depends_on=["c"]))

    def test_validate_simple_dag(self):
        dag = DAG()
        dag.add_node(DAGNode(node_id="a", name="Research", agent="researcher"))
        dag.add_node(DAGNode(node_id="b", name="Generate", agent="coder", depends_on=["a"]))
        dag.validate()  # Should not raise

    def test_topological_sort(self):
        dag = DAG()
        dag.add_node(DAGNode(node_id="a", name="Research", agent="researcher"))
        dag.add_node(DAGNode(node_id="b", name="Generate", agent="coder", depends_on=["a"]))
        dag.add_node(DAGNode(node_id="c", name="Review", agent="reviewer", depends_on=["b"]))
        order = dag.topological_sort()
        ids = [n.node_id for n in order]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_root_nodes(self):
        dag = DAG()
        dag.add_node(DAGNode(node_id="a", name="Research A", agent="researcher"))
        dag.add_node(DAGNode(node_id="b", name="Research B", agent="researcher"))
        dag.add_node(DAGNode(node_id="c", name="Generate", agent="coder", depends_on=["a", "b"]))
        roots = dag.root_nodes
        root_ids = {n.node_id for n in roots}
        assert root_ids == {"a", "b"}

    def test_dependents(self):
        dag = DAG()
        dag.add_node(DAGNode(node_id="a", name="Research", agent="researcher"))
        dag.add_node(DAGNode(node_id="b", name="Generate", agent="coder", depends_on=["a"]))
        dag.add_node(DAGNode(node_id="c", name="Review", agent="reviewer", depends_on=["a"]))
        deps = dag.dependents("a")
        dep_ids = {n.node_id for n in deps}
        assert dep_ids == {"b", "c"}

    def test_parallel_levels(self):
        dag = DAG()
        dag.add_node(DAGNode(node_id="a", name="Research", agent="researcher"))
        dag.add_node(DAGNode(node_id="b", name="Generate", agent="coder", depends_on=["a"]))
        dag.add_node(DAGNode(node_id="c", name="Review", agent="reviewer", depends_on=["b"]))
        levels = dag.parallel_levels()
        assert len(levels) == 3
        assert levels[0][0].node_id == "a"
        assert levels[1][0].node_id == "b"
        assert levels[2][0].node_id == "c"
