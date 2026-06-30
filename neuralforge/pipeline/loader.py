"""YAML pipeline definition loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from neuralforge.core.dag import DAG, DAGNode


class PipelineLoadError(Exception):
    """Raised when a pipeline YAML file cannot be loaded or parsed."""


class PipelineLoader:
    """Load and construct DAG objects from YAML definitions.

    YAML format:
        ```yaml
        name: my_pipeline
        version: "1.0"
        nodes:
          - id: research
            agent: researcher
            config:
              query: "Find info about {topic}"
              sources: [google, arxiv]
          - id: code
            agent: coder
            depends_on: [research]
            config:
              task: "Generate code based on research"
              language: python
        ```
    """

    @staticmethod
    def from_yaml(path: str | Path) -> tuple[DAG, dict[str, Any]]:
        """Load a pipeline from a YAML file. Returns (dag, metadata)."""
        path = Path(path)
        if not path.exists():
            raise PipelineLoadError(f"File not found: {path}")
        if not path.suffix in (".yaml", ".yml"):
            raise PipelineLoadError(f"Expected .yaml/.yml, got: {path.suffix}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return PipelineLoader.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> tuple[DAG, dict[str, Any]]:
        """Load a pipeline from a dict (e.g., parsed YAML)."""
        if not isinstance(data, dict):
            raise PipelineLoadError("Pipeline definition must be a dict")

        metadata = {
            "name": data.get("name", "unnamed"),
            "version": data.get("version", "1.0"),
            "description": data.get("description", ""),
        }

        nodes_data = data.get("nodes", [])
        if not nodes_data:
            raise PipelineLoadError("Pipeline must have at least one node")

        dag = DAG()
        for node_def in nodes_data:
            node = DAGNode(
                node_id=node_def["id"],
                name=node_def.get("name", node_def["id"]),
                agent=node_def["agent"],
                config=node_def.get("config", {}),
                depends_on=node_def.get("depends_on", []),
                retry_policy=node_def.get("retry_policy", {"max_retries": 3, "backoff_base": 2}),
                timeout=node_def.get("timeout", 0.0),
            )
            dag.add_node(node)

        dag.validate()
        return dag, metadata

    @staticmethod
    def to_yaml(dag: DAG, metadata: dict[str, Any] | None = None) -> str:
        """Serialize a DAG back to YAML string."""
        data: dict[str, Any] = {
            "name": (metadata or {}).get("name", "unnamed"),
            "version": (metadata or {}).get("version", "1.0"),
            "nodes": [],
        }
        for node in dag.nodes:
            data["nodes"].append({
                "id": node.node_id,
                "name": node.name,
                "agent": node.agent,
                "config": node.config,
                "depends_on": node.depends_on,
                "retry_policy": node.retry_policy,
                "timeout": node.timeout,
            })
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
