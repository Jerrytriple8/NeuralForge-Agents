"""Pipeline schema validator."""

from __future__ import annotations

from typing import Any

_SCHEMA_VERSION = "1.0"
_REQUIRED_NODE_KEYS = {"id", "agent"}
_VALID_RETRY_KEYS = {"max_retries", "backoff_base"}


class ValidationError:
    """Single validation error with context."""

    def __init__(self, path: str, message: str, severity: str = "error") -> None:
        self.path = path
        self.message = message
        self.severity = severity

    def __repr__(self) -> str:
        return f"[{self.severity}] {self.path}: {self.message}"


class PipelineValidator:
    """Validate pipeline YAML structure and constraints."""

    @staticmethod
    def validate(data: dict[str, Any]) -> list[ValidationError]:
        """Validate a pipeline definition dict. Returns list of errors."""
        errors: list[ValidationError] = []

        # Top-level checks
        if "name" not in data:
            errors.append(ValidationError("pipeline", "Missing 'name' field", "warning"))

        version = data.get("version", "")
        if version and version != _SCHEMA_VERSION:
            errors.append(
                ValidationError("pipeline.version", f"Expected schema v{_SCHEMA_VERSION}, got v{version}", "warning")
            )

        nodes = data.get("nodes", [])
        if not nodes:
            errors.append(ValidationError("pipeline.nodes", "Must have at least one node"))
            return errors

        if not isinstance(nodes, list):
            errors.append(ValidationError("pipeline.nodes", "Must be a list"))
            return errors

        node_ids: set[str] = set()

        for i, node in enumerate(nodes):
            path = f"pipeline.nodes[{i}]"

            if not isinstance(node, dict):
                errors.append(ValidationError(path, "Node must be a dict"))
                continue

            # Required fields
            for key in _REQUIRED_NODE_KEYS:
                if key not in node:
                    errors.append(ValidationError(f"{path}.{key}", f"Missing required field '{key}'"))

            node_id = node.get("id", f"<unnamed_{i}>")

            # Unique IDs
            if node_id in node_ids:
                errors.append(ValidationError(f"{path}.id", f"Duplicate node id: '{node_id}'"))
            node_ids.add(node_id)

            # Dependencies exist
            deps = node.get("depends_on", [])
            if not isinstance(deps, list):
                errors.append(ValidationError(f"{path}.depends_on", "Must be a list"))

            # Retry policy structure
            retry = node.get("retry_policy", {})
            if retry:
                if not isinstance(retry, dict):
                    errors.append(ValidationError(f"{path}.retry_policy", "Must be a dict"))
                else:
                    unknown = set(retry.keys()) - _VALID_RETRY_KEYS
                    if unknown:
                        errors.append(
                            ValidationError(
                                f"{path}.retry_policy",
                                f"Unknown keys: {unknown}",
                                "warning",
                            )
                        )

            # Timeout type
            timeout = node.get("timeout", 0)
            if not isinstance(timeout, (int, float)):
                errors.append(ValidationError(f"{path}.timeout", "Must be a number"))

        # Check for forward references in depends_on (warn only — resolved at runtime)
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            for dep in node.get("depends_on", []):
                if dep not in node_ids:
                    errors.append(
                        ValidationError(
                            f"pipeline.nodes[{i}].depends_on",
                            f"Dependency '{dep}' not found in declared nodes",
                        )
                    )

        return errors
