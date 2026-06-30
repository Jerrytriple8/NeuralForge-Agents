"""Built-in pipeline templates for common patterns."""

from __future__ import annotations

from typing import Any


TEMPLATES: dict[str, dict[str, Any]] = {
    "research_and_code": {
        "name": "Research → Code Pipeline",
        "version": "1.0",
        "description": "Research a topic, generate code, then review it.",
        "nodes": [
            {
                "id": "research",
                "name": "Research Phase",
                "agent": "researcher",
                "config": {
                    "query": "Research {topic}",
                    "sources": ["google", "arxiv"],
                    "max_results": 5,
                    "strategy": "merge",
                },
            },
            {
                "id": "generate",
                "name": "Code Generation",
                "agent": "coder",
                "depends_on": ["research"],
                "config": {
                    "task": "Generate implementation based on research findings",
                    "language": "python",
                    "style": "functional",
                },
            },
            {
                "id": "review",
                "name": "Code Review",
                "agent": "reviewer",
                "depends_on": ["generate"],
                "config": {
                    "language": "python",
                    "severity_threshold": "warning",
                },
            },
        ],
    },
    "parallel_research": {
        "name": "Parallel Research Pipeline",
        "version": "1.0",
        "description": "Research multiple topics in parallel, then merge findings.",
        "nodes": [
            {
                "id": "research_api",
                "name": "API Research",
                "agent": "researcher",
                "config": {"query": "Best practices for {topic} APIs", "sources": ["google"]},
            },
            {
                "id": "research_security",
                "name": "Security Research",
                "agent": "researcher",
                "config": {"query": "Security considerations for {topic}", "sources": ["google"]},
            },
            {
                "id": "research_perf",
                "name": "Performance Research",
                "agent": "researcher",
                "config": {"query": "Performance optimization for {topic}", "sources": ["google"]},
            },
            {
                "id": "generate",
                "name": "Generate Optimized Code",
                "agent": "coder",
                "depends_on": ["research_api", "research_security", "research_perf"],
                "config": {
                    "task": "Generate code incorporating all research findings",
                    "language": "python",
                    "style": "oop",
                },
            },
        ],
    },
    "code_review_only": {
        "name": "Code Review Pipeline",
        "version": "1.0",
        "description": "Review existing code for issues.",
        "nodes": [
            {
                "id": "review",
                "name": "Review",
                "agent": "reviewer",
                "config": {
                    "language": "python",
                    "severity_threshold": "info",
                },
            },
        ],
    },
}


def get_template(name: str) -> dict[str, Any]:
    """Get a built-in pipeline template by name."""
    if name not in TEMPLATES:
        raise KeyError(f"Template '{name}' not found. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]


def list_templates() -> list[str]:
    """List available template names."""
    return list(TEMPLATES.keys())
