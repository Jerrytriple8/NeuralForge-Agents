"""Built-in Code Review Agent."""

from __future__ import annotations

import ast
import asyncio
import re
from typing import Any

from neuralforge.agents.base import AgentCapability, BaseAgent
from neuralforge.core.context import PipelineContext


class ReviewAgent(BaseAgent):
    """Agent that performs static code review with configurable rules.

    Config:
        code: Source code string to review (or pulled from dependency).
        language: Language identifier (default "python").
        rules: List of rule IDs to apply (default: all).
        severity_threshold: Minimum severity to report ("info", "warning", "error").
    """

    @property
    def name(self) -> str:
        return "reviewer"

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [AgentCapability.CODE_REVIEW]

    @property
    def description(self) -> str:
        return "Performs static code analysis and review with configurable rules."

    async def execute(
        self,
        config: dict[str, Any],
        dependencies: dict[str, Any],
        context: PipelineContext,
    ) -> dict[str, Any]:
        # Extract code from config or upstream dependency
        code = config.get("code", "")
        if not code and dependencies:
            for dep_result in dependencies.values():
                if isinstance(dep_result, dict) and "code" in dep_result:
                    code = dep_result["code"]
                    break

        language = config.get("language", "python")
        rules = config.get("rules", "all")
        threshold = config.get("severity_threshold", "info")

        findings: list[dict[str, Any]] = []

        if language == "python" and code:
            findings.extend(self._review_python(code))

        # Generic checks for all languages
        findings.extend(self._review_generic(code))

        # Filter by severity
        severity_order = {"info": 0, "warning": 1, "error": 2}
        min_severity = severity_order.get(threshold, 0)
        findings = [
            f for f in findings if severity_order.get(f["severity"], 0) >= min_severity
        ]

        # Filter by specific rules if requested
        if rules != "all" and isinstance(rules, list):
            findings = [f for f in findings if f["rule_id"] in rules]

        score = max(0, 100 - len(findings) * 10)

        return {
            "language": language,
            "lines_reviewed": len(code.splitlines()) if code else 0,
            "findings": findings,
            "finding_count": len(findings),
            "score": score,
            "verdict": "pass" if score >= 70 else "fail",
        }

    # ------------------------------------------------------------------
    # Python-specific review
    # ------------------------------------------------------------------

    def _review_python(self, code: str) -> list[dict[str, Any]]:
        """Python-specific static analysis checks."""
        findings: list[dict[str, Any]] = []

        # Try AST parse
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            findings.append({
                "rule_id": "PY001",
                "severity": "error",
                "message": f"Syntax error: {e.msg}",
                "line": e.lineno or 0,
            })
            return findings

        # Check for bare except
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append({
                    "rule_id": "PY002",
                    "severity": "warning",
                    "message": "Bare 'except:' catches all exceptions including SystemExit and KeyboardInterrupt",
                    "line": node.lineno,
                })

        # Check for mutable default arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append({
                            "rule_id": "PY003",
                            "severity": "warning",
                            "message": f"Mutable default argument in function '{node.name}'",
                            "line": node.lineno,
                        })

        # Check for print statements (should use logging)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    findings.append({
                        "rule_id": "PY004",
                        "severity": "info",
                        "message": "Consider using logging instead of print()",
                        "line": node.lineno,
                    })

        # Check for hardcoded secrets patterns
        for i, line in enumerate(code.splitlines(), 1):
            if re.search(r'(password|secret|api_key|token)\s*=\s*["\'][^"\']{8,}', line, re.I):
                findings.append({
                    "rule_id": "PY005",
                    "severity": "error",
                    "message": "Possible hardcoded secret detected",
                    "line": i,
                })

        return findings

    # ------------------------------------------------------------------
    # Generic review (all languages)
    # ------------------------------------------------------------------

    @staticmethod
    def _review_generic(code: str) -> list[dict[str, Any]]:
        """Language-agnostic code quality checks."""
        findings: list[dict[str, Any]] = []

        if not code:
            return findings

        lines = code.splitlines()

        # Check for very long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                findings.append({
                    "rule_id": "GEN001",
                    "severity": "info",
                    "message": f"Line exceeds 120 characters ({len(line)} chars)",
                    "line": i,
                })

        # Check for TODO/FIXME/HACK markers
        for i, line in enumerate(lines, 1):
            if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', line):
                findings.append({
                    "rule_id": "GEN002",
                    "severity": "info",
                    "message": f"Unresolved marker found: {re.search(r'(TODO|FIXME|HACK|XXX)', line).group()}",
                    "line": i,
                })

        # Check for excessive nesting (more than 4 levels of indentation)
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent > 16 and stripped:  # ~4 levels at 4-space indent
                findings.append({
                    "rule_id": "GEN003",
                    "severity": "warning",
                    "message": "Deeply nested code — consider refactoring",
                    "line": i,
                })

        return findings
