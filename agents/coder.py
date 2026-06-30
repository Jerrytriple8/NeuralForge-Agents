"""
Coder Agent - Generates, refactors, and debugs code.
"""

import re
import ast
import textwrap
from typing import Any, Dict, List, Optional
from .base import BaseAgent, AgentConfig, AgentResult


class CoderAgent(BaseAgent):
    """
    Agent specialized in code generation and manipulation.
    
    Capabilities:
    - Code generation from natural language
    - Code refactoring
    - Bug detection
    - Code explanation
    - Template expansion
    """

    TEMPLATES = {
        "function": '''def {name}({params}):
    """{docstring}"""
    {body}
''',
        "class": '''class {name}{bases}:
    """{docstring}"""
    
    def __init__(self{init_params}):
        {init_body}
    
    {methods}
''',
        "async_function": '''async def {name}({params}):
    """{docstring}"""
    {body}
''',
        "decorator": '''def {name}(func):
    def wrapper(*args, **kwargs):
        {pre_logic}
        result = func(*args, **kwargs)
        {post_logic}
        return result
    return wrapper
''',
    }

    def __init__(self, config: AgentConfig = None):
        config = config or AgentConfig(
            name="coder",
            description="Generates, refactors, and debugs code",
        )
        super().__init__(config)

    async def execute(self, payload: Dict[str, Any]) -> AgentResult:
        task = payload.get("task", "generate")
        code = payload.get("code", "")
        language = payload.get("language", "python")
        spec = payload.get("spec", {})

        if task == "generate":
            result = self._generate_code(spec, language)
        elif task == "refactor":
            result = self._refactor(code, spec)
        elif task == "debug":
            result = self._debug(code, spec.get("error", ""))
        elif task == "explain":
            result = self._explain(code)
        elif task == "review":
            result = self._review(code)
        else:
            result = self._generate_code(spec, language)

        return AgentResult(
            agent_name=self.name,
            task_id="",
            output=result,
            confidence=result.get("confidence", 0.8),
            reasoning=result.get("reasoning", ""),
        )

    def _generate_code(self, spec: Dict[str, Any], language: str) -> Dict[str, Any]:
        name = spec.get("name", "generated_func")
        desc = spec.get("description", "Generated function")
        params = spec.get("parameters", [])
        return_type = spec.get("return_type", "Any")

        if language == "python":
            code = self._gen_python(name, desc, params, return_type, spec)
        elif language == "javascript":
            code = self._gen_javascript(name, desc, params, spec)
        elif language == "go":
            code = self._gen_go(name, desc, params, spec)
        else:
            code = self._gen_python(name, desc, params, return_type, spec)

        is_valid = self._validate_syntax(code, language)

        return {
            "code": code,
            "language": language,
            "name": name,
            "syntax_valid": is_valid,
            "confidence": 0.9 if is_valid else 0.5,
            "reasoning": f"Generated {language} code for '{name}' with {len(params)} parameters",
        }

    def _gen_python(self, name: str, desc: str, params: List, return_type: str, spec: Dict) -> str:
        param_str = ", ".join(
            f"{p.get('name', 'arg')}: {p.get('type', 'Any')}" +
            (f" = {p['default']}" if 'default' in p else "")
            for p in params
        ) if params else ""

        body_lines = spec.get("body", ["pass"])
        if isinstance(body_lines, str):
            body_lines = [body_lines]
        body = "\n    ".join(body_lines)

        if spec.get("async", False):
            template = self.TEMPLATES["async_function"]
        else:
            template = self.TEMPLATES["function"]

        return template.format(
            name=name,
            params=param_str,
            docstring=desc,
            body=body,
        )

    def _gen_javascript(self, name: str, desc: str, params: List, spec: Dict) -> str:
        param_str = ", ".join(p.get("name", "arg") for p in params) if params else ""
        body = spec.get("body", ["return null;"])
        if isinstance(body, str):
            body = [body]
        body_str = "\n  ".join(body)
        is_async = "async " if spec.get("async", False) else ""

        return f"""{is_async}function {name}({param_str}) {{
  // {desc}
  {body_str}
}}
"""

    def _gen_go(self, name: str, desc: str, params: List, spec: Dict) -> str:
        go_name = name[0].upper() + name[1:]
        param_str = ", ".join(
            f"{p.get('name', 'arg')} {p.get('type', 'interface{}')}"
            for p in params
        ) if params else ""
        return_type = spec.get("return_type", "")
        ret_str = f" {return_type}" if return_type else ""
        body = spec.get("body", ["return nil"])
        body_str = "\n\t".join(body)

        return f"""// {go_name} - {desc}
func {go_name}({param_str}){ret_str} {{
\t{body_str}
}}
"""

    def _refactor(self, code: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        refactored = code
        changes = []

        # Remove trailing whitespace
        refactored = re.sub(r"[ \t]+$", "", refactored, flags=re.MULTILINE)
        if refactored != code:
            changes.append("Removed trailing whitespace")

        # Add type hints if missing
        func_pattern = r"def (\w+)\(([^)]*)\):"
        matches = re.finditer(func_pattern, refactored)
        for match in matches:
            func_name = match.group(1)
            params = match.group(2)
            if "->" not in match.group(0):
                changes.append(f"Consider adding return type hint to '{func_name}'")

        # Check for long functions
        lines = code.split("\n")
        if len(lines) > 50:
            changes.append(f"Function is {len(lines)} lines - consider splitting")

        # Check for magic numbers
        numbers = re.findall(r"(?<![a-zA-Z_])\d{2,}(?![a-zA-Z_])", code)
        if numbers:
            changes.append(f"Found {len(numbers)} potential magic numbers - consider constants")

        return {
            "original": code,
            "refactored": refactored,
            "changes": changes,
            "confidence": 0.7,
            "reasoning": f"Applied {len(changes)} refactoring suggestions",
        }

    def _debug(self, code: str, error: str) -> Dict[str, Any]:
        issues = []
        suggestions = []

        # Common patterns
        if "IndentationError" in error:
            issues.append("Indentation mismatch")
            suggestions.append("Check for mixed tabs and spaces")
        elif "SyntaxError" in error:
            line_match = re.search(r"line (\d+)", error)
            if line_match:
                line_num = int(line_match.group(1))
                lines = code.split("\n")
                if line_num <= len(lines):
                    issues.append(f"Syntax error near line {line_num}: '{lines[line_num-1].strip()}'")
        elif "TypeError" in error:
            issues.append("Type mismatch in operation")
            suggestions.append("Check variable types before operations")
        elif "KeyError" in error:
            key_match = re.search(r"KeyError: ['\"](.+?)['\"]", error)
            if key_match:
                issues.append(f"Missing key: '{key_match.group(1)}'")
                suggestions.append("Use .get() with default value")
        elif "IndexError" in error:
            issues.append("Index out of range")
            suggestions.append("Add bounds checking before access")
        elif "AttributeError" in error:
            issues.append("Attribute not found on object")
            suggestions.append("Check object type and available attributes")

        # Static analysis
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for child in ast.walk(node):
                        if isinstance(child, ast.Raise):
                            issues.append(f"Function '{node.name}' may raise exceptions")
        except SyntaxError:
            pass

        if not issues:
            issues.append("No obvious issues detected")
            suggestions.append("Run with debugger for detailed traceback")

        return {
            "error": error,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.75,
            "reasoning": f"Found {len(issues)} potential issues",
        }

    def _explain(self, code: str) -> Dict[str, Any]:
        lines = code.strip().split("\n")
        explanation = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("def "):
                match = re.match(r"def (\w+)\(([^)]*)\)", stripped)
                if match:
                    explanation.append(f"Defines function '{match.group(1)}' with parameters: {match.group(2) or 'none'}")
            elif stripped.startswith("class "):
                match = re.match(r"class (\w+)", stripped)
                if match:
                    explanation.append(f"Defines class '{match.group(1)}'")
            elif stripped.startswith("import ") or stripped.startswith("from "):
                explanation.append(f"Imports: {stripped}")
            elif stripped.startswith("return "):
                explanation.append(f"Returns: {stripped[7:]}")
            elif "=" in stripped and not stripped.startswith("if") and not stripped.startswith("for"):
                var = stripped.split("=")[0].strip()
                explanation.append(f"Assigns value to '{var}'")

        return {
            "explanation": explanation,
            "line_count": len(lines),
            "confidence": 0.8,
            "reasoning": f"Explained {len(explanation)} code elements",
        }

    def _review(self, code: str) -> Dict[str, Any]:
        issues = []
        score = 100

        # Check for bare excepts
        if re.search(r"except\s*:", code):
            issues.append(("warning", "Bare except clause - catch specific exceptions"))
            score -= 10

        # Check for mutable default args
        if re.search(r"def \w+\([^)]*=\s*(\[\]|\{\})", code):
            issues.append(("error", "Mutable default argument - use None and initialize in body"))
            score -= 20

        # Check for print statements (should use logging)
        if re.search(r"(?<!\w)print\(", code):
            issues.append(("info", "Print statement found - consider using logging"))

        # Check for TODO/FIXME
        todos = re.findall(r"#\s*(TODO|FIXME|HACK|XXX).*", code)
        if todos:
            issues.append(("info", f"Found {len(todos)} TODO/FIXME comments"))

        # Check complexity
        branches = len(re.findall(r"\b(if|elif|else|for|while|try|except|with)\b", code))
        if branches > 10:
            issues.append(("warning", f"High complexity: {branches} branches"))
            score -= 15

        # Check docstrings
        funcs = re.findall(r"def (\w+)\(", code)
        docstrings = re.findall(r'"""[\s\S]*?"""', code)
        if len(funcs) > len(docstrings):
            issues.append(("info", f"{len(funcs) - len(docstrings)} functions missing docstrings"))

        return {
            "score": max(0, score),
            "issues": issues,
            "issue_count": len(issues),
            "confidence": 0.85,
            "reasoning": f"Review score: {max(0, score)}/100 with {len(issues)} issues",
        }

    def _validate_syntax(self, code: str, language: str) -> bool:
        if language == "python":
            try:
                ast.parse(code)
                return True
            except SyntaxError:
                return False
        return True
