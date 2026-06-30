"""
Tests for agents.
"""

import asyncio
import pytest
from agents.base import BaseAgent, AgentConfig, AgentResult
from agents.researcher import ResearchAgent
from agents.coder import CoderAgent
from agents.critic import CriticAgent


class TestBaseAgent:
    def test_agent_config(self):
        config = AgentConfig(name="test", description="Test agent")
        assert config.name == "test"
        assert config.description == "Test agent"

    def test_agent_result(self):
        result = AgentResult(
            agent_name="test", task_id="123", output="data", confidence=0.9
        )
        assert result.agent_name == "test"
        assert result.confidence == 0.9


class TestResearchAgent:
    def setup_method(self):
        self.agent = ResearchAgent()

    def test_analyze(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({"task": "analyze", "data": "This is a test sentence with keywords."})
        )
        assert result.output["word_count"] > 0
        assert "keywords" in result.output

    def test_extract_patterns(self):
        data = "Email: test@example.com, URL: https://example.com, IP: 192.168.1.1"
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({"task": "extract", "data": data})
        )
        assert len(result.output["patterns"]["emails"]) > 0
        assert len(result.output["patterns"]["urls"]) > 0

    def test_summarize(self):
        data = "First sentence here. Second sentence there. Third sentence everywhere. Fourth sentence nowhere."
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({"task": "summarize", "data": data})
        )
        assert "summary" in result.output
        assert result.output["compression_ratio"] < 1.0

    def test_search_knowledge(self):
        self.agent.add_knowledge("python", "A programming language")
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({"task": "search", "data": "python"})
        )
        assert result.output["result_count"] > 0


class TestCoderAgent:
    def setup_method(self):
        self.agent = CoderAgent()

    def test_generate_python(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({
                "task": "generate",
                "language": "python",
                "spec": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": [
                        {"name": "a", "type": "int"},
                        {"name": "b", "type": "int"},
                    ],
                    "body": ["return a + b"],
                },
            })
        )
        assert "def add" in result.output["code"]
        assert result.output["syntax_valid"]

    def test_generate_javascript(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({
                "task": "generate",
                "language": "javascript",
                "spec": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": [{"name": "a"}, {"name": "b"}],
                    "body": ["return a + b;"],
                },
            })
        )
        assert "function add" in result.output["code"]

    def test_debug(self):
        code = "def f():\n    return 1/0"
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({
                "task": "debug",
                "code": code,
                "spec": {"error": "ZeroDivisionError: division by zero"},
            })
        )
        assert len(result.output["issues"]) > 0

    def test_review(self):
        code = "def f():\n    return 1"
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({"task": "review", "code": code})
        )
        assert "score" in result.output
        assert 0 <= result.output["score"] <= 100

    def test_explain(self):
        code = "def hello():\n    print('Hello')"
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({"task": "explain", "code": code})
        )
        assert len(result.output["explanation"]) > 0


class TestCriticAgent:
    def setup_method(self):
        self.agent = CriticAgent()

    def test_score_output(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({
                "task": "score",
                "output": "This is a well-written response with specific details.",
                "criteria": {"min_length": 10},
            })
        )
        assert 0 <= result.output["overall_score"] <= 1

    def test_compare_outputs(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({
                "task": "compare",
                "outputs": [
                    "Short answer",
                    "A much longer and more detailed answer with specific information",
                ],
                "criteria": {"min_length": 20},
            })
        )
        assert "rankings" in result.output
        assert len(result.output["rankings"]) == 2

    def test_detect_bias(self):
        text = "Obviously this is always the best solution. Everyone knows this."
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({"task": "detect_bias", "output": text})
        )
        assert result.output["bias_score"] > 0

    def test_check_completeness(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.agent.execute({
                "task": "completeness",
                "output": "This covers topic A and topic B thoroughly.",
                "criteria": {
                    "required_elements": ["topic A", "topic B", "topic C"],
                    "min_length": 20,
                },
            })
        )
        assert result.output["coverage"] < 1.0  # Missing topic C


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
