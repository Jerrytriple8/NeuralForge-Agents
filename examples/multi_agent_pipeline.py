"""
Example: Multi-agent research pipeline.

Demonstrates how to use NeuralForge agents in a pipeline
for comprehensive text analysis.
"""

import asyncio
import json
from core import NeuralEngine
from agents import ResearchAgent, CoderAgent, CriticAgent


async def main():
    # Initialize engine
    engine = NeuralEngine()

    # Register agents
    researcher = ResearchAgent()
    coder = CoderAgent()
    critic = CriticAgent()

    engine.register_agent("researcher", researcher)
    engine.register_agent("coder", coder)
    engine.register_agent("critic", critic)

    # Sample text to analyze
    sample_text = """
    NeuralForge is an advanced AI agent orchestration framework.
    It provides capabilities for multi-agent coordination, task routing,
    and intelligent pipeline management. The framework supports various
    agent types including researchers, coders, and critics.
    """

    # Step 1: Research - Analyze the text
    print("=== Step 1: Research Analysis ===")
    research_task = await engine.submit_task(
        "researcher",
        {"task": "analyze", "data": sample_text}
    )
    await asyncio.sleep(1)
    research_result = engine.get_task(research_task.task_id)
    print(f"Keywords: {research_result.result.output.get('keywords', [])[:5]}")
    print(f"Word count: {research_result.result.output.get('word_count', 0)}")
    print(f"Complexity: {research_result.result.output.get('complexity', 'unknown')}")
    print()

    # Step 2: Research - Extract patterns
    print("=== Step 2: Pattern Extraction ===")
    pattern_task = await engine.submit_task(
        "researcher",
        {"task": "extract", "data": sample_text}
    )
    await asyncio.sleep(1)
    pattern_result = engine.get_task(pattern_task.task_id)
    patterns = pattern_result.result.output.get("patterns", {})
    print(f"Total matches: {pattern_result.result.output.get('total_matches', 0)}")
    print()

    # Step 3: Coder - Generate code based on analysis
    print("=== Step 3: Code Generation ===")
    code_task = await engine.submit_task(
        "coder",
        {
            "task": "generate",
            "language": "python",
            "spec": {
                "name": "analyze_text",
                "description": "Analyze text and return statistics",
                "parameters": [
                    {"name": "text", "type": "str"},
                ],
                "body": [
                    "words = text.split()",
                    "return {",
                    "    'word_count': len(words),",
                    "    'char_count': len(text),",
                    "    'avg_word_len': sum(len(w) for w in words) / len(words) if words else 0",
                    "}",
                ],
            },
        }
    )
    await asyncio.sleep(1)
    code_result = engine.get_task(code_task.task_id)
    print(f"Generated code:\n{code_result.result.output.get('code', '')}")
    print(f"Syntax valid: {code_result.result.output.get('syntax_valid', False)}")
    print()

    # Step 4: Critic - Review the generated code
    print("=== Step 4: Code Review ===")
    review_task = await engine.submit_task(
        "critic",
        {
            "task": "score",
            "output": code_result.result.output.get("code", ""),
            "criteria": {
                "keywords": ["def", "return", "split"],
                "min_length": 50,
            },
        }
    )
    await asyncio.sleep(1)
    review_result = engine.get_task(review_task.task_id)
    print(f"Score: {review_result.result.output.get('overall_score', 0):.1%}")
    print(f"Dimensions: {json.dumps(review_result.result.output.get('dimension_scores', {}), indent=2)}")
    print()

    # Show engine stats
    print("=== Engine Stats ===")
    print(json.dumps(engine.stats, indent=2))

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
