"""
Research Agent - Analyzes tasks and gathers relevant information.
"""

import asyncio
import hashlib
import re
from typing import Any, Dict, List
from .base import BaseAgent, AgentConfig, AgentResult


class ResearchAgent(BaseAgent):
    """
    Agent specialized in information gathering and analysis.
    
    Capabilities:
    - Text analysis and summarization
    - Pattern recognition in data
    - Keyword extraction
    - Similarity scoring
    - Structured data parsing
    """

    def __init__(self, config: AgentConfig = None):
        config = config or AgentConfig(
            name="researcher",
            description="Analyzes tasks and gathers relevant information",
        )
        super().__init__(config)
        self._knowledge_base: Dict[str, Any] = {}

    async def execute(self, payload: Dict[str, Any]) -> AgentResult:
        task = payload.get("task", "")
        data = payload.get("data", "")
        mode = payload.get("mode", "analyze")

        if mode == "analyze":
            result = self._analyze(task, data)
        elif mode == "extract":
            result = self._extract_patterns(data)
        elif mode == "summarize":
            result = self._summarize(data)
        elif mode == "search":
            result = self._search_knowledge(task)
        else:
            result = self._analyze(task, data)

        return AgentResult(
            agent_name=self.name,
            task_id="",
            output=result,
            confidence=result.get("confidence", 0.8),
            reasoning=result.get("reasoning", ""),
        )

    def _analyze(self, task: str, data: Any) -> Dict[str, Any]:
        text = str(data) if data else task
        words = text.lower().split()
        word_freq = {}
        for w in words:
            w = re.sub(r"[^a-z0-9]", "", w)
            if len(w) > 2:
                word_freq[w] = word_freq.get(w, 0) + 1

        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        complexity = self._estimate_complexity(text)
        sentiment = self._estimate_sentiment(text)

        return {
            "keywords": [k for k, _ in top_keywords],
            "keyword_freq": dict(top_keywords),
            "sentence_count": len(sentences),
            "word_count": len(words),
            "char_count": len(text),
            "complexity": complexity,
            "sentiment": sentiment,
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "confidence": 0.85,
            "reasoning": f"Analyzed {len(words)} words across {len(sentences)} sentences",
        }

    def _extract_patterns(self, data: Any) -> Dict[str, Any]:
        text = str(data)
        patterns = {
            "emails": re.findall(r"[\w.-]+@[\w.-]+\.\w+", text),
            "urls": re.findall(r"https?://[^\s]+", text),
            "numbers": re.findall(r"\b\d+\.?\d*\b", text),
            "dates": re.findall(r"\d{4}[-/]\d{2}[-/]\d{2}", text),
            "ip_addresses": re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text),
            "code_blocks": re.findall(r"```[\s\S]*?```", text),
        }
        total = sum(len(v) for v in patterns.values())
        return {
            "patterns": patterns,
            "total_matches": total,
            "confidence": 0.9,
            "reasoning": f"Found {total} pattern matches across {len(patterns)} categories",
        }

    def _summarize(self, data: Any) -> Dict[str, Any]:
        text = str(data)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= 3:
            summary = ". ".join(sentences) + "."
        else:
            scored = []
            words = text.lower().split()
            word_freq = {}
            for w in words:
                w = re.sub(r"[^a-z0-9]", "", w)
                if len(w) > 2:
                    word_freq[w] = word_freq.get(w, 0) + 1

            for sent in sentences:
                score = sum(word_freq.get(w.lower(), 0) for w in sent.split())
                scored.append((score, sent))

            scored.sort(reverse=True)
            top_n = max(2, len(sentences) // 3)
            summary = ". ".join(s for _, s in scored[:top_n]) + "."

        compression = len(summary) / max(len(text), 1)
        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round(compression, 3),
            "confidence": 0.75,
            "reasoning": f"Compressed {len(text)} chars to {len(summary)} chars ({compression:.1%})",
        }

    def _search_knowledge(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        results = []
        for key, value in self._knowledge_base.items():
            if query_lower in str(key).lower() or query_lower in str(value).lower():
                results.append({"key": key, "value": value})

        return {
            "query": query,
            "results": results,
            "result_count": len(results),
            "confidence": 0.95 if results else 0.3,
            "reasoning": f"Found {len(results)} matching entries in knowledge base",
        }

    def _estimate_complexity(self, text: str) -> str:
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        if avg_word_len > 8:
            return "high"
        elif avg_word_len > 5:
            return "medium"
        return "low"

    def _estimate_sentiment(self, text: str) -> str:
        positive = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "best"]
        negative = ["bad", "terrible", "awful", "horrible", "worst", "hate", "poor", "fail"]
        text_lower = text.lower()
        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        return "neutral"

    def add_knowledge(self, key: str, value: Any) -> None:
        self._knowledge_base[key] = value

    def load_knowledge(self, data: Dict[str, Any]) -> None:
        self._knowledge_base.update(data)
