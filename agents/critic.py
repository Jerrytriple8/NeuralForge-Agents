"""
Critic Agent - Reviews and scores outputs from other agents.
"""

import re
import math
from typing import Any, Dict, List, Optional
from .base import BaseAgent, AgentConfig, AgentResult


class CriticAgent(BaseAgent):
    """
    Agent specialized in quality assessment and critique.
    
    Capabilities:
    - Output quality scoring
    - Consistency checking
    - Bias detection
    - Completeness analysis
    - Comparative ranking
    """

    def __init__(self, config: AgentConfig = None):
        config = config or AgentConfig(
            name="critic",
            description="Reviews and scores outputs from other agents",
        )
        super().__init__(config)
        self._review_history: List[Dict] = []

    async def execute(self, payload: Dict[str, Any]) -> AgentResult:
        task = payload.get("task", "score")
        output = payload.get("output", "")
        criteria = payload.get("criteria", {})
        reference = payload.get("reference", "")

        if task == "score":
            result = self._score_output(output, criteria)
        elif task == "compare":
            outputs = payload.get("outputs", [])
            result = self._compare_outputs(outputs, criteria)
        elif task == "check_consistency":
            result = self._check_consistency(output, reference)
        elif task == "detect_bias":
            result = self._detect_bias(output)
        elif task == "completeness":
            result = self._check_completeness(output, criteria)
        else:
            result = self._score_output(output, criteria)

        return AgentResult(
            agent_name=self.name,
            task_id="",
            output=result,
            confidence=result.get("confidence", 0.8),
            reasoning=result.get("reasoning", ""),
        )

    def _score_output(self, output: Any, criteria: Dict[str, Any]) -> Dict[str, Any]:
        text = str(output)
        scores = {}

        # Fluency score
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        scores["fluency"] = min(1.0, avg_len / 20) if avg_len > 0 else 0

        # Coherence score
        words = text.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        scores["coherence"] = max(0, 1 - unique_ratio * 0.5)

        # Relevance score
        if "keywords" in criteria:
            keyword_hits = sum(
                1 for kw in criteria["keywords"] if kw.lower() in text.lower()
            )
            scores["relevance"] = keyword_hits / max(len(criteria["keywords"]), 1)
        else:
            scores["relevance"] = 0.7

        # Completeness score
        min_length = criteria.get("min_length", 50)
        scores["completeness"] = min(1.0, len(text) / max(min_length, 1))

        # Specificity score
        specific_terms = len(re.findall(r"\b[A-Z][a-z]+\b", text))
        scores["specificity"] = min(1.0, specific_terms / max(len(sentences), 1))

        # Weighted average
        weights = criteria.get("weights", {
            "fluency": 0.2,
            "coherence": 0.2,
            "relevance": 0.3,
            "completeness": 0.2,
            "specificity": 0.1,
        })
        total_score = sum(scores[k] * weights.get(k, 0.2) for k in scores)

        review = {
            "overall_score": round(total_score, 3),
            "dimension_scores": {k: round(v, 3) for k, v in scores.items()},
            "sentence_count": len(sentences),
            "word_count": len(words),
            "confidence": 0.85,
            "reasoning": f"Scored {total_score:.1%} across {len(scores)} dimensions",
        }
        self._review_history.append(review)
        return review

    def _compare_outputs(self, outputs: List[Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        if len(outputs) < 2:
            return {"error": "Need at least 2 outputs to compare", "confidence": 0.0, "reasoning": ""}

        scored = []
        for i, output in enumerate(outputs):
            score_result = self._score_output(output, criteria)
            scored.append({
                "index": i,
                "score": score_result["overall_score"],
                "details": score_result["dimension_scores"],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "rankings": scored,
            "best_index": scored[0]["index"],
            "best_score": scored[0]["score"],
            "score_spread": scored[0]["score"] - scored[-1]["score"],
            "confidence": 0.8,
            "reasoning": f"Compared {len(outputs)} outputs, best is index {scored[0]['index']}",
        }

    def _check_consistency(self, output: Any, reference: Any) -> Dict[str, Any]:
        output_text = str(output).lower()
        ref_text = str(reference).lower()

        out_words = set(output_text.split())
        ref_words = set(ref_text.split())

        overlap = out_words & ref_words
        jaccard = len(overlap) / max(len(out_words | ref_words), 1)

        # Check for contradictions
        contradictions = []
        negation_patterns = [
            (r"(\w+) is not (\w+)", r"\1 is \2"),
            (r"(\w+) cannot (\w+)", r"\1 can \2"),
            (r"(\w+) will not (\w+)", r"\1 will \2"),
        ]
        for neg_pattern, pos_pattern in negation_patterns:
            neg_matches = set(re.findall(neg_pattern, output_text))
            pos_matches = set(re.findall(pos_pattern, ref_text))
            overlap = neg_matches & pos_matches
            if overlap:
                contradictions.extend([f"{m[0]} contradicts reference" for m in overlap])

        return {
            "consistency_score": round(jaccard, 3),
            "word_overlap": len(overlap),
            "contradictions": contradictions,
            "contradiction_count": len(contradictions),
            "confidence": 0.75,
            "reasoning": f"Jaccard similarity: {jaccard:.1%}, {len(contradictions)} contradictions",
        }

    def _detect_bias(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        bias_indicators = {
            "always": 0.3, "never": 0.3, "everyone": 0.2, "nobody": 0.2,
            "obviously": 0.4, "clearly": 0.3, "undoubtedly": 0.4,
            "best": 0.2, "worst": 0.2, "perfect": 0.3, "impossible": 0.3,
            "definitely": 0.3, "absolutely": 0.3, "certainly": 0.3,
        }

        found_biases = []
        total_bias_score = 0
        for indicator, weight in bias_indicators.items():
            count = text_lower.count(indicator)
            if count > 0:
                found_biases.append({"term": indicator, "count": count, "weight": weight})
                total_bias_score += count * weight

        sentences = re.split(r"[.!?]+", text)
        first_person = sum(1 for s in sentences if re.search(r"\b(i|my|me|we|our)\b", s.lower()))
        first_person_ratio = first_person / max(len(sentences), 1)

        return {
            "bias_score": round(min(1.0, total_bias_score / 10), 3),
            "bias_indicators": found_biases,
            "first_person_ratio": round(first_person_ratio, 3),
            "sentence_count": len(sentences),
            "confidence": 0.7,
            "reasoning": f"Found {len(found_biases)} bias indicators, {first_person:.0%} first-person",
        }

    def _check_completeness(self, output: Any, criteria: Dict[str, Any]) -> Dict[str, Any]:
        text = str(output)
        required = criteria.get("required_elements", [])
        present = []
        missing = []

        for element in required:
            if element.lower() in text.lower():
                present.append(element)
            else:
                missing.append(element)

        coverage = len(present) / max(len(required), 1)
        length_score = min(1.0, len(text) / max(criteria.get("min_length", 100), 1))

        return {
            "completeness_score": round((coverage * 0.7 + length_score * 0.3), 3),
            "coverage": round(coverage, 3),
            "present_elements": present,
            "missing_elements": missing,
            "length_score": round(length_score, 3),
            "char_count": len(text),
            "confidence": 0.85,
            "reasoning": f"{len(present)}/{len(required)} required elements present",
        }
