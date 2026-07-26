from __future__ import annotations

from typing import Any

from proofmode.gemma_client import GemmaClient


TEACHBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pre_question", "post_question", "rubric", "concept_invariant"],
    "properties": {
        "pre_question": {"type": "string"},
        "post_question": {"type": "string"},
        "rubric": {"type": "array", "minItems": 3, "maxItems": 5, "items": {"type": "string"}},
        "concept_invariant": {"type": "string"},
    },
}


SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "accurate_points", "missing_points", "misconceptions", "feedback", "uncertainty"],
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "accurate_points": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "missing_points": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "misconceptions": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "feedback": {"type": "string"},
        "uncertainty": {"type": "string"},
    },
}


def generate_transfer_pair(client: GemmaClient, topic: dict[str, Any]) -> dict[str, Any]:
    return (
        client.structured(
            "You design fair peer-instruction experiments. Produce two different but isomorphic application questions that test the same underlying concept at equal difficulty. Do not leak either answer.",
            f"Topic: {topic}",
            TEACHBACK_SCHEMA,
            schema_name="teachback_pair",
            max_tokens=1500,
        ).payload
        or {}
    )


def score_transfer(client: GemmaClient, question: str, answer: str, rubric: list[str]) -> dict[str, Any]:
    return (
        client.structured(
            "Score only against the rubric. Acknowledge valid alternative reasoning. Be conservative and report uncertainty.",
            f"QUESTION: {question}\nRUBRIC: {rubric}\nANSWER: {answer}",
            SCORE_SCHEMA,
            schema_name="transfer_score",
            max_tokens=1100,
        ).payload
        or {}
    )


def score_teaching_explanation(client: GemmaClient, topic: str, explanation: str) -> dict[str, Any]:
    return (
        client.structured(
            "Evaluate a peer explanation for factual accuracy, conceptual depth, useful examples, and misconception handling. Do not reward verbosity or confidence by itself.",
            f"TOPIC: {topic}\nPEER EXPLANATION: {explanation}",
            SCORE_SCHEMA,
            schema_name="teaching_quality",
            max_tokens=1100,
        ).payload
        or {}
    )


def teaching_impact(pre_score: float, post_score: float, teaching_quality: float) -> dict[str, float]:
    raw_gain = max(-1.0, min(1.0, post_score - pre_score))
    positive_gain = max(0.0, raw_gain)
    impact = 100 * positive_gain * (0.6 + 0.4 * max(0.0, min(1.0, teaching_quality)))
    return {"pre": round(pre_score * 100, 1), "post": round(post_score * 100, 1), "gain": round(raw_gain * 100, 1), "impact": round(impact, 1)}
