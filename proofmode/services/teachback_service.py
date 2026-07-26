from __future__ import annotations

import math
from typing import Any

from proofmode.gemma_client import GemmaClient, StructuredOutputError


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


_SCORE_LIST_LIMITS = {
    "accurate_points": 5,
    "missing_points": 5,
    "misconceptions": 5,
}


def _normalise_model_score(value: Any, field: str = "score") -> float:
    """Recover fraction, 1-to-5, and percentage model score formats safely."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite numeric score")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite numeric score")
    if number < 0.0 or number > 100.0:
        raise ValueError(f"{field} must be between 0 and 100")
    if number > 5.0:
        number /= 100.0
    elif number > 1.0:
        number /= 5.0
    return number


def _bounded_list(value: Any, field: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must contain only strings")
    return list(value[:maximum])


def _normalise_pair_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("teachback pair must be an object")
    normalised = dict(payload)
    for field in ("pre_question", "post_question", "concept_invariant"):
        if not isinstance(payload.get(field), str):
            raise ValueError(f"{field} must be a string")
    rubric = _bounded_list(payload.get("rubric"), "rubric", 5)
    if len(rubric) < 3:
        raise ValueError("rubric must contain at least 3 items")
    normalised["rubric"] = rubric
    return normalised


def _normalise_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("teachback score must be an object")
    normalised = dict(payload)
    normalised["score"] = _normalise_model_score(payload.get("score"))
    for field, maximum in _SCORE_LIST_LIMITS.items():
        normalised[field] = _bounded_list(payload.get(field), field, maximum)
    return normalised


def _validated_payload(payload: Any, normaliser: Any, schema_name: str) -> dict[str, Any]:
    """Translate only local payload-validation failures to the UI-safe error."""

    try:
        return normaliser(payload)
    except (TypeError, ValueError, KeyError) as error:
        raise StructuredOutputError(f"Gemma returned invalid {schema_name} payload: {error}") from error


def _structured_score_with_retry(
    client: GemmaClient,
    system: str,
    user: str,
    *,
    schema_name: str,
) -> dict[str, Any]:
    """Request a score, retrying once only when structured JSON is invalid."""

    try:
        result = client.structured(
            system,
            user,
            SCORE_SCHEMA,
            schema_name=schema_name,
            max_tokens=1100,
        )
    except StructuredOutputError:
        result = client.structured(
            system
            + " Keep the response terse: return at most 3 concise items in each list, "
            "feedback in at most 30 words, and uncertainty in at most 15 words. Complete the JSON object.",
            user,
            SCORE_SCHEMA,
            schema_name=f"{schema_name}_retry",
            max_tokens=1600,
        )
    return result.payload or {}


def generate_transfer_pair(client: GemmaClient, topic: dict[str, Any]) -> dict[str, Any]:
    payload = (
        client.structured(
            "You design fair peer-instruction experiments. Produce two different but isomorphic application questions that test the same underlying concept at equal difficulty. Do not leak either answer.",
            f"Topic: {topic}",
            TEACHBACK_SCHEMA,
            schema_name="teachback_pair",
            max_tokens=1500,
        ).payload
        or {}
    )
    return _validated_payload(payload, _normalise_pair_payload, "teachback_pair")


def score_transfer(client: GemmaClient, question: str, answer: str, rubric: list[str]) -> dict[str, Any]:
    payload = _structured_score_with_retry(
        client,
        "Score only against the rubric. Acknowledge valid alternative reasoning. Be conservative and report uncertainty. Return score only as a decimal fraction from 0.0 through 1.0 (for example 0.8), never as a 1-to-5 rating or a percentage.",
        f"QUESTION: {question}\nRUBRIC: {rubric}\nANSWER: {answer}",
        schema_name="transfer_score",
    )
    return _validated_payload(payload, _normalise_score_payload, "transfer_score")


def score_teaching_explanation(client: GemmaClient, topic: str, explanation: str) -> dict[str, Any]:
    payload = _structured_score_with_retry(
        client,
        "Evaluate a peer explanation for factual accuracy, conceptual depth, useful examples, and misconception handling. Do not reward verbosity or confidence by itself. Return score only as a decimal fraction from 0.0 through 1.0 (for example 0.8), never as a 1-to-5 rating or a percentage.",
        f"TOPIC: {topic}\nPEER EXPLANATION: {explanation}",
        schema_name="teaching_quality",
    )
    return _validated_payload(payload, _normalise_score_payload, "teaching_quality")


def teaching_impact(pre_score: float, post_score: float, teaching_quality: float) -> dict[str, float]:
    raw_gain = max(-1.0, min(1.0, post_score - pre_score))
    positive_gain = max(0.0, raw_gain)
    impact = 100 * positive_gain * (0.6 + 0.4 * max(0.0, min(1.0, teaching_quality)))
    return {"pre": round(pre_score * 100, 1), "post": round(post_score * 100, 1), "gain": round(raw_gain * 100, 1), "impact": round(impact, 1)}
