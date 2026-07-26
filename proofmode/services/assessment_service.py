from __future__ import annotations

import math
from typing import Any

from proofmode.gemma_client import GemmaClient, StructuredOutputError


QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mcqs", "open_question", "open_rubric"],
    "properties": {
        "mcqs": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "options", "correct_index", "explanation", "skill"],
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string"}},
                    "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
                    "explanation": {"type": "string"},
                    "skill": {"type": "string"},
                },
            },
        },
        "open_question": {"type": "string"},
        "open_rubric": {"type": "array", "minItems": 2, "maxItems": 5, "items": {"type": "string"}},
    },
}


ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "correctness",
        "depth",
        "evidence_quality",
        "coverage",
        "strengths",
        "misconceptions",
        "feedback",
        "next_probe",
        "uncertainty",
    ],
    "properties": {
        "correctness": {"type": "number", "minimum": 0, "maximum": 1},
        "depth": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_quality": {"type": "number", "minimum": 0, "maximum": 1},
        "coverage": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "strengths": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "misconceptions": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "feedback": {"type": "string"},
        "next_probe": {"type": "string"},
        "uncertainty": {"type": "string"},
    },
}


_ASSESSMENT_LIST_LIMITS = {
    "coverage": 8,
    "strengths": 5,
    "misconceptions": 5,
}


def _normalise_model_score(value: Any, field: str) -> float:
    """Convert common model rating scales to a bounded decimal fraction."""

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


def _normalise_question_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("question result must be an object")
    normalised = dict(payload)
    raw_mcqs = payload.get("mcqs")
    if not isinstance(raw_mcqs, list) or not 2 <= len(raw_mcqs) <= 3:
        raise ValueError("mcqs must contain 2 to 3 questions")
    mcqs: list[Any] = []
    for index, raw_mcq in enumerate(raw_mcqs):
        if not isinstance(raw_mcq, dict):
            raise ValueError(f"mcqs[{index}] must be an object")
        mcq = dict(raw_mcq)
        for field in ("question", "explanation", "skill"):
            if not isinstance(raw_mcq.get(field), str):
                raise ValueError(f"mcqs[{index}].{field} must be a string")
        options = raw_mcq.get("options")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"mcqs[{index}].options must contain exactly 4 items")
        if any(not isinstance(option, str) for option in options):
            raise ValueError(f"mcqs[{index}].options must contain only strings")
        correct_index = raw_mcq.get("correct_index")
        if isinstance(correct_index, bool) or not isinstance(correct_index, int) or not 0 <= correct_index <= 3:
            raise ValueError(f"mcqs[{index}].correct_index must be an integer from 0 to 3")
        mcq["options"] = list(options)
        mcqs.append(mcq)
    if not isinstance(payload.get("open_question"), str):
        raise ValueError("open_question must be a string")
    open_rubric = payload.get("open_rubric")
    if not isinstance(open_rubric, list) or not 2 <= len(open_rubric) <= 5:
        raise ValueError("open_rubric must contain 2 to 5 items")
    if any(not isinstance(item, str) for item in open_rubric):
        raise ValueError("open_rubric must contain only strings")
    normalised["mcqs"] = mcqs
    normalised["open_rubric"] = list(open_rubric)
    return normalised


def _normalise_assessment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("learning receipt must be an object")
    normalised = dict(payload)
    for field in ("correctness", "depth", "evidence_quality"):
        normalised[field] = _normalise_model_score(payload.get(field), field)
    for field, maximum in _ASSESSMENT_LIST_LIMITS.items():
        normalised[field] = _bounded_list(payload.get(field), field, maximum)
    return normalised


def _validated_payload(payload: Any, normaliser: Any, schema_name: str) -> dict[str, Any]:
    """Translate only local payload-validation failures to the UI-safe error."""

    try:
        return normaliser(payload)
    except (TypeError, ValueError, KeyError) as error:
        raise StructuredOutputError(f"Gemma returned invalid {schema_name} payload: {error}") from error


def _structured_assessment_with_retry(
    client: GemmaClient,
    system: str,
    content: str | list[dict[str, Any]],
    *,
    modality: str,
) -> dict[str, Any]:
    """Request an assessment, retrying once only for malformed/truncated JSON."""

    try:
        result = client.structured(
            system,
            content,
            ASSESSMENT_SCHEMA,
            schema_name="learning_receipt",
            max_tokens=1800,
            modality=modality,
        )
    except StructuredOutputError:
        result = client.structured(
            system
            + " Keep coverage, strengths, and misconceptions to at most 3 concise items each; "
            "keep feedback under 35 words and next_probe and uncertainty under 20 words each. "
            "Complete the JSON object.",
            content,
            ASSESSMENT_SCHEMA,
            schema_name="learning_receipt_retry",
            max_tokens=2600,
            modality=modality,
        )
    return result.payload or {}


def generate_questions(client: GemmaClient, topic: dict[str, Any], target_mark: float, depth_mode: str) -> dict[str, Any]:
    user = (
        f"Topic record: {topic}\nTarget mark: {target_mark}. Depth: {depth_mode}. "
        "Create two diagnostic MCQs with plausible misconception-based distractors and one open transfer question. "
        "Do not merely test vocabulary. Treat the topic record as the sole evidence: use only concepts, "
        "relationships, and examples explicitly present in its name, description, prerequisites, or evidence. "
        "Do not introduce adjacent facts or entities. If the record is underspecified, test interpretation of "
        "the supplied evidence and state that limitation in the explanation."
    )
    payload = (
        client.structured(
            "You are a rigorous, evidence-grounded assessment designer. Questions must distinguish recall, application, and transfer. Avoid trick questions and never fill gaps in the supplied topic record from memory.",
            user,
            QUESTION_SCHEMA,
            schema_name="retrieval_questions",
            max_tokens=2400,
        ).payload
        or {}
    )
    return _validated_payload(payload, _normalise_question_payload, "retrieval_questions")


def assess_learning_receipt(
    client: GemmaClient,
    topic: dict[str, Any],
    question: str,
    answer: str,
    rubric: list[str],
    notes_data: bytes | None = None,
    notes_mime: str | None = None,
) -> dict[str, Any]:
    prompt = (
        f"TOPIC: {topic}\nQUESTION: {question}\nRUBRIC: {rubric}\nSTUDENT ANSWER: {answer}\n\n"
        "Score conceptual correctness and transfer depth. If notes are attached, score only conceptual coverage, links, worked reasoning, and visible misconceptions—not neatness, handwriting, length, or aesthetics. "
        "Treat unreadable or absent evidence as uncertainty, not failure. "
        "Return correctness, depth, and evidence_quality only as decimal fractions from 0.0 through 1.0 "
        "(for example 0.8), never as a 1-to-5 rating or a percentage."
    )
    content: str | list[dict[str, Any]] = prompt
    modality = "text"
    if notes_data and notes_mime and notes_mime.startswith("image/"):
        content = [{"type": "text", "text": prompt}, client.file_part(notes_data, notes_mime)]
        modality = "image+text"
    payload = _structured_assessment_with_retry(
        client,
        "You are ProofMode's evidence assessor. Be conservative, rubric-bound, specific, and kind. Never infer ability from presentation quality. State uncertainty and do not claim a precise grade prediction. All numeric scores must be decimal fractions from 0.0 through 1.0, never 1-to-5 ratings or percentages.",
        content,
        modality=modality,
    )
    return _validated_payload(payload, _normalise_assessment_payload, "learning_receipt")
