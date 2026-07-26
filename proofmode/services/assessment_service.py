from __future__ import annotations

from typing import Any

from proofmode.gemma_client import GemmaClient


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


def generate_questions(client: GemmaClient, topic: dict[str, Any], target_mark: float, depth_mode: str) -> dict[str, Any]:
    user = (
        f"Topic record: {topic}\nTarget mark: {target_mark}. Depth: {depth_mode}. "
        "Create two diagnostic MCQs with plausible misconception-based distractors and one open transfer question. "
        "Do not merely test vocabulary. Keep all questions answerable from established knowledge about the topic."
    )
    return (
        client.structured(
            "You are a rigorous assessment designer. Questions must distinguish recall, application, and transfer. Avoid trick questions and reveal uncertainty if the topic record is underspecified.",
            user,
            QUESTION_SCHEMA,
            schema_name="retrieval_questions",
            max_tokens=2400,
        ).payload
        or {}
    )


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
        "Treat unreadable or absent evidence as uncertainty, not failure."
    )
    content: str | list[dict[str, Any]] = prompt
    modality = "text"
    if notes_data and notes_mime and notes_mime.startswith("image/"):
        content = [{"type": "text", "text": prompt}, client.file_part(notes_data, notes_mime)]
        modality = "image+text"
    return (
        client.structured(
            "You are ProofMode's evidence assessor. Be conservative, rubric-bound, specific, and kind. Never infer ability from presentation quality. State uncertainty and do not claim a precise grade prediction.",
            content,
            ASSESSMENT_SCHEMA,
            schema_name="learning_receipt",
            max_tokens=1800,
            modality=modality,
        ).payload
        or {}
    )
