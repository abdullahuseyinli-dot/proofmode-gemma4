from __future__ import annotations

from datetime import datetime
from typing import Any

from proofmode.gemma_client import GemmaClient
from proofmode.learning_engine import topic_priority
from proofmode.services.document_service import StudyDocument


TOPIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["course_title", "exam_summary", "topics"],
    "properties": {
        "course_title": {"type": "string"},
        "exam_summary": {"type": "string"},
        "topics": {
            "type": "array",
            "minItems": 2,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "description",
                    "exam_weight",
                    "difficulty",
                    "mastery",
                    "confidence",
                    "prerequisite_centrality",
                    "estimated_minutes",
                    "prerequisites",
                    "evidence",
                    "uncertainty",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "exam_weight": {"type": "number", "minimum": 0.05, "maximum": 1},
                    "difficulty": {"type": "number", "minimum": 0.1, "maximum": 1},
                    "mastery": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "prerequisite_centrality": {"type": "number", "minimum": 0.1, "maximum": 1},
                    "estimated_minutes": {"type": "integer", "minimum": 15, "maximum": 360},
                    "prerequisites": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                    "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                    "uncertainty": {"type": "string"},
                },
            },
        },
    },
}


SYSTEM = """You are ProofMode's curriculum analyst. Build a conservative topic and prerequisite map from only the supplied student material and calendar context. Infer likely difficulty and weighting, but explicitly capture uncertainty. Do not invent an official syllabus, exact exam weighting, or prior mastery. A neutral low starting mastery is appropriate until assessed. Keep topics distinct and actionable."""


def extract_course_map(
    client: GemmaClient,
    documents: list[StudyDocument],
    exam_context: str,
    target_mark: float,
    depth_mode: str,
) -> dict[str, Any]:
    text_chunks = [f"FILE {doc.name}:\n{doc.text[:12000]}" for doc in documents if doc.text]
    prompt = (
        f"Target mark: {target_mark}%. Derived depth mode: {depth_mode}.\n"
        f"Calendar/exam context:\n{exam_context or 'No reliable exam event found.'}\n\n"
        + ("\n\n".join(text_chunks) or "No machine-readable text was supplied; inspect the attached images.")
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for document in documents:
        if document.is_image:
            content.append(client.file_part(document.data, document.mime_type))
    result = client.structured(
        SYSTEM,
        content,
        TOPIC_SCHEMA,
        schema_name="course_map",
        max_tokens=1800,
        modality="image+text" if len(content) > 1 else "text",
    )
    return result.payload or {}


def prioritise_topics(topics: list[dict[str, Any]], exam_date: datetime, depth_mode: str) -> list[dict[str, Any]]:
    prioritised = []
    for topic in topics:
        item = dict(topic)
        item["priority"] = topic_priority(item, exam_date, depth_mode)
        prioritised.append(item)
    return sorted(prioritised, key=lambda value: value["priority"], reverse=True)


def fallback_course_map() -> dict[str, Any]:
    return {
        "course_title": "Machine Learning Foundations",
        "exam_summary": "Demo data — replace with calendar and notes for a personal plan.",
        "topics": [
            {
                "name": "Backpropagation",
                "description": "Chain rule, computational graphs and gradient flow.",
                "exam_weight": 0.85,
                "difficulty": 0.8,
                "mastery": 0.28,
                "confidence": 0.35,
                "prerequisite_centrality": 0.9,
                "estimated_minutes": 90,
                "prerequisites": ["Derivatives", "Linear algebra"],
                "evidence": ["Seeded demo diagnostic"],
                "uncertainty": "Exam weighting is demonstration data.",
            },
            {
                "name": "Bias–variance trade-off",
                "description": "Generalisation, underfitting, overfitting and regularisation.",
                "exam_weight": 0.7,
                "difficulty": 0.65,
                "mastery": 0.42,
                "confidence": 0.62,
                "prerequisite_centrality": 0.7,
                "estimated_minutes": 60,
                "prerequisites": ["Probability"],
                "evidence": ["Seeded demo diagnostic"],
                "uncertainty": "Replace with an actual baseline answer.",
            },
            {
                "name": "Model evaluation",
                "description": "Cross-validation, leakage, classification and regression metrics.",
                "exam_weight": 0.65,
                "difficulty": 0.55,
                "mastery": 0.55,
                "confidence": 0.7,
                "prerequisite_centrality": 0.6,
                "estimated_minutes": 45,
                "prerequisites": ["Probability"],
                "evidence": ["Seeded demo diagnostic"],
                "uncertainty": "Replace with an actual baseline answer.",
            },
        ],
    }


def learning_contract(topic: dict[str, Any], depth_mode: str) -> str:
    depth_task = {
        "core": "recall the key definition and one example",
        "apply": "solve one representative problem without notes",
        "transfer": "solve a new scenario and justify every step",
        "teach_research": "teach the mechanism, its limits, and a contrasting example",
    }.get(depth_mode, "explain and apply the idea")
    return f"Proof: {depth_task}; finish with one confidence-rated retrieval question on {topic['name']}."

