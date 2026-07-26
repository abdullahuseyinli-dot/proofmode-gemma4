from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Mapping

from proofmode.gemma_client import GemmaClient, StructuredOutputError
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


SYSTEM = """You are ProofMode's curriculum analyst. Build a conservative topic and prerequisite map from only the supplied student material and calendar context. Infer likely difficulty and weighting, but explicitly capture uncertainty. Do not invent an official syllabus, exact exam weighting, or prior mastery. A neutral low starting mastery is appropriate until assessed. Keep topics distinct and actionable.

NUMERIC CONTRACT: exam_weight, difficulty, mastery, confidence, and prerequisite_centrality must each be a decimal fraction from 0.0 through 1.0 (for example 0.6). Never use a 1-to-5 rating, a percentage such as 60, or a fraction string such as "3/5". estimated_minutes must be a whole number from 15 through 360."""

RETRY_SYSTEM = SYSTEM + """

CONCISE RETRY CONTRACT: Return only 4 to 8 highest-value topics. Keep each description to at most 18 words, prerequisites to at most 3 short names, evidence to at most 2 short source labels, uncertainty to at most 8 words, and exam_summary to one sentence. Complete the JSON object; do not add commentary."""


# Gemma runs locally with an 8K context in the default laptop setup. These are
# collection-level character budgets (roughly four characters per text token),
# not per-file limits. They leave room for the system/schema, image tokens, and
# bounded model output. Retry is deliberately smaller despite its larger output
# allowance. Text-file count and image count are also bounded so uploads cannot
# grow a planner request without limit.
PLANNER_TEXT_BUDGET_CHARS = 9_000
PLANNER_RETRY_TEXT_BUDGET_CHARS = 4_500
PLANNER_EXAM_CONTEXT_BUDGET_CHARS = 1_500
PLANNER_RETRY_EXAM_CONTEXT_BUDGET_CHARS = 750
PLANNER_MAX_TEXT_DOCUMENTS = 24
PLANNER_MAX_IMAGES = 2
PLANNER_RETRY_MAX_IMAGES = 1


_TOPIC_REQUIRED_FIELDS = {
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
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _clamped_fraction(value: Any, field: str, minimum: float = 0.0) -> float:
    number = _finite_number(value, field)
    return round(min(1.0, max(minimum, number)), 4)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip()


def _normalise_string_list(value: Any, field: str, maximum: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    items: list[str] = []
    for raw_item in value:
        if not isinstance(raw_item, str):
            raise ValueError(f"{field} must contain only strings")
        item = raw_item.strip()
        if item and item not in items:
            items.append(item)
    return items[:maximum]


def normalise_topic(
    topic: Mapping[str, Any],
    *,
    difficulty_scale_1_to_5: bool = False,
) -> dict[str, Any]:
    """Return one planner topic with bounded, finite, schema-safe values."""

    missing = _TOPIC_REQUIRED_FIELDS - set(topic)
    if missing:
        raise ValueError(f"Planner topic is missing required fields: {', '.join(sorted(missing))}")

    difficulty = _finite_number(topic["difficulty"], "difficulty")
    if difficulty_scale_1_to_5:
        difficulty /= 5.0

    minutes = _finite_number(topic["estimated_minutes"], "estimated_minutes")
    return {
        "name": _required_text(topic["name"], "name"),
        "description": _text(topic["description"], "description"),
        "exam_weight": _clamped_fraction(topic["exam_weight"], "exam_weight", 0.05),
        "difficulty": _clamped_fraction(difficulty, "difficulty", 0.1),
        "mastery": _clamped_fraction(topic["mastery"], "mastery"),
        "confidence": _clamped_fraction(topic["confidence"], "confidence"),
        "prerequisite_centrality": _clamped_fraction(
            topic["prerequisite_centrality"], "prerequisite_centrality", 0.1
        ),
        "estimated_minutes": min(360, max(15, int(round(minutes)))),
        "prerequisites": _normalise_string_list(topic["prerequisites"], "prerequisites", 5),
        "evidence": _normalise_string_list(topic["evidence"], "evidence", 4),
        "uncertainty": _text(topic["uncertainty"], "uncertainty"),
    }


def normalise_course_map(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and bound a Gemma course map before scheduling consumes it.

    Some local models ignore JSON Schema numeric bounds. If every retained
    difficulty is on an obvious 1-to-5 scale and at least one exceeds 1, the
    complete scale is converted to fractions instead of flattening most topics
    to 1.0. Malformed and non-finite input raises ``ValueError`` so the existing
    caller can select the deterministic fallback plan.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("Course map must be an object")
    course_title = _required_text(payload.get("course_title"), "course_title")
    exam_summary = _text(payload.get("exam_summary"), "exam_summary")
    raw_topics = payload.get("topics")
    if not isinstance(raw_topics, list):
        raise ValueError("Course map topics must be a list")
    if len(raw_topics) < 2:
        raise ValueError("Course map must contain at least two topics")
    if any(not isinstance(topic, Mapping) for topic in raw_topics):
        raise ValueError("Every course-map topic must be an object")

    retained_topics = raw_topics[:12]
    raw_difficulties = [
        _finite_number(topic.get("difficulty"), f"topics[{index}].difficulty")
        for index, topic in enumerate(retained_topics)
    ]
    uses_five_point_scale = (
        bool(raw_difficulties)
        and all(1.0 <= value <= 5.0 for value in raw_difficulties)
        and any(value > 1.0 for value in raw_difficulties)
    )
    topics = [
        normalise_topic(topic, difficulty_scale_1_to_5=uses_five_point_scale)
        for topic in retained_topics
    ]
    return {
        "course_title": course_title,
        "exam_summary": exam_summary,
        "topics": topics,
    }


def _bounded_excerpt(text: str, maximum: int) -> str:
    """Keep both ends of text within an exact character limit."""

    clean = text.strip()
    if len(clean) <= maximum:
        return clean
    marker = "\n...[truncated]...\n"
    if maximum <= len(marker):
        return clean[:maximum]
    remaining = maximum - len(marker)
    head = (remaining + 1) // 2
    tail = remaining - head
    return clean[:head] + marker + (clean[-tail:] if tail else "")


def _evenly_spaced_documents(
    documents: list[StudyDocument],
    maximum: int,
) -> list[StudyDocument]:
    """Retain representation from the start, middle, and end of large uploads."""

    if len(documents) <= maximum:
        return documents
    if maximum <= 1:
        return documents[:1]
    indices = [round(index * (len(documents) - 1) / (maximum - 1)) for index in range(maximum)]
    return [documents[index] for index in indices]


def _safe_document_name(name: str) -> str:
    return " ".join(name.split())[:80] or "unnamed"


def _fair_text_allocations(texts: list[str], total: int) -> list[int]:
    """Water-fill a global body budget so no early file monopolizes it."""

    allocations = [0] * len(texts)
    active = set(range(len(texts)))
    remaining = max(0, total)
    while active and remaining:
        share = max(1, remaining // len(active))
        progressed = False
        for index in sorted(active):
            grant = min(share, len(texts[index]) - allocations[index], remaining)
            if grant > 0:
                allocations[index] += grant
                remaining -= grant
                progressed = True
            if allocations[index] >= len(texts[index]):
                active.discard(index)
            if not remaining:
                break
        if not progressed:
            break
    return allocations


def _bounded_document_material(
    documents: list[StudyDocument],
    maximum: int,
) -> str:
    """Render fairly sampled file sections inside one exact global cap."""

    text_documents = [document for document in documents if document.text.strip()]
    selected = _evenly_spaced_documents(text_documents, PLANNER_MAX_TEXT_DOCUMENTS)
    if not selected:
        return _bounded_excerpt(
            "No machine-readable text was supplied; inspect the attached images.",
            maximum,
        )

    names = [_safe_document_name(document.name) for document in selected]
    texts = [document.text.strip() for document in selected]
    headers = [f"FILE {name}:\n" for name in names]
    separator = "\n\n"
    overhead = sum(len(header) for header in headers) + len(separator) * (len(headers) - 1)
    if overhead >= maximum:
        # Names are bounded and at most 24 are retained, so this is only a guard
        # for callers supplying an unusually tiny custom limit.
        return separator.join(headers)[:maximum]
    allocations = _fair_text_allocations(texts, maximum - overhead)
    sections = [
        header + _bounded_excerpt(text, allocation)
        for header, text, allocation in zip(headers, texts, allocations)
    ]
    return separator.join(sections)


def _planner_content(
    client: GemmaClient,
    documents: list[StudyDocument],
    exam_context: str,
    target_mark: float,
    depth_mode: str,
    *,
    text_budget: int,
    exam_budget: int,
    max_images: int,
) -> list[dict[str, Any]]:
    material = _bounded_document_material(documents, text_budget)
    bounded_exam = _bounded_excerpt(
        exam_context or "No reliable exam event found.",
        exam_budget,
    )
    images = [document for document in documents if document.is_image][:max_images]
    image_names = ", ".join(_safe_document_name(document.name) for document in images) or "none"
    prompt = (
        f"Target mark: {target_mark}%. Derived depth mode: {depth_mode}.\n"
        f"Calendar/exam context (bounded):\n{bounded_exam}\n\n"
        f"Attached image files (bounded): {image_names}\n"
        "UPLOADED MATERIAL (globally bounded across files):\n"
        "<<<MATERIAL>>>\n"
        f"{material}\n"
        "<<<END MATERIAL>>>"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(client.file_part(document.data, document.mime_type) for document in images)
    return content


def extract_course_map(
    client: GemmaClient,
    documents: list[StudyDocument],
    exam_context: str,
    target_mark: float,
    depth_mode: str,
) -> dict[str, Any]:
    content = _planner_content(
        client,
        documents,
        exam_context,
        target_mark,
        depth_mode,
        text_budget=PLANNER_TEXT_BUDGET_CHARS,
        exam_budget=PLANNER_EXAM_CONTEXT_BUDGET_CHARS,
        max_images=PLANNER_MAX_IMAGES,
    )
    modality = "image+text" if len(content) > 1 else "text"
    try:
        result = client.structured(
            SYSTEM,
            content,
            TOPIC_SCHEMA,
            schema_name="course_map",
            max_tokens=1800,
            modality=modality,
        )
    except StructuredOutputError:
        # Local E4B occasionally reaches the first output limit mid-object.
        # Retry once with both a smaller requested map and smaller input payload;
        # the larger output allowance therefore still fits the local 8K context.
        retry_content = _planner_content(
            client,
            documents,
            exam_context,
            target_mark,
            depth_mode,
            text_budget=PLANNER_RETRY_TEXT_BUDGET_CHARS,
            exam_budget=PLANNER_RETRY_EXAM_CONTEXT_BUDGET_CHARS,
            max_images=PLANNER_RETRY_MAX_IMAGES,
        )
        retry_modality = "image+text" if len(retry_content) > 1 else "text"
        result = client.structured(
            RETRY_SYSTEM,
            retry_content,
            TOPIC_SCHEMA,
            schema_name="course_map",
            max_tokens=2600,
            modality=retry_modality,
        )
    return normalise_course_map(result.payload)


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
