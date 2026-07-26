from __future__ import annotations

from copy import deepcopy
import math
from types import SimpleNamespace
from typing import Any

import pytest

from proofmode.gemma_client import StructuredOutputError
from proofmode.services.document_service import StudyDocument
from proofmode.services.planner_service import (
    PLANNER_EXAM_CONTEXT_BUDGET_CHARS,
    PLANNER_MAX_IMAGES,
    PLANNER_RETRY_EXAM_CONTEXT_BUDGET_CHARS,
    PLANNER_RETRY_MAX_IMAGES,
    PLANNER_RETRY_TEXT_BUDGET_CHARS,
    PLANNER_TEXT_BUDGET_CHARS,
    extract_course_map,
    normalise_course_map,
)


def _topic(name: str, *, difficulty: float = 0.6) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Study {name}",
        "exam_weight": 0.7,
        "difficulty": difficulty,
        "mastery": 0.2,
        "confidence": 0.3,
        "prerequisite_centrality": 0.5,
        "estimated_minutes": 60,
        "prerequisites": ["Foundations"],
        "evidence": ["Uploaded notes"],
        "uncertainty": "Weight is inferred.",
    }


def _course_map(*topics: dict[str, Any]) -> dict[str, Any]:
    return {
        "course_title": "Real course",
        "exam_summary": "Exam inferred from calendar.",
        "topics": list(topics),
    }


def test_obvious_one_to_five_difficulty_scale_is_converted_as_a_whole() -> None:
    payload = _course_map(
        *[_topic(f"Topic {index}", difficulty=value) for index, value in enumerate([2, 1, 3, 2, 2])]
    )

    result = normalise_course_map(payload)

    assert [topic["difficulty"] for topic in result["topics"]] == [0.4, 0.2, 0.6, 0.4, 0.4]


def test_course_map_clamps_fractions_minutes_and_list_sizes() -> None:
    first = _topic("One")
    first.update(
        {
            "exam_weight": 9,
            "difficulty": -4,
            "mastery": -2,
            "confidence": 8,
            "prerequisite_centrality": 4,
            "estimated_minutes": 999,
            "prerequisites": [f"P{index}" for index in range(8)],
            "evidence": [f"E{index}" for index in range(7)],
        }
    )
    second = _topic("Two")
    second["estimated_minutes"] = -100
    extra = [_topic(f"Extra {index}") for index in range(13)]

    result = normalise_course_map(_course_map(first, second, *extra))
    normalised = result["topics"][0]

    assert len(result["topics"]) == 12
    assert normalised["exam_weight"] == 1.0
    assert normalised["difficulty"] == 0.1
    assert normalised["mastery"] == 0.0
    assert normalised["confidence"] == 1.0
    assert normalised["prerequisite_centrality"] == 1.0
    assert normalised["estimated_minutes"] == 360
    assert result["topics"][1]["estimated_minutes"] == 15
    assert len(normalised["prerequisites"]) == 5
    assert len(normalised["evidence"]) == 4


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    [
        "exam_weight",
        "difficulty",
        "mastery",
        "confidence",
        "prerequisite_centrality",
        "estimated_minutes",
    ],
)
def test_nonfinite_numeric_fields_fail_safely(field: str, nonfinite: float) -> None:
    topic = _topic("Unsafe")
    topic[field] = nonfinite

    with pytest.raises(ValueError, match="finite"):
        normalise_course_map(_course_map(topic, _topic("Valid")))


@pytest.mark.parametrize("malformed", ["0.7", True, None, [], {}])
def test_malformed_numeric_fields_are_rejected(malformed: Any) -> None:
    topic = _topic("Unsafe")
    topic["confidence"] = malformed

    with pytest.raises(ValueError, match="finite number"):
        normalise_course_map(_course_map(topic, _topic("Valid")))


def test_malformed_structure_and_lists_fail_safely() -> None:
    with pytest.raises(ValueError, match="at least two"):
        normalise_course_map(_course_map(_topic("Only")))

    payload = _course_map(_topic("One"), _topic("Two"))
    payload["topics"][0]["prerequisites"] = ["Valid", 42]
    with pytest.raises(ValueError, match="only strings"):
        normalise_course_map(payload)

    missing = _course_map(_topic("One"), _topic("Two"))
    del missing["topics"][0]["mastery"]
    with pytest.raises(ValueError, match="missing required"):
        normalise_course_map(missing)


def test_extract_course_map_emphasises_fraction_contract_and_normalises_payload() -> None:
    payload = _course_map(_topic("One", difficulty=2), _topic("Two", difficulty=4))

    class FakeClient:
        system = ""

        def structured(self, system: str, content: Any, schema: Any, **kwargs: Any):
            self.system = system
            return SimpleNamespace(payload=deepcopy(payload))

        def file_part(self, data: bytes, mime_type: str) -> dict[str, Any]:
            raise AssertionError("No image input expected")

    client = FakeClient()
    result = extract_course_map(client, [], "Exam in two weeks", 80, "transfer")  # type: ignore[arg-type]

    assert "decimal fraction from 0.0 through 1.0" in client.system
    assert "Never use a 1-to-5 rating" in client.system
    assert [topic["difficulty"] for topic in result["topics"]] == [0.4, 0.8]


def _material_from_content(content: list[dict[str, Any]]) -> str:
    prompt = content[0]["text"]
    return prompt.split("<<<MATERIAL>>>\n", 1)[1].split("\n<<<END MATERIAL>>>", 1)[0]


def _exam_context_from_content(content: list[dict[str, Any]]) -> str:
    prompt = content[0]["text"]
    return prompt.split("Calendar/exam context (bounded):\n", 1)[1].split(
        "\n\nAttached image files",
        1,
    )[0]


def test_course_map_uses_one_fair_global_text_budget_across_documents() -> None:
    payload = _course_map(_topic("One"), _topic("Two"))

    class CaptureClient:
        calls: list[list[dict[str, Any]]] = []

        def structured(self, system: str, content: Any, schema: Any, **kwargs: Any):
            self.calls.append(content)
            return SimpleNamespace(payload=deepcopy(payload))

        def file_part(self, data: bytes, mime_type: str) -> dict[str, Any]:
            raise AssertionError("No image input expected")

    documents = [
        StudyDocument("first.txt", "text/plain", b"", "A" * 20_000),
        StudyDocument("middle.txt", "text/plain", b"", "B" * 20_000),
        StudyDocument("last.txt", "text/plain", b"", "C" * 20_000),
    ]
    client = CaptureClient()

    extract_course_map(  # type: ignore[arg-type]
        client,
        documents,
        "E" * 10_000,
        80,
        "transfer",
    )

    content = client.calls[0]
    material = _material_from_content(content)
    exam_context = _exam_context_from_content(content)
    assert len(material) <= PLANNER_TEXT_BUDGET_CHARS
    assert len(exam_context) <= PLANNER_EXAM_CONTEXT_BUDGET_CHARS
    assert all(f"FILE {name}:" in material for name in ("first.txt", "middle.txt", "last.txt"))
    represented = [material.count(character) for character in "ABC"]
    assert min(represented) > 0
    assert max(represented) - min(represented) <= 1


def test_course_map_caps_images_while_representing_multiple_text_files() -> None:
    payload = _course_map(_topic("One"), _topic("Two"))

    class CaptureClient:
        calls: list[dict[str, Any]] = []
        attached: list[bytes] = []

        def structured(self, system: str, content: Any, schema: Any, **kwargs: Any):
            self.calls.append({"content": content, **kwargs})
            return SimpleNamespace(payload=deepcopy(payload))

        def file_part(self, data: bytes, mime_type: str) -> dict[str, Any]:
            self.attached.append(data)
            return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data.hex()}"}}

    text_documents = [
        StudyDocument(f"notes-{index}.txt", "text/plain", b"", f"topic-{index} " * 2_000)
        for index in range(4)
    ]
    image_documents = [
        StudyDocument(f"image-{index}.png", "image/png", bytes([index]))
        for index in range(5)
    ]
    client = CaptureClient()

    extract_course_map(  # type: ignore[arg-type]
        client,
        [*text_documents, *image_documents],
        "Exam soon",
        75,
        "apply",
    )

    content = client.calls[0]["content"]
    material = _material_from_content(content)
    assert len(content) == 1 + PLANNER_MAX_IMAGES
    assert client.attached == [b"\x00", b"\x01"]
    assert client.calls[0]["modality"] == "image+text"
    assert all(f"FILE notes-{index}.txt:" in material for index in range(4))


def test_course_map_retry_rebuilds_a_smaller_text_and_image_payload() -> None:
    payload = _course_map(_topic("One"), _topic("Two"))

    class RetryCaptureClient:
        calls: list[dict[str, Any]] = []

        def structured(self, system: str, content: Any, schema: Any, **kwargs: Any):
            self.calls.append({"system": system, "content": content, **kwargs})
            if len(self.calls) == 1:
                raise StructuredOutputError("truncated")
            return SimpleNamespace(payload=deepcopy(payload))

        def file_part(self, data: bytes, mime_type: str) -> dict[str, Any]:
            return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data.hex()}"}}

    documents = [
        StudyDocument(f"long-{index}.txt", "text/plain", b"", str(index) * 20_000)
        for index in range(3)
    ] + [
        StudyDocument(f"scan-{index}.png", "image/png", bytes([index]))
        for index in range(3)
    ]
    client = RetryCaptureClient()

    extract_course_map(  # type: ignore[arg-type]
        client,
        documents,
        "E" * 10_000,
        85,
        "transfer",
    )

    initial = client.calls[0]["content"]
    retry = client.calls[1]["content"]
    initial_material = _material_from_content(initial)
    retry_material = _material_from_content(retry)
    assert len(initial_material) <= PLANNER_TEXT_BUDGET_CHARS
    assert len(retry_material) <= PLANNER_RETRY_TEXT_BUDGET_CHARS
    assert len(retry_material) < len(initial_material)
    assert len(_exam_context_from_content(retry)) <= PLANNER_RETRY_EXAM_CONTEXT_BUDGET_CHARS
    assert len(initial) == 1 + PLANNER_MAX_IMAGES
    assert len(retry) == 1 + PLANNER_RETRY_MAX_IMAGES
    assert all(f"FILE long-{index}.txt:" in retry_material for index in range(3))


def test_extract_course_map_retries_one_truncated_structured_result_concisely() -> None:
    payload = _course_map(_topic("One"), _topic("Two"))

    class RetryClient:
        calls: list[dict[str, Any]] = []

        def structured(self, system: str, content: Any, schema: Any, **kwargs: Any):
            self.calls.append({"system": system, **kwargs})
            if len(self.calls) == 1:
                raise StructuredOutputError("first JSON object was truncated")
            return SimpleNamespace(payload=deepcopy(payload))

        def file_part(self, data: bytes, mime_type: str) -> dict[str, Any]:
            raise AssertionError("No image input expected")

    client = RetryClient()
    result = extract_course_map(client, [], "Exam in two weeks", 80, "transfer")  # type: ignore[arg-type]

    assert len(client.calls) == 2
    assert client.calls[0]["max_tokens"] == 1800
    assert client.calls[0]["schema_name"] == "course_map"
    assert "CONCISE RETRY CONTRACT" not in client.calls[0]["system"]
    assert client.calls[1]["max_tokens"] == 2600
    assert client.calls[1]["schema_name"] == "course_map"
    assert "Return only 4 to 8 highest-value topics" in client.calls[1]["system"]
    assert [topic["name"] for topic in result["topics"]] == ["One", "Two"]


def test_extract_course_map_propagates_second_structured_failure_after_one_retry() -> None:
    class FailingClient:
        calls: list[dict[str, Any]] = []

        def structured(self, system: str, content: Any, schema: Any, **kwargs: Any):
            self.calls.append({"system": system, **kwargs})
            raise StructuredOutputError(f"failure {len(self.calls)}")

        def file_part(self, data: bytes, mime_type: str) -> dict[str, Any]:
            raise AssertionError("No image input expected")

    client = FailingClient()
    with pytest.raises(StructuredOutputError, match="failure 2"):
        extract_course_map(client, [], "Exam in two weeks", 80, "transfer")  # type: ignore[arg-type]

    assert len(client.calls) == 2
    assert [call["max_tokens"] for call in client.calls] == [1800, 2600]


def test_extract_course_map_does_not_retry_non_structured_failures() -> None:
    class UnavailableClient:
        calls = 0

        def structured(self, system: str, content: Any, schema: Any, **kwargs: Any):
            self.calls += 1
            raise RuntimeError("model offline")

        def file_part(self, data: bytes, mime_type: str) -> dict[str, Any]:
            raise AssertionError("No image input expected")

    client = UnavailableClient()
    with pytest.raises(RuntimeError, match="model offline"):
        extract_course_map(client, [], "Exam in two weeks", 80, "transfer")  # type: ignore[arg-type]

    assert client.calls == 1
