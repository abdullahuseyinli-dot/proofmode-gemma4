from __future__ import annotations

from copy import deepcopy
import math
from types import SimpleNamespace
from typing import Any

import pytest

from proofmode.gemma_client import StructuredOutputError
from proofmode.services.assessment_service import assess_learning_receipt, generate_questions
from proofmode.services.teachback_service import (
    generate_transfer_pair,
    score_teaching_explanation,
    score_transfer,
)


class StubClient:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def structured(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((args, kwargs))
        return SimpleNamespace(payload=deepcopy(self.payload))


class ScriptedClient:
    def __init__(self, actions: list[Any]):
        self.actions = list(actions)
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def structured(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((args, kwargs))
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return SimpleNamespace(payload=deepcopy(action))


def _assessment_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "correctness": 0.7,
        "depth": 0.6,
        "evidence_quality": 0.5,
        "coverage": [f"coverage-{index}" for index in range(10)],
        "strengths": [f"strength-{index}" for index in range(7)],
        "misconceptions": [f"misconception-{index}" for index in range(7)],
        "feedback": "Useful feedback",
        "next_probe": "Apply it to a new case.",
        "uncertainty": "None",
    }
    payload.update(overrides)
    return payload


def _score_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "score": 0.75,
        "accurate_points": [f"accurate-{index}" for index in range(7)],
        "missing_points": [f"missing-{index}" for index in range(7)],
        "misconceptions": [f"misconception-{index}" for index in range(7)],
        "feedback": "Useful feedback",
        "uncertainty": "None",
    }
    payload.update(overrides)
    return payload


def _question_payload() -> dict[str, Any]:
    return {
        "mcqs": [
            {
                "question": f"Question {index}",
                "options": [f"Option {option}" for option in range(4)],
                "correct_index": 0,
                "explanation": "Explanation",
                "skill": "application",
            }
            for index in range(3)
        ],
        "open_question": "Transfer question",
        "open_rubric": [f"Criterion {index}" for index in range(5)],
    }


def _assess(client: StubClient) -> dict[str, Any]:
    return assess_learning_receipt(
        client,  # type: ignore[arg-type]
        {"name": "Eigenvalues"},
        "Explain the zero eigenvalue case.",
        "It corresponds to a non-trivial nullspace.",
        ["Connect zero eigenvalue and singularity."],
    )


def test_learning_receipt_recovers_common_scales_caps_lists_and_keeps_shape() -> None:
    raw = _assessment_payload(correctness=4, depth=73, evidence_quality=0.25)
    client = StubClient(raw)

    result = _assess(client)

    assert result["correctness"] == 0.8
    assert result["depth"] == 0.73
    assert result["evidence_quality"] == 0.25
    assert len(result["coverage"]) == 8
    assert len(result["strengths"]) == 5
    assert len(result["misconceptions"]) == 5
    assert set(result) == set(raw)
    assert len(raw["coverage"]) == 10
    system, user = client.calls[0][0][:2]
    assert "decimal fractions from 0.0 through 1.0" in system
    assert "never as a 1-to-5 rating or a percentage" in user


def test_learning_receipt_retries_one_truncated_response_with_terse_contract() -> None:
    client = ScriptedClient(
        [StructuredOutputError("truncated JSON"), _assessment_payload(correctness=4)]
    )

    result = _assess(client)  # type: ignore[arg-type]

    assert result["correctness"] == 0.8
    assert len(client.calls) == 2
    first_args, first_kwargs = client.calls[0]
    retry_args, retry_kwargs = client.calls[1]
    assert first_kwargs["schema_name"] == "learning_receipt"
    assert first_kwargs["max_tokens"] == 1800
    assert retry_kwargs["schema_name"] == "learning_receipt_retry"
    assert retry_kwargs["max_tokens"] == 2600
    assert retry_args[1] == first_args[1]
    assert "at most 3 concise items each" in retry_args[0]
    assert "Complete the JSON object" in retry_args[0]


def test_learning_receipt_second_structured_failure_propagates() -> None:
    client = ScriptedClient(
        [StructuredOutputError("first failure"), StructuredOutputError("second failure")]
    )

    with pytest.raises(StructuredOutputError, match="second failure"):
        _assess(client)  # type: ignore[arg-type]

    assert len(client.calls) == 2


def test_learning_receipt_does_not_retry_transport_failure() -> None:
    client = ScriptedClient([RuntimeError("transport failed")])

    with pytest.raises(RuntimeError, match="transport failed"):
        _assess(client)  # type: ignore[arg-type]

    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "bad_score",
    ["0.8", True, None, math.nan, math.inf, -math.inf, -0.01, 101, 999],
)
@pytest.mark.parametrize("field", ["correctness", "depth", "evidence_quality"])
def test_learning_receipt_rejects_non_numeric_or_non_finite_scores(field: str, bad_score: Any) -> None:
    client = StubClient(_assessment_payload(**{field: bad_score}))

    with pytest.raises(StructuredOutputError, match="invalid learning_receipt payload"):
        _assess(client)

    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [(0.0, 0.0), (0.84, 0.84), (1.0, 1.0), (1.5, 0.3), (4, 0.8), (5, 1.0), (73, 0.73), (100, 1.0)],
)
@pytest.mark.parametrize("scorer", [score_transfer, score_teaching_explanation])
def test_teachback_scorers_normalise_supported_scales(
    raw_score: float,
    expected: float,
    scorer: Any,
) -> None:
    raw = _score_payload(score=raw_score)
    client = StubClient(raw)

    if scorer is score_transfer:
        result = scorer(client, "Question", "Answer", ["Criterion"])
    else:
        result = scorer(client, "Topic", "Explanation")

    assert result["score"] == expected
    assert len(result["accurate_points"]) == 5
    assert len(result["missing_points"]) == 5
    assert len(result["misconceptions"]) == 5
    assert set(result) == set(raw)
    assert "decimal fraction from 0.0 through 1.0" in client.calls[0][0][0]


@pytest.mark.parametrize(
    "bad_score",
    ["4", True, None, math.nan, math.inf, -math.inf, -0.01, 101, 999],
)
def test_teachback_scorers_reject_non_numeric_or_non_finite_scores(bad_score: Any) -> None:
    client = StubClient(_score_payload(score=bad_score))

    with pytest.raises(StructuredOutputError, match="invalid transfer_score payload"):
        score_transfer(client, "Question", "Answer", ["Criterion"])  # type: ignore[arg-type]

    assert len(client.calls) == 1


def test_all_model_generated_lists_are_bounded_to_schema_maxima() -> None:
    question_payload = _question_payload()
    questions = generate_questions(
        question_client := StubClient(question_payload),  # type: ignore[arg-type]
        {"name": "Epidemiology"},
        80,
        "transfer",
    )
    pair_payload = {
        "pre_question": "Pre",
        "post_question": "Post",
        "rubric": [f"Criterion {index}" for index in range(8)],
        "concept_invariant": "Same causal concept",
    }
    pair = generate_transfer_pair(StubClient(pair_payload), {"name": "Causality"})  # type: ignore[arg-type]

    assert len(questions["mcqs"]) == 3
    assert all(len(mcq["options"]) == 4 for mcq in questions["mcqs"])
    assert len(questions["open_rubric"]) == 5
    assert len(pair["rubric"]) == 5
    question_system, question_user = question_client.calls[0][0][:2]
    assert "never fill gaps" in question_system
    assert "sole evidence" in question_user
    assert "Do not introduce adjacent facts or entities" in question_user


@pytest.mark.parametrize(
    "case",
    [
        "one_mcq",
        "four_mcqs",
        "three_options",
        "five_options",
        "non_string_option",
        "negative_correct_index",
        "high_correct_index",
        "boolean_correct_index",
        "one_rubric_item",
        "six_rubric_items",
        "non_string_rubric_item",
    ],
)
def test_question_result_strictly_enforces_cardinality_types_and_index(case: str) -> None:
    payload = _question_payload()
    if case == "one_mcq":
        payload["mcqs"] = payload["mcqs"][:1]
    elif case == "four_mcqs":
        payload["mcqs"].append(deepcopy(payload["mcqs"][0]))
    elif case == "three_options":
        payload["mcqs"][0]["options"].pop()
    elif case == "five_options":
        payload["mcqs"][0]["options"].append("Extra")
    elif case == "non_string_option":
        payload["mcqs"][0]["options"][0] = 42
    elif case == "negative_correct_index":
        payload["mcqs"][0]["correct_index"] = -1
    elif case == "high_correct_index":
        payload["mcqs"][0]["correct_index"] = 4
    elif case == "boolean_correct_index":
        payload["mcqs"][0]["correct_index"] = True
    elif case == "one_rubric_item":
        payload["open_rubric"] = payload["open_rubric"][:1]
    elif case == "six_rubric_items":
        payload["open_rubric"].append("Extra")
    else:
        payload["open_rubric"][0] = 42

    client = StubClient(payload)
    with pytest.raises(StructuredOutputError, match="invalid retrieval_questions payload"):
        generate_questions(client, {"name": "Topic"}, 80, "transfer")  # type: ignore[arg-type]

    assert len(client.calls) == 1


@pytest.mark.parametrize("rubric_size", [0, 1, 2])
def test_teachback_pair_requires_at_least_three_rubric_items(rubric_size: int) -> None:
    payload = {
        "pre_question": "Pre",
        "post_question": "Post",
        "rubric": [f"Criterion {index}" for index in range(rubric_size)],
        "concept_invariant": "Same concept",
    }

    with pytest.raises(StructuredOutputError, match="invalid teachback_pair payload"):
        generate_transfer_pair(StubClient(payload), {"name": "Topic"})  # type: ignore[arg-type]


def test_question_generation_does_not_translate_unrelated_client_failure() -> None:
    client = ScriptedClient([RuntimeError("unexpected failure")])

    with pytest.raises(RuntimeError, match="unexpected failure"):
        generate_questions(client, {"name": "Topic"}, 80, "transfer")  # type: ignore[arg-type]

    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("scorer", "arguments", "schema_name"),
    [
        (score_transfer, ("Question", "Answer", ["Criterion"]), "transfer_score"),
        (score_teaching_explanation, ("Topic", "Explanation"), "teaching_quality"),
    ],
)
def test_score_structured_output_retries_once_with_terse_bounded_contract(
    scorer: Any,
    arguments: tuple[Any, ...],
    schema_name: str,
) -> None:
    client = ScriptedClient(
        [StructuredOutputError("truncated JSON"), _score_payload(score=4)]
    )

    result = scorer(client, *arguments)

    assert result["score"] == 0.8
    assert len(client.calls) == 2
    first_args, first_kwargs = client.calls[0]
    retry_args, retry_kwargs = client.calls[1]
    assert first_kwargs["schema_name"] == schema_name
    assert first_kwargs["max_tokens"] == 1100
    assert retry_kwargs["schema_name"] == f"{schema_name}_retry"
    assert retry_kwargs["max_tokens"] == 1600
    assert retry_args[1] == first_args[1]
    assert "at most 3 concise items in each list" in retry_args[0]
    assert "Complete the JSON object" in retry_args[0]


def test_second_structured_score_failure_propagates_after_one_retry() -> None:
    client = ScriptedClient(
        [StructuredOutputError("first failure"), StructuredOutputError("second failure")]
    )

    with pytest.raises(StructuredOutputError, match="second failure"):
        score_teaching_explanation(client, "Topic", "Explanation")  # type: ignore[arg-type]

    assert len(client.calls) == 2


def test_unrelated_score_failure_is_not_retried() -> None:
    client = ScriptedClient([RuntimeError("transport failed")])

    with pytest.raises(RuntimeError, match="transport failed"):
        score_transfer(client, "Question", "Answer", ["Criterion"])  # type: ignore[arg-type]

    assert len(client.calls) == 1
