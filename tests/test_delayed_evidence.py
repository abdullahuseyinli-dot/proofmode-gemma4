from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from proofmode.database import Database
from proofmode.delayed_evidence import (
    delayed_gap_hours,
    find_due_delayed_block,
    latest_exposure,
    qualifies_as_delayed,
    retrieval_prompt_fingerprint,
)
from proofmode.services.calendar_service import StudyBlock


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _block(*, topic: str = "Vectors", hours: int = -1, task_id: str = "delayed-vectors") -> StudyBlock:
    start = NOW + timedelta(hours=hours)
    return StudyBlock(
        uid=f"block-{topic}-{hours}",
        topic=topic,
        title=topic,
        start=start,
        end=start + timedelta(minutes=20),
        task_id=task_id,
    )


def test_only_due_matching_unconsumed_delayed_block_is_selected() -> None:
    due = _block()
    blocks = [
        _block(topic="Matrices"),
        _block(hours=1),
        _block(task_id="ordinary-task"),
        due,
    ]
    assert find_due_delayed_block(blocks, "vectors", NOW) is due
    assert find_due_delayed_block(blocks, "vectors", NOW, consumed_block_ids=[due.uid]) is None


def test_latest_exposure_resets_spacing_and_exact_boundary_qualifies() -> None:
    old = NOW - timedelta(hours=30)
    recent = NOW - timedelta(hours=2)
    assert latest_exposure([old, recent], before=NOW) == recent
    assert delayed_gap_hours(recent, NOW) == 2
    assert delayed_gap_hours(NOW - timedelta(hours=20), NOW) == 20
    assert qualifies_as_delayed(
        has_due_block=True,
        delay_hours=20,
        question_source="gemma",
        assessment_source="gemma",
        retrieval_score=0.6,
        first_submission=True,
        fresh_prompt=True,
    )


def test_fallback_low_score_or_replay_never_becomes_delayed_evidence() -> None:
    base = {
        "has_due_block": True,
        "delay_hours": 25,
        "question_source": "gemma",
        "assessment_source": "gemma",
        "retrieval_score": 0.8,
        "first_submission": True,
        "fresh_prompt": True,
    }
    for change in (
        {"question_source": "fallback"},
        {"assessment_source": "fallback"},
        {"retrieval_score": 0.59},
        {"first_submission": False},
        {"has_due_block": False},
        {"fresh_prompt": False},
    ):
        assert not qualifies_as_delayed(**{**base, **change})


def test_retrieval_fingerprint_ignores_transfer_prompt_but_includes_mcq_options() -> None:
    first = {
        "mcqs": [
            {"question": "What is leakage?", "options": ["A", "B"], "correct_index": 0},
            {"question": "Where should fitting happen?", "options": ["Inside", "Before"], "correct_index": 0},
        ],
        "open_question": "Apply it to a pipeline.",
    }
    disguised = {**first, "open_question": "A completely different transfer question."}
    permuted = {
        **first,
        "mcqs": [
            {"question": "  WHERE should fitting happen? ", "options": ["Before", "Inside"], "correct_index": 1},
            {"question": "what is leakage?", "options": ["B", "A"], "correct_index": 1},
        ],
    }
    changed_retrieval = {
        **first,
        "mcqs": [
            {"question": "What is leakage?", "options": ["A", "C"], "correct_index": 0},
            first["mcqs"][1],
        ],
    }
    assert retrieval_prompt_fingerprint(first) == retrieval_prompt_fingerprint(disguised)
    assert retrieval_prompt_fingerprint(first) == retrieval_prompt_fingerprint(permuted)
    assert retrieval_prompt_fingerprint(first) != retrieval_prompt_fingerprint(changed_retrieval)


def test_persistent_issuance_allows_only_one_block_and_one_submission(tmp_path) -> None:
    database = Database(tmp_path / "delayed.db")
    values = (
        "issue-1",
        "block-1",
        "alice",
        "Vectors",
        NOW.isoformat(),
        (NOW - timedelta(hours=25)).isoformat(),
        25.0,
        "prompt-1",
        "gemma",
        "{}",
    )
    database.execute(
        "INSERT INTO question_issuances (issuance_id, block_uid, learner_name, topic, issued_at, anchor_at, delay_hours, prompt_id, question_source, questions_payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        values,
    )
    with pytest.raises(sqlite3.IntegrityError):
        database.execute(
            "INSERT INTO question_issuances (issuance_id, block_uid, learner_name, topic, issued_at, anchor_at, delay_hours, prompt_id, question_source, questions_payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("issue-2", *values[1:]),
        )

    with database.connect() as connection:
        first = connection.execute(
            "UPDATE question_issuances SET submitted_at = ? WHERE issuance_id = ? AND submitted_at IS NULL",
            ((NOW + timedelta(minutes=10)).isoformat(), "issue-1"),
        ).rowcount
    with database.connect() as connection:
        replay = connection.execute(
            "UPDATE question_issuances SET submitted_at = ? WHERE issuance_id = ? AND submitted_at IS NULL",
            ((NOW + timedelta(minutes=11)).isoformat(), "issue-1"),
        ).rowcount
    assert first == 1
    assert replay == 0
