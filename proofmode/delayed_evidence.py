"""Pure rules for issuing and accepting delayed retrieval challenges.

A calendar block is only an invitation to produce evidence.  It never earns
credit itself.  Delayed status comes from the gap between the latest exposure
and the moment a fresh, server-recorded challenge is revealed.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Any, Iterable, Mapping


MIN_DELAY_HOURS = 20.0
MIN_PASS_SCORE = 0.60


def retrieval_prompt_fingerprint(questions: Mapping[str, Any]) -> str:
    """Fingerprint only the closed-book retrieval challenge.

    The transfer prompt is deliberately excluded so changing it cannot disguise
    reused MCQs. Options are included because they are part of what the learner
    saw and could memorise.
    """

    def canonical_text(value: Any) -> str:
        return " ".join(str(value).split()).casefold()

    payload = [
        {
            "question": canonical_text(item.get("question", "")),
            "options": sorted(canonical_text(option) for option in item.get("options", [])),
        }
        for item in questions.get("mcqs", [])
    ]
    payload.sort(key=lambda item: (item["question"], item["options"]))
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def delayed_gap_hours(anchor: datetime | None, issued_at: datetime) -> float:
    if anchor is None:
        return 0.0
    return max(0.0, (issued_at - anchor).total_seconds() / 3600.0)


def latest_exposure(
    moments: Iterable[datetime],
    *,
    before: datetime,
) -> datetime | None:
    eligible = [moment for moment in moments if moment < before]
    return max(eligible) if eligible else None


def find_due_delayed_block(
    blocks: Iterable[Any],
    topic: str,
    now: datetime,
    *,
    consumed_block_ids: Iterable[str] = (),
) -> Any | None:
    consumed = set(consumed_block_ids)
    candidates = [
        block
        for block in blocks
        if str(getattr(block, "task_id", "")).startswith("delayed-")
        and str(getattr(block, "topic", "")).casefold() == topic.casefold()
        and str(getattr(block, "uid", "")) not in consumed
        and getattr(block, "start") <= now
    ]
    return min(candidates, key=lambda block: (block.start, block.uid)) if candidates else None


def qualifies_as_delayed(
    *,
    has_due_block: bool,
    delay_hours: float,
    question_source: str,
    assessment_source: str,
    retrieval_score: float,
    first_submission: bool,
    fresh_prompt: bool,
) -> bool:
    return bool(
        has_due_block
        and delay_hours >= MIN_DELAY_HOURS
        and question_source == "gemma"
        and assessment_source == "gemma"
        and retrieval_score >= MIN_PASS_SCORE
        and first_submission
        and fresh_prompt
    )


__all__ = [
    "MIN_DELAY_HOURS",
    "MIN_PASS_SCORE",
    "delayed_gap_hours",
    "find_due_delayed_block",
    "latest_exposure",
    "qualifies_as_delayed",
    "retrieval_prompt_fingerprint",
]
