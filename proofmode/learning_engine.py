from __future__ import annotations

import math
from datetime import datetime
from typing import Any


DEPTH_MULTIPLIER = {
    "core": 0.9,
    "apply": 1.0,
    "transfer": 1.18,
    "teach_research": 1.35,
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def topic_priority(topic: dict[str, Any], exam_date: datetime, depth_mode: str) -> float:
    """Transparent priority score; Gemma proposes factors, deterministic code ranks them."""
    days_left = max(1.0, (exam_date - datetime.now(exam_date.tzinfo)).total_seconds() / 86400)
    exam_weight = clamp(topic.get("exam_weight", 0.5), 0.05, 1.0)
    difficulty = clamp(topic.get("difficulty", 0.5), 0.1, 1.0)
    mastery = clamp(topic.get("mastery", 0.2))
    confidence = clamp(topic.get("confidence", 0.3))
    centrality = clamp(topic.get("prerequisite_centrality", 0.5), 0.1, 1.0)
    effort = max(15.0, float(topic.get("estimated_minutes", 45)))
    mastery_gap = max(0.08, 1.0 - mastery)
    forgetting_risk = clamp(0.35 + 0.45 * (1 - confidence) + 0.2 * mastery_gap, 0.2, 1.0)
    urgency = 1.0 + 14.0 / (days_left + 3.0)
    learnability = 1.15 - 0.35 * difficulty
    raw = (
        exam_weight
        * mastery_gap
        * forgetting_risk
        * (0.6 + centrality)
        * DEPTH_MULTIPLIER.get(depth_mode, 1.0)
        * urgency
        * learnability
        / math.sqrt(effort / 30.0)
    )
    return round(raw * 100, 2)


def update_mastery(
    previous_mastery: float,
    correctness: float,
    depth_score: float,
    evidence_quality: float,
    confidence: float,
) -> dict[str, float]:
    correctness = clamp(correctness)
    depth_score = clamp(depth_score)
    evidence_quality = clamp(evidence_quality)
    confidence = clamp(confidence)
    observed = 0.58 * correctness + 0.27 * depth_score + 0.15 * evidence_quality
    learning_rate = 0.42 if observed >= previous_mastery else 0.25
    mastery = clamp(previous_mastery + learning_rate * (observed - previous_mastery))
    calibration = clamp(1.0 - abs(confidence - correctness))
    return {
        "mastery": round(mastery, 3),
        "knowledge": round(correctness, 3),
        "depth": round(depth_score, 3),
        "evidence_quality": round(evidence_quality, 3),
        "calibration": round(calibration, 3),
    }


def ambition_gap(target_mark: float, available_minutes: int, required_minutes: int) -> dict[str, Any]:
    gap = required_minutes - available_minutes
    if gap < 0:
        status = "realistic"
        message = f"The plan has about {abs(gap)} minutes of buffer."
    elif gap == 0:
        status = "tight"
        message = "The plan is fully allocated with no buffer."
    elif gap <= 120:
        status = "tight"
        message = f"The current target needs roughly {gap} more focused minutes."
    else:
        status = "overcommitted"
        message = f"The target and calendar differ by about {gap} minutes; ProofMode will prioritise essentials."
    return {"status": status, "gap_minutes": gap, "target_mark": target_mark, "message": message}
