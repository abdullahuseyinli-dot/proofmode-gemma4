from __future__ import annotations

from datetime import datetime, timedelta, timezone

from proofmode.database import depth_for_mark
from proofmode.learning_engine import topic_priority, update_mastery
from proofmode.services.intervention_service import fallback_intervention
from proofmode.services.teachback_service import teaching_impact


BASE_TOPIC = {
    "name": "Backpropagation",
    "exam_weight": 0.8,
    "difficulty": 0.7,
    "mastery": 0.3,
    "confidence": 0.4,
    "prerequisite_centrality": 0.8,
    "estimated_minutes": 60,
    "prerequisites": ["Chain rule"],
}


def test_priority_increases_as_exam_approaches() -> None:
    now = datetime.now(timezone.utc)
    soon = topic_priority(BASE_TOPIC, now + timedelta(days=3), "transfer")
    later = topic_priority(BASE_TOPIC, now + timedelta(days=30), "transfer")
    assert soon > later


def test_mastery_rewards_correct_deep_evidence() -> None:
    result = update_mastery(0.3, correctness=0.9, depth_score=0.8, evidence_quality=0.7, confidence=0.85)
    assert result["mastery"] > 0.3
    assert result["calibration"] >= 0.9


def test_overconfidence_reduces_calibration_not_knowledge() -> None:
    result = update_mastery(0.4, correctness=0.2, depth_score=0.3, evidence_quality=0.5, confidence=0.95)
    assert result["knowledge"] == 0.2
    assert result["calibration"] < 0.3


def test_target_mark_derives_depth_without_more_questions() -> None:
    assert depth_for_mark(55) == "core"
    assert depth_for_mark(68) == "apply"
    assert depth_for_mark(75) == "transfer"
    assert depth_for_mark(90) == "teach_research"


def test_confusion_selects_prerequisite_rescue() -> None:
    result = fallback_intervention(BASE_TOPIC, "confused")
    assert result["name"] == "start_prerequisite"
    assert result["arguments"]["prerequisite"] == "Chain rule"


def test_teaching_impact_never_rewards_negative_gain() -> None:
    result = teaching_impact(0.8, 0.5, 1.0)
    assert result["impact"] == 0
    assert result["gain"] == -30


def test_teaching_impact_requires_both_gain_and_quality() -> None:
    weak = teaching_impact(0.3, 0.7, 0.0)["impact"]
    strong = teaching_impact(0.3, 0.7, 1.0)["impact"]
    assert 0 < weak < strong

