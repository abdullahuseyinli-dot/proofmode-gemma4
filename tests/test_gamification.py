from __future__ import annotations

from dataclasses import fields
from datetime import date, datetime, timedelta

import pytest

from proofmode.gamification import (
    CommitmentEvent,
    CommitmentStatus,
    EvidenceEvent,
    EvidenceKind,
    IntegrityState,
    assess_integrity,
    bounded_difficulty,
    calculate_proof_score,
    daily_learning_bonus,
    difficulty_adjusted_score,
    diminishing_topic_credit,
    is_near_copy,
    near_copy_ratio,
    normalize_response,
    private_momentum,
    reliability_score,
    teachback_pair_credit_multiplier,
)


BASE = datetime(2026, 7, 20, 10, 0)


def _event(
    event_id: str,
    topic: str,
    kind: EvidenceKind | str,
    score: float,
    *,
    learner: str = "alice",
    hours: int = 0,
    delay: float | None = None,
    confidence: float | None = None,
    state: IntegrityState | str = IntegrityState.VERIFIED,
    response: str = "",
    prompt: str | None = None,
    difficulty: float = 0.5,
    **kwargs,
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=event_id,
        learner_id=learner,
        topic_id=topic,
        kind=kind,
        score=score,
        occurred_at=BASE + timedelta(hours=hours),
        difficulty=difficulty,
        confidence=confidence,
        delay_hours=(48 if EvidenceKind(kind) is EvidenceKind.RETENTION else 0)
        if delay is None
        else delay,
        prompt_id=prompt if prompt is not None else f"prompt-{event_id}",
        response_text=response,
        integrity_state=state,
        **kwargs,
    )


def _verified_core(*, confidence: float = 0.8) -> list[EvidenceEvent]:
    return [
        _event("r1", "vectors", "retention", 0.82, confidence=confidence),
        _event("r2", "gradients", "retention", 0.78, hours=1, confidence=confidence),
        _event("t1", "vectors", "transfer", 0.76, hours=2, confidence=confidence),
        _event("t2", "gradients", "transfer", 0.74, hours=3, confidence=confidence),
    ]


def test_difficulty_is_bounded_and_never_overrules_correctness() -> None:
    assert bounded_difficulty(-999) == 0.25
    assert bounded_difficulty(999) == 0.85
    assert difficulty_adjusted_score(0.70, 999) == pytest.approx(0.742)
    assert difficulty_adjusted_score(0.70, -999) == pytest.approx(0.67)
    assert difficulty_adjusted_score(0.60, 999) < difficulty_adjusted_score(0.80, -999)
    assert difficulty_adjusted_score(100, 100) == 1.0


def test_repetition_credit_diminishes_deterministically() -> None:
    weights = [diminishing_topic_credit(number) for number in range(1, 8)]
    assert weights[0] == 1.0
    assert all(left > right for left, right in zip(weights, weights[1:]))
    assert weights[-1] > 0


def test_proofscore_is_normalized_and_requires_real_evidence_coverage() -> None:
    result = calculate_proof_score(
        _verified_core(),
        curriculum_topics=["vectors", "gradients"],
    )

    assert 0 <= result.proof_score <= 100
    assert all(
        0 <= value <= 100
        for value in (
            result.knowledge,
            result.transfer_depth,
            result.calibration,
            result.topic_breadth,
        )
    )
    assert result.integrity_state is IntegrityState.VERIFIED
    assert result.leaderboard_eligible
    assert result.verified_topic_count == 2
    assert sum(result.component_weights.values()) == pytest.approx(1.0, abs=1e-5)


def test_repeating_one_topic_cannot_replace_breadth() -> None:
    one_topic_spam = [
        _event(
            f"spam-{index}",
            "vectors",
            "retention" if index % 2 == 0 else "transfer",
            0.95,
            hours=index,
            confidence=0.95,
        )
        for index in range(20)
    ]
    balanced = [
        _event(f"r-{topic}", topic, "retention", 0.90, hours=index, confidence=0.90)
        for index, topic in enumerate(("vectors", "gradients", "matrices", "eigenvalues"))
    ] + [
        _event(f"t-{topic}", topic, "transfer", 0.87, hours=10 + index, confidence=0.87)
        for index, topic in enumerate(("vectors", "gradients", "matrices", "eigenvalues"))
    ]

    spam_score = calculate_proof_score(one_topic_spam)
    balanced_score = calculate_proof_score(balanced)

    assert spam_score.integrity_state is IntegrityState.PROVISIONAL
    assert balanced_score.topic_breadth > spam_score.topic_breadth * 3
    assert balanced_score.proof_score > spam_score.proof_score


def test_near_copy_detection_is_unicode_and_punctuation_tolerant_but_conservative() -> None:
    original = (
        "Gradient descent changes each parameter opposite to the local gradient, "
        "using a learning rate to control the step size."
    )
    reformatted = (
        "  GRADIENT descent changes each parameter opposite to the local gradient—"
        "using a learning-rate to control the step size! "
    )
    unrelated = (
        "Cross validation rotates held-out folds so model selection can estimate "
        "generalisation on observations not used for fitting."
    )

    assert normalize_response("Ａ  Test!!!") == "a test"
    assert near_copy_ratio(original, reformatted) > 0.90
    assert is_near_copy(original, reformatted)
    assert not is_near_copy(original, unrelated)
    assert not is_near_copy("Paris", "Paris")  # short correct answers are not evidence of copying


def test_cross_learner_near_copy_is_held_for_fresh_transfer_not_labelled_cheating() -> None:
    response = (
        "A derivative gives the local rate of change and its sign shows whether "
        "the function rises or falls around the selected input value."
    )
    peer = _event(
        "peer-first",
        "calculus",
        "transfer",
        0.8,
        learner="bob",
        response=response,
        hours=0,
    )
    replay = _event(
        "alice-copy",
        "calculus",
        "transfer",
        0.95,
        learner="alice",
        response=response.upper() + "!",
        hours=1,
    )
    report = assess_integrity([replay], comparison_history=[peer], learner_id="alice")

    assert report.state is IntegrityState.HELD
    assert report.held_event_ids == ("alice-copy",)
    assert report.flags[0].code == "cross_learner_near_copy"
    assert report.verification_requests[0].held_event_id == "alice-copy"
    combined_text = " ".join(
        [report.flags[0].explanation, report.verification_requests[0].instruction]
    ).lower()
    assert "fresh" in combined_text
    assert "cheat" not in combined_text
    assert "ai-generated" not in combined_text


def test_fresh_isomorphic_transfer_releases_hold_without_reusing_questionable_event() -> None:
    shared = (
        "Momentum is conserved when the system has no net external impulse over "
        "the interval, so the vector total before equals the vector total after."
    )
    peer = _event("peer", "momentum", "transfer", 0.9, learner="bob", response=shared)
    copied = _event(
        "copy",
        "momentum",
        "transfer",
        1.0,
        learner="alice",
        response=shared,
        hours=1,
    )
    verification = _event(
        "fresh",
        "momentum",
        "transfer",
        0.72,
        learner="alice",
        response=(
            "Treat both carts as one system. Their collision forces are internal, "
            "so calculate the final shared velocity from total initial momentum."
        ),
        prompt="new-isomorphic-prompt",
        verification_for="copy",
        hours=2,
    )
    own = [*_verified_core(), copied, verification]
    result = calculate_proof_score(own, learner_id="alice", comparison_history=[peer])

    assert result.integrity_state is IntegrityState.VERIFIED
    assert not result.held_event_ids
    assert result.leaderboard_eligible
    # Four core events plus the fresh verification; the copied event stays excluded.
    assert result.evidence_count == 5


def test_provisional_high_scores_are_not_public_leaderboard_scores() -> None:
    provisional = [
        _event(
            f"p-{index}",
            "one-topic",
            "retention",
            1.0,
            state="provisional",
            hours=index * 24,
            delay=48,
            confidence=1.0,
        )
        for index in range(12)
    ]
    result = calculate_proof_score(provisional)

    assert result.integrity_state is IntegrityState.PROVISIONAL
    assert not result.leaderboard_eligible
    assert result.provisional_reasons


def test_failed_delayed_retrieval_does_not_satisfy_public_delay_gate() -> None:
    events = [
        _event("r-a", "a", "retention", 0.59, delay=48, confidence=0.5),
        _event("r-b", "b", "retention", 0.20, delay=48, confidence=0.3, hours=1),
        _event("t-a", "a", "transfer", 0.80, confidence=0.8, hours=2),
        _event("t-b", "b", "transfer", 0.80, confidence=0.8, hours=3),
    ]
    result = calculate_proof_score(events, curriculum_topics=["a", "b"])

    assert not result.leaderboard_eligible
    assert any("delay" in reason.lower() for reason in result.provisional_reasons)


def test_replayed_retrieval_prompt_cannot_supply_two_delayed_checks() -> None:
    events = [
        _event("r-a-1", "a", "retention", 0.90, delay=48, prompt="same-mcq", confidence=0.9),
        _event("r-a-2", "a", "retention", 0.95, delay=72, prompt="same-mcq", confidence=0.9, hours=24),
        _event("t-a", "a", "transfer", 0.80, confidence=0.8, hours=25),
        _event("t-b", "b", "transfer", 0.80, confidence=0.8, hours=26),
    ]
    result = calculate_proof_score(events, curriculum_topics=["a", "b"])

    assert not result.leaderboard_eligible
    assert result.evidence_count == 3
    assert any("delay" in reason.lower() for reason in result.provisional_reasons)


def test_calibration_rewards_honest_pre_grading_confidence_not_confident_style() -> None:
    outcomes = (0.2, 0.85, 0.35, 0.9)
    kinds = ("retention", "retention", "transfer", "transfer")
    topics = ("a", "b", "a", "b")
    calibrated = [
        _event(
            f"c-{index}",
            topics[index],
            kinds[index],
            outcome,
            hours=index,
            confidence=outcome,
        )
        for index, outcome in enumerate(outcomes)
    ]
    overconfident = [
        _event(
            f"o-{index}",
            topics[index],
            kinds[index],
            outcome,
            hours=index,
            confidence=1.0,
        )
        for index, outcome in enumerate(outcomes)
    ]

    honest = calculate_proof_score(calibrated, curriculum_topics=["a", "b"])
    inflated = calculate_proof_score(overconfident, curriculum_topics=["a", "b"])
    assert honest.calibration > inflated.calibration
    assert honest.proof_score > inflated.proof_score


def test_teachback_pair_and_reciprocity_credit_are_capped() -> None:
    first = _event(
        "teach-1",
        "vectors",
        "teachback",
        0.8,
        partner_id="bob",
        pre_score=0.2,
        post_score=0.8,
        teaching_quality=0.9,
    )
    second = _event(
        "teach-2",
        "vectors",
        "teachback",
        0.8,
        partner_id="bob",
        pre_score=0.2,
        post_score=0.8,
        teaching_quality=0.9,
        hours=24,
    )
    third = _event(
        "teach-3",
        "vectors",
        "teachback",
        0.8,
        partner_id="bob",
        pre_score=0.2,
        post_score=0.8,
        teaching_quality=0.9,
        hours=48,
    )
    reciprocal = _event(
        "teach-reverse",
        "vectors",
        "teachback",
        0.8,
        learner="bob",
        partner_id="alice",
        pre_score=0.2,
        post_score=0.8,
        teaching_quality=0.9,
        hours=12,
    )

    assert teachback_pair_credit_multiplier(first, [first]) == 1.0
    assert teachback_pair_credit_multiplier(second, [first, second]) == 0.5
    assert teachback_pair_credit_multiplier(third, [first, second, third]) == 0.0
    assert teachback_pair_credit_multiplier(reciprocal, [first, reciprocal]) == 0.25

    base = _verified_core()
    two = calculate_proof_score([*base, first, second])
    many = calculate_proof_score([*base, first, second, third])
    assert many.teaching_impact == two.teaching_impact
    assert many.proof_score == two.proof_score


def test_reliability_requires_proof_and_checkbox_spam_cannot_inflate_it() -> None:
    proof = _event("proof", "vectors", "retention", 0.8)
    one = CommitmentEvent("one", "alice", BASE, "completed", proof_event_id="proof")
    duplicate_checkboxes = [
        CommitmentEvent(
            f"box-{index}",
            "alice",
            BASE + timedelta(minutes=index),
            "completed",
            proof_event_id="proof",
        )
        for index in range(20)
    ]
    unproved = CommitmentEvent("unproved", "alice", BASE, "completed", proof_event_id=None)

    assert reliability_score([one], [proof], now=BASE) == 100.0
    assert reliability_score(duplicate_checkboxes, [proof], now=BASE) == 100.0
    assert reliability_score([unproved], [proof], now=BASE) == 0.0

    missed = CommitmentEvent("missed", "alice", BASE, "missed")
    honest = reliability_score([one, missed], [proof], now=BASE)
    spammed = reliability_score([one, missed, *duplicate_checkboxes], [proof], now=BASE)
    assert honest == spammed == 50.0


def test_momentum_is_private_and_proof_backed() -> None:
    evidence = [
        _event("p1", "a", "retention", 0.8),
        _event("p2", "b", "retention", 0.8, hours=24),
    ]
    commitments = [
        CommitmentEvent("c1", "alice", BASE, "completed", "p1"),
        CommitmentEvent("c2", "alice", BASE + timedelta(days=1), "recovered", "p2"),
        CommitmentEvent("c3", "alice", BASE + timedelta(days=2), "missed"),
    ]
    snapshot = private_momentum(commitments, evidence, now=BASE + timedelta(days=2))

    assert snapshot.public is False
    assert snapshot.best_streak == 2
    assert snapshot.current_streak == 0
    assert snapshot.successful_days == 2
    assert "never changes ProofScore" in snapshot.explanation


def test_daily_bonus_is_capped_but_same_day_mastery_evidence_is_not() -> None:
    events = [
        _event(
            f"topic-{index}",
            f"topic-{index}",
            "transfer",
            0.8,
            hours=index,
            confidence=0.8,
        )
        for index in range(10)
    ]
    bonus = daily_learning_bonus(events, BASE.date(), cap=12)
    first_two = calculate_proof_score(events[:2])
    all_topics = calculate_proof_score(events)

    assert bonus.points == 12
    assert bonus.affects_proof_score is False
    assert bonus.affects_mastery is False
    # Bonus hits its cap, while additional genuine topic evidence still increases breadth.
    assert all_topics.topic_breadth > first_two.topic_breadth


def test_api_has_no_activity_or_ai_authorship_point_inputs_and_is_explainable() -> None:
    evidence_fields = {item.name for item in fields(EvidenceEvent)}
    assert evidence_fields.isdisjoint(
        {"study_minutes", "message_count", "notes_length", "ai_probability", "word_count"}
    )
    with pytest.raises(TypeError):
        EvidenceEvent(  # type: ignore[call-arg]
            "e",
            "alice",
            "topic",
            "retention",
            0.8,
            BASE,
            study_minutes=600,
        )

    result = calculate_proof_score(_verified_core())
    explanation = " ".join(result.explanations).lower()
    assert "study time" in explanation
    assert "note length" in explanation
    assert "presumed ai authorship" in explanation
    assert result.as_dict()["integrity_state"] == "verified"
