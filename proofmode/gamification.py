"""Fair, evidence-based gamification for ProofMode.

``ProofScore`` is a normalized estimate of demonstrated learning, not a total of
actions performed.  The module deliberately has no inputs for study minutes,
message counts, note length, or presumed AI authorship.  Cosmetic daily bonus
points and private momentum are calculated separately and never alter mastery or
the public score.

All values supplied by an assessor (including Gemma) are treated as bounded
observations.  The public score and integrity state are derived deterministically
here so that the rules remain inspectable and testable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


MIN_DIFFICULTY = 0.25
MAX_DIFFICULTY = 0.85
NEAR_COPY_THRESHOLD = 0.90
MIN_COPY_CHARS = 48
MIN_COPY_WORDS = 8

# These are product-policy weights rather than learned parameters.  Optional
# components are omitted and the remaining weights are normalized, so a learner
# is not disadvantaged merely because peer teaching is unavailable to them.
PROOF_SCORE_WEIGHTS: Mapping[str, float] = {
    "knowledge": 0.32,
    "transfer_depth": 0.24,
    "calibration": 0.10,
    # Breadth is intentionally material: near-perfect drilling of one topic must
    # not outrank strong evidence distributed across the curriculum.
    "topic_breadth": 0.20,
    "teaching_impact": 0.07,
    "reliability": 0.07,
}


class EvidenceKind(str, Enum):
    """Kinds of evidence that can affect a ProofScore component."""

    RETENTION = "retention"
    TRANSFER = "transfer"
    TEACHBACK = "teachback"


class IntegrityState(str, Enum):
    """Whether a score is ready for a public comparison."""

    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    HELD = "held"


class CommitmentStatus(str, Enum):
    """Outcome of a planned learning contract."""

    COMPLETED = "completed"
    RECOVERED = "recovered"
    MISSED = "missed"
    EXCUSED = "excused"


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    """Return a finite value inside an inclusive range."""

    return max(low, min(high, _finite_number(value, low)))


def bounded_difficulty(difficulty: Any) -> float:
    """Bound self/LLM-reported difficulty so it cannot manufacture a huge bonus."""

    return clamp(difficulty, MIN_DIFFICULTY, MAX_DIFFICULTY)


def difficulty_adjusted_score(score: Any, difficulty: Any) -> float:
    """Apply a deliberately small, bounded difficulty adjustment.

    Difficulty changes an observation by at most 4.2 percentage points.  It is a
    tie-breaker for comparable demonstrations, never a substitute for correctness.
    """

    raw = clamp(score)
    adjustment = (bounded_difficulty(difficulty) - 0.50) * 0.12
    return clamp(raw + adjustment)


def diminishing_topic_credit(attempt_number: int) -> float:
    """Credit weight for the Nth observation of the same component/topic.

    Attempts are numbered from one.  Scores are eventually aggregated once per
    topic, so generating more attempts cannot create more topic breadth.
    """

    attempt = max(1, int(attempt_number))
    return 1.0 / math.sqrt(attempt)


def _utc_timestamp(moment: datetime) -> float:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).timestamp()


def _calendar_day(moment: datetime) -> date:
    return moment.date()


@dataclass(frozen=True)
class EvidenceEvent:
    """One assessable learning observation.

    ``score``, ``confidence``, ``pre_score``, ``post_score`` and
    ``teaching_quality`` use a 0..1 scale.  ``learner_id`` identifies the person
    whose ProofScore receives the evidence; for a teach-back it is the teacher and
    ``partner_id`` is the learner who received the explanation.

    A held or auto-detected duplicate does not count.  A fresh, independently
    worded transfer event may set ``verification_for`` to the held event id.  This
    resolves the comparison hold without resurrecting the questionable evidence.
    """

    event_id: str
    learner_id: str
    topic_id: str
    kind: EvidenceKind | str
    score: float
    occurred_at: datetime
    difficulty: float = 0.50
    confidence: float | None = None
    delay_hours: float = 0.0
    prompt_id: str = ""
    response_text: str = field(default="", repr=False)
    integrity_state: IntegrityState | str = IntegrityState.VERIFIED
    partner_id: str | None = None
    pre_score: float | None = None
    post_score: float | None = None
    teaching_quality: float | None = None
    verification_for: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EvidenceKind(self.kind))
        object.__setattr__(self, "integrity_state", IntegrityState(self.integrity_state))
        if not self.event_id.strip():
            raise ValueError("event_id is required")
        if not self.learner_id.strip():
            raise ValueError("learner_id is required")
        if not self.topic_id.strip():
            raise ValueError("topic_id is required")
        if not isinstance(self.occurred_at, datetime):
            raise TypeError("occurred_at must be a datetime")


@dataclass(frozen=True)
class CommitmentEvent:
    """A scheduled learning promise used for reliability and private momentum.

    Completed/recovered promises count positively only when ``proof_event_id``
    links to verified learning evidence.  This prevents check-box or micro-session
    spam from becoming a score source.
    """

    commitment_id: str
    learner_id: str
    scheduled_for: datetime
    status: CommitmentStatus | str
    proof_event_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CommitmentStatus(self.status))
        if not self.commitment_id.strip():
            raise ValueError("commitment_id is required")
        if not self.learner_id.strip():
            raise ValueError("learner_id is required")
        if not isinstance(self.scheduled_for, datetime):
            raise TypeError("scheduled_for must be a datetime")


@dataclass(frozen=True)
class IntegrityFlag:
    event_id: str
    code: str
    explanation: str
    related_event_id: str | None = None


@dataclass(frozen=True)
class VerificationRequest:
    held_event_id: str
    topic_id: str
    reason: str
    instruction: str = (
        "Answer a fresh isomorphic transfer question without seeing the earlier answer."
    )


@dataclass(frozen=True)
class IntegrityReport:
    state: IntegrityState
    held_event_ids: tuple[str, ...] = ()
    resolved_event_ids: tuple[str, ...] = ()
    flags: tuple[IntegrityFlag, ...] = ()
    verification_requests: tuple[VerificationRequest, ...] = ()


@dataclass(frozen=True)
class ScoreBreakdown:
    """Inspectable ProofScore result.

    Component values are 0..100.  Optional components are ``None`` until relevant
    evidence exists.  ``proof_score`` remains visible while provisional/held for
    formative feedback, but only a verified result is leaderboard eligible.
    """

    proof_score: float
    integrity_state: IntegrityState
    leaderboard_eligible: bool
    knowledge: float
    transfer_depth: float
    calibration: float
    topic_breadth: float
    teaching_impact: float | None
    reliability: float | None
    component_weights: Mapping[str, float]
    evidence_count: int
    verified_topic_count: int
    held_event_ids: tuple[str, ...] = ()
    provisional_reasons: tuple[str, ...] = ()
    verification_requests: tuple[VerificationRequest, ...] = ()
    explanations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["integrity_state"] = self.integrity_state.value
        return data


@dataclass(frozen=True)
class MomentumSnapshot:
    """Private behaviour support; never part of a public ProofScore."""

    momentum: float
    current_streak: int
    best_streak: int
    planned_days: int
    successful_days: int
    public: bool = False
    explanation: str = (
        "Private momentum uses proof-backed planned days; it never changes ProofScore."
    )


@dataclass(frozen=True)
class DailyBonus:
    """Cosmetic, capped reinforcement that cannot alter proficiency."""

    day: date
    points: int
    cap: int
    reasons: tuple[str, ...] = ()
    affects_proof_score: bool = False
    affects_mastery: bool = False


_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def normalize_response(text: str) -> str:
    """Normalize Unicode, case, punctuation and whitespace for copy comparison."""

    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    normalized = _NON_WORD_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def near_copy_ratio(first: str, second: str) -> float:
    """Return an order-sensitive similarity ratio using only the standard library."""

    left = normalize_response(first)
    right = normalize_response(second)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def is_near_copy(
    first: str,
    second: str,
    *,
    threshold: float = NEAR_COPY_THRESHOLD,
    min_chars: int = MIN_COPY_CHARS,
    min_words: int = MIN_COPY_WORDS,
) -> bool:
    """Conservatively identify a substantial duplicate/near-copy response.

    Short answers are ignored because independent correct answers are naturally
    similar.  This is copy-evidence comparison, not AI-authorship detection.
    """

    left = normalize_response(first)
    right = normalize_response(second)
    if min(len(left), len(right)) < max(1, min_chars):
        return False
    if min(len(left.split()), len(right.split())) < max(1, min_words):
        return False
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= clamp(
        threshold, 0.0, 1.0
    )


def _unique_events(events: Iterable[EvidenceEvent]) -> list[EvidenceEvent]:
    """Remove the same object supplied twice while preserving true id collisions."""

    output: list[EvidenceEvent] = []
    seen_objects: set[int] = set()
    for event in events:
        if id(event) in seen_objects:
            continue
        seen_objects.add(id(event))
        output.append(event)
    return output


def _fresh_verification_resolves(
    verification: EvidenceEvent,
    original: EvidenceEvent,
) -> bool:
    return bool(
        verification.kind is EvidenceKind.TRANSFER
        and verification.integrity_state is IntegrityState.VERIFIED
        and verification.verification_for == original.event_id
        and verification.event_id != original.event_id
        and verification.prompt_id
        and verification.prompt_id != original.prompt_id
        and not is_near_copy(verification.response_text, original.response_text)
    )


def assess_integrity(
    events: Iterable[EvidenceEvent],
    *,
    comparison_history: Iterable[EvidenceEvent] = (),
    learner_id: str | None = None,
) -> IntegrityReport:
    """Hold duplicated evidence for a fresh transfer check, without accusation.

    Text style, fluency and presumed AI authorship are intentionally not inspected.
    Cross-learner near-copies and rapid replay of the same learner response are
    treated as ambiguous evidence: the later event is held, not punished.
    """

    target = list(events)
    if learner_id is not None:
        target = [event for event in target if event.learner_id == learner_id]
    target_ids = {id(event) for event in target}
    universe = _unique_events([*target, *comparison_history])
    ordered = sorted(universe, key=lambda item: (_utc_timestamp(item.occurred_at), item.event_id))
    flags_by_event: dict[str, IntegrityFlag] = {}

    # A duplicated event id can never produce two units of evidence.
    first_by_id: dict[str, EvidenceEvent] = {}
    for event in ordered:
        previous = first_by_id.get(event.event_id)
        if previous is not None and id(event) in target_ids:
            flags_by_event[event.event_id] = IntegrityFlag(
                event_id=event.event_id,
                code="duplicate_event_id",
                related_event_id=previous.event_id,
                explanation="This event id was already recorded; fresh transfer evidence is requested.",
            )
        else:
            first_by_id[event.event_id] = event

    # Compare only meaningful answer text.  Same-learner answers are held only for
    # an identical prompt or an implausibly rapid replay on the same topic.  This
    # avoids treating legitimate delayed retrieval as misconduct.
    for index, current in enumerate(ordered):
        if id(current) not in target_ids or current.event_id in flags_by_event:
            continue
        if not current.response_text.strip():
            continue
        for previous in ordered[:index]:
            if previous.topic_id.casefold() != current.topic_id.casefold():
                continue
            if not is_near_copy(previous.response_text, current.response_text):
                continue
            same_learner = previous.learner_id == current.learner_id
            same_prompt = bool(current.prompt_id and current.prompt_id == previous.prompt_id)
            seconds_apart = _utc_timestamp(current.occurred_at) - _utc_timestamp(previous.occurred_at)
            rapid_replay = same_learner and 0 <= seconds_apart <= 10 * 60
            cross_learner_copy = not same_learner
            if not (same_prompt or rapid_replay or cross_learner_copy):
                continue
            code = "cross_learner_near_copy" if cross_learner_copy else "replayed_response"
            flags_by_event[current.event_id] = IntegrityFlag(
                event_id=current.event_id,
                code=code,
                related_event_id=previous.event_id,
                explanation=(
                    "This response substantially overlaps earlier evidence. "
                    "It is held only until a fresh transfer check is completed."
                ),
            )
            break

    for event in target:
        if event.integrity_state is IntegrityState.HELD:
            flags_by_event.setdefault(
                event.event_id,
                IntegrityFlag(
                    event_id=event.event_id,
                    code="explicit_hold",
                    explanation="This evidence is awaiting an independent transfer check.",
                ),
            )

    by_id = {event.event_id: event for event in universe}
    resolved: set[str] = set()
    for verification in universe:
        original_id = verification.verification_for
        if not original_id or original_id not in flags_by_event:
            continue
        original = by_id.get(original_id)
        if original and _fresh_verification_resolves(verification, original):
            resolved.add(original_id)

    unresolved_flags = tuple(
        flag
        for event_id, flag in sorted(flags_by_event.items())
        if event_id not in resolved
    )
    held_ids = tuple(flag.event_id for flag in unresolved_flags)
    requests = tuple(
        VerificationRequest(
            held_event_id=flag.event_id,
            topic_id=by_id[flag.event_id].topic_id if flag.event_id in by_id else "unknown",
            reason=flag.explanation,
        )
        for flag in unresolved_flags
    )
    has_provisional = any(
        event.integrity_state is IntegrityState.PROVISIONAL for event in target
    )
    state = (
        IntegrityState.HELD
        if held_ids
        else IntegrityState.PROVISIONAL
        if has_provisional
        else IntegrityState.VERIFIED
    )
    return IntegrityReport(
        state=state,
        held_event_ids=held_ids,
        resolved_event_ids=tuple(sorted(resolved)),
        flags=unresolved_flags,
        verification_requests=requests,
    )


def _event_weight(event: EvidenceEvent, attempt_number: int) -> float:
    status_weight = 0.35 if event.integrity_state is IntegrityState.PROVISIONAL else 1.0
    return status_weight * diminishing_topic_credit(attempt_number)


def _retention_delay_weight(delay_hours: float) -> float:
    # Immediate retrieval is useful but provides weaker evidence about retention.
    delay = max(0.0, _finite_number(delay_hours))
    return 0.60 + 0.40 * min(1.0, math.log1p(delay) / math.log1p(168.0))


def _topic_estimates(
    events: Sequence[EvidenceEvent],
    kinds: set[EvidenceKind],
    *,
    retention_delay: bool = False,
) -> dict[str, float]:
    grouped: dict[str, list[EvidenceEvent]] = defaultdict(list)
    for event in events:
        if event.kind in kinds:
            grouped[event.topic_id.casefold()].append(event)

    estimates: dict[str, float] = {}
    for topic, topic_events in grouped.items():
        # Most recent evidence has the first/full repetition weight.
        ordered = sorted(topic_events, key=lambda item: _utc_timestamp(item.occurred_at), reverse=True)
        weighted_sum = 0.0
        total_weight = 0.0
        for index, event in enumerate(ordered, start=1):
            weight = _event_weight(event, index)
            if retention_delay:
                weight *= _retention_delay_weight(event.delay_hours)
            weighted_sum += weight * difficulty_adjusted_score(event.score, event.difficulty)
            total_weight += weight
        # A neutral prior prevents a single observation from looking certain.
        prior_weight = 0.55
        estimates[topic] = (weighted_sum + prior_weight * 0.50) / (
            total_weight + prior_weight
        )
    return estimates


def _calibration_score(events: Sequence[EvidenceEvent]) -> float:
    grouped: dict[str, list[EvidenceEvent]] = defaultdict(list)
    for event in events:
        if event.kind in {EvidenceKind.RETENTION, EvidenceKind.TRANSFER} and event.confidence is not None:
            grouped[event.topic_id.casefold()].append(event)
    if not grouped:
        return 50.0

    topic_values: list[float] = []
    for topic_events in grouped.values():
        ordered = sorted(topic_events, key=lambda item: _utc_timestamp(item.occurred_at), reverse=True)
        total = 0.0
        weight_sum = 0.0
        for index, event in enumerate(ordered, start=1):
            weight = _event_weight(event, index)
            observed = difficulty_adjusted_score(event.score, event.difficulty)
            # Absolute calibration error is transparent: 0 error => 100, a full
            # confidence miss => 0.  Confidence must be captured before grading.
            value = 1.0 - abs(clamp(event.confidence) - observed)
            total += value * weight
            weight_sum += weight
        topic_values.append(total / max(weight_sum, 1e-9))
    return 100.0 * sum(topic_values) / len(topic_values)


def _breadth_score(
    events: Sequence[EvidenceEvent],
    curriculum_topics: Iterable[str] | None,
) -> float:
    learning = [
        event
        for event in events
        if event.kind in {EvidenceKind.RETENTION, EvidenceKind.TRANSFER}
    ]
    grouped: dict[str, list[EvidenceEvent]] = defaultdict(list)
    for event in learning:
        grouped[event.topic_id.casefold()].append(event)

    if curriculum_topics is not None:
        denominator_topics = list(
            dict.fromkeys(topic.strip().casefold() for topic in curriculum_topics if topic.strip())
        )
    else:
        # Without a syllabus, four independently evidenced topics are the neutral
        # breadth horizon.  One repeatedly drilled topic can therefore reach no
        # more than one quarter of the breadth component.
        observed = list(grouped)
        denominator_topics = [*observed, *[f"__unseen_{i}" for i in range(max(0, 4 - len(observed)))]]
    if not denominator_topics:
        return 0.0

    contributions: list[float] = []
    for topic in denominator_topics:
        topic_events = sorted(
            grouped.get(topic, ()),
            key=lambda item: _utc_timestamp(item.occurred_at),
            reverse=True,
        )
        if not topic_events:
            contributions.append(0.0)
            continue
        weights = [_event_weight(event, index) for index, event in enumerate(topic_events, 1)]
        proficiency = sum(
            weight * difficulty_adjusted_score(event.score, event.difficulty)
            for event, weight in zip(topic_events, weights)
        ) / max(sum(weights), 1e-9)
        # Independent evidence confidence saturates; it never adds raw points.
        evidence_strength = min(1.0, sum(weights) / 1.75)
        contributions.append(proficiency * evidence_strength)
    return 100.0 * sum(contributions) / len(contributions)


def teachback_pair_credit_multiplier(
    event: EvidenceEvent,
    history: Iterable[EvidenceEvent],
) -> float:
    """Cap credit from a repeated/reciprocal teaching pair.

    An unordered pair receives weights 1.0 then 0.5, then zero within the supplied
    history.  If the latest prior event reverses teacher/learner roles within seven
    days, the current weight is halved.  Mutual teaching remains allowed; it just
    cannot manufacture unlimited leaderboard evidence.
    """

    if event.kind is not EvidenceKind.TEACHBACK or not event.partner_id:
        return 0.0
    pair = frozenset((event.learner_id, event.partner_id))
    earlier = sorted(
        (
            item
            for item in history
            if item.kind is EvidenceKind.TEACHBACK
            and item.partner_id
            and frozenset((item.learner_id, item.partner_id)) == pair
            and (_utc_timestamp(item.occurred_at), item.event_id)
            < (_utc_timestamp(event.occurred_at), event.event_id)
        ),
        key=lambda item: (_utc_timestamp(item.occurred_at), item.event_id),
    )
    pair_weights = (1.0, 0.5)
    multiplier = pair_weights[len(earlier)] if len(earlier) < len(pair_weights) else 0.0
    if earlier:
        latest = earlier[-1]
        reciprocal = latest.learner_id == event.partner_id and latest.partner_id == event.learner_id
        within_week = _utc_timestamp(event.occurred_at) - _utc_timestamp(latest.occurred_at) <= 7 * 86400
        if reciprocal and within_week:
            multiplier *= 0.5
    return multiplier


def _teaching_event_value(event: EvidenceEvent) -> float | None:
    if (
        event.pre_score is None
        or event.post_score is None
        or event.teaching_quality is None
        or not event.partner_id
    ):
        return None
    pre = clamp(event.pre_score)
    post = clamp(event.post_score)
    gain = max(0.0, post - pre)
    normalized_gain = min(1.0, gain / max(0.25, 1.0 - pre))
    quality = clamp(event.teaching_quality)
    # Post-transfer proficiency prevents a large relative gain from a still-low
    # result becoming a top teaching score.
    return normalized_gain * (0.55 + 0.45 * quality) * post


def _teaching_score(
    events: Sequence[EvidenceEvent],
    history: Sequence[EvidenceEvent],
) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for event in sorted(events, key=lambda item: (_utc_timestamp(item.occurred_at), item.event_id)):
        if event.kind is not EvidenceKind.TEACHBACK:
            continue
        value = _teaching_event_value(event)
        if value is None:
            continue
        multiplier = teachback_pair_credit_multiplier(event, history)
        multiplier *= 0.35 if event.integrity_state is IntegrityState.PROVISIONAL else 1.0
        weighted_sum += value * multiplier
        weight_sum += multiplier
    if weight_sum <= 0:
        return None
    # Neutral prior: one pair can demonstrate promise, not absolute certainty.
    prior_weight = 0.50
    return 100.0 * (weighted_sum + 0.50 * prior_weight) / (weight_sum + prior_weight)


def _valid_proof_ids(events: Sequence[EvidenceEvent]) -> set[str]:
    integrity = assess_integrity(events)
    excluded = {*integrity.held_event_ids, *integrity.resolved_event_ids}
    return {
        event.event_id
        for event in events
        if event.event_id not in excluded
        if event.integrity_state is IntegrityState.VERIFIED
        and event.kind in {EvidenceKind.RETENTION, EvidenceKind.TRANSFER}
    }


def _daily_commitment_values(
    commitments: Iterable[CommitmentEvent],
    valid_proof_ids: set[str],
) -> dict[date, float]:
    grouped: dict[date, list[float]] = defaultdict(list)
    used_proofs: set[str] = set()
    for commitment in sorted(commitments, key=lambda item: _utc_timestamp(item.scheduled_for)):
        if commitment.status is CommitmentStatus.EXCUSED:
            continue
        # One receipt can close one contract.  Silently ignore repeated links so
        # checkbox splitting neither improves nor unfairly damages reliability.
        if commitment.proof_event_id and commitment.proof_event_id in used_proofs:
            continue
        value = 0.0
        proof_backed = bool(
            commitment.proof_event_id and commitment.proof_event_id in valid_proof_ids
        )
        if proof_backed and commitment.status is CommitmentStatus.COMPLETED:
            value = 1.0
        elif proof_backed and commitment.status is CommitmentStatus.RECOVERED:
            value = 0.85
        if proof_backed and commitment.proof_event_id:
            used_proofs.add(commitment.proof_event_id)
        grouped[_calendar_day(commitment.scheduled_for)].append(value)
    # Average by planned day: splitting one contract into twenty checkboxes cannot
    # improve reliability.
    return {day: sum(values) / len(values) for day, values in grouped.items() if values}


def reliability_score(
    commitments: Iterable[CommitmentEvent],
    evidence_events: Iterable[EvidenceEvent],
    *,
    now: datetime | None = None,
) -> float | None:
    """Return proof-backed follow-through on recent planned days (0..100)."""

    evidence = list(evidence_events)
    days = _daily_commitment_values(commitments, _valid_proof_ids(evidence))
    if not days:
        return None
    reference = _calendar_day(now or datetime.now())
    weighted = 0.0
    total_weight = 0.0
    for day, value in days.items():
        age = max(0, (reference - day).days)
        weight = 0.92 ** min(age, 45)
        weighted += value * weight
        total_weight += weight
    return round(100.0 * weighted / max(total_weight, 1e-9), 2)


def private_momentum(
    commitments: Iterable[CommitmentEvent],
    evidence_events: Iterable[EvidenceEvent],
    *,
    now: datetime | None = None,
) -> MomentumSnapshot:
    """Calculate a private streak/momentum signal, separate from ProofScore."""

    commitments = list(commitments)
    evidence = list(evidence_events)
    values = _daily_commitment_values(commitments, _valid_proof_ids(evidence))
    ordered = sorted(values)
    successful = {day for day, value in values.items() if value >= 0.80}

    best = 0
    run = 0
    for day in ordered:
        if day in successful:
            run += 1
            best = max(best, run)
        else:
            run = 0

    current = 0
    for day in reversed(ordered):
        if day in successful:
            current += 1
        else:
            break

    if values:
        reference = _calendar_day(now or datetime.now())
        weighted = sum(value * (0.86 ** min(max(0, (reference - day).days), 30)) for day, value in values.items())
        weight_sum = sum(0.86 ** min(max(0, (reference - day).days), 30) for day in values)
        momentum = 100.0 * weighted / weight_sum
    else:
        momentum = 0.0
    return MomentumSnapshot(
        momentum=round(momentum, 2),
        current_streak=current,
        best_streak=best,
        planned_days=len(values),
        successful_days=len(successful),
    )


def daily_learning_bonus(
    events: Iterable[EvidenceEvent],
    day: date,
    *,
    cap: int = 20,
) -> DailyBonus:
    """Return capped cosmetic points for varied verified evidence on one day.

    The first retention and transfer proof per topic and the first teach-back per
    partner are eligible.  Replays do not generate bonus.  The return type marks
    explicitly that these points affect neither ProofScore nor mastery.
    """

    cap = max(0, int(cap))
    reasons: list[str] = []
    points = 0
    seen: set[tuple[str, str]] = set()
    for event in sorted(events, key=lambda item: (_utc_timestamp(item.occurred_at), item.event_id)):
        if _calendar_day(event.occurred_at) != day:
            continue
        if event.integrity_state is not IntegrityState.VERIFIED:
            continue
        if event.kind is EvidenceKind.TEACHBACK:
            if not event.partner_id or _teaching_event_value(event) is None:
                continue
            key = (event.kind.value, event.partner_id)
            award = 4
            label = f"Teach-back transfer with {event.partner_id}"
        else:
            key = (event.kind.value, event.topic_id.casefold())
            award = 5 if event.kind is EvidenceKind.TRANSFER else 3
            if event.kind is EvidenceKind.RETENTION and event.delay_hours >= 20:
                award += 1
            label = f"{event.kind.value.title()} proof: {event.topic_id}"
        if key in seen:
            continue
        seen.add(key)
        if points >= cap:
            break
        granted = min(award, cap - points)
        points += granted
        if granted:
            reasons.append(f"+{granted} {label}")
    return DailyBonus(day=day, points=points, cap=cap, reasons=tuple(reasons))


def _minimum_public_reasons(events: Sequence[EvidenceEvent]) -> list[str]:
    verified = [
        event
        for event in events
        if event.integrity_state is IntegrityState.VERIFIED
        and event.kind in {EvidenceKind.RETENTION, EvidenceKind.TRANSFER}
    ]
    topics = {event.topic_id.casefold() for event in verified}
    delayed = [
        event
        for event in verified
        if event.kind is EvidenceKind.RETENTION
        and event.delay_hours >= 20
        and clamp(event.score) >= 0.60
    ]
    transfers = [event for event in verified if event.kind is EvidenceKind.TRANSFER]
    reasons: list[str] = []
    if len(verified) < 4:
        reasons.append("At least four verified learning checks are needed.")
    if len(topics) < 2:
        reasons.append("Evidence across at least two topics is needed.")
    if len(delayed) < 2:
        reasons.append("Two checks after a delay of at least 20 hours are needed.")
    if not transfers:
        reasons.append("At least one fresh transfer question is needed.")
    return reasons


def calculate_proof_score(
    events: Iterable[EvidenceEvent],
    *,
    learner_id: str | None = None,
    curriculum_topics: Iterable[str] | None = None,
    commitments: Iterable[CommitmentEvent] = (),
    comparison_history: Iterable[EvidenceEvent] = (),
    now: datetime | None = None,
) -> ScoreBreakdown:
    """Calculate a deterministic, normalized and explainable ProofScore.

    ``comparison_history`` may include other learners' evidence for conservative
    near-copy checks and peer-pair caps.  It is never scored for ``learner_id``.
    """

    supplied = list(events)
    learner_ids = {event.learner_id for event in supplied}
    if learner_id is None:
        if len(learner_ids) > 1:
            raise ValueError("learner_id is required when events contain multiple learners")
        learner_id = next(iter(learner_ids), "")
    target = [event for event in supplied if event.learner_id == learner_id]
    history = _unique_events([*target, *comparison_history])
    integrity = assess_integrity(
        target,
        comparison_history=comparison_history,
        learner_id=learner_id,
    )
    held = set(integrity.held_event_ids)
    resolved = set(integrity.resolved_event_ids)
    usable = [
        event
        for event in target
        if event.event_id not in held
        and event.event_id not in resolved  # original stays excluded after verification
        and event.integrity_state is not IntegrityState.HELD
    ]
    # A memorised retrieval challenge is one piece of evidence, however many
    # times it is replayed. The prompt id is a fingerprint of MCQ text/options.
    distinct_usable: list[EvidenceEvent] = []
    seen_retrieval_prompts: set[tuple[str, str]] = set()
    for event in usable:
        if event.kind is EvidenceKind.RETENTION and event.prompt_id:
            retrieval_key = (event.topic_id.casefold(), event.prompt_id)
            if retrieval_key in seen_retrieval_prompts:
                continue
            seen_retrieval_prompts.add(retrieval_key)
        distinct_usable.append(event)
    usable = distinct_usable

    knowledge_by_topic = _topic_estimates(
        usable,
        {EvidenceKind.RETENTION},
        retention_delay=True,
    )
    transfer_by_topic = _topic_estimates(usable, {EvidenceKind.TRANSFER})
    knowledge = (
        100.0 * sum(knowledge_by_topic.values()) / len(knowledge_by_topic)
        if knowledge_by_topic
        else 50.0
    )
    depth = (
        100.0 * sum(transfer_by_topic.values()) / len(transfer_by_topic)
        if transfer_by_topic
        else 50.0
    )
    calibration = _calibration_score(usable)
    breadth = _breadth_score(usable, curriculum_topics)
    teaching = _teaching_score(usable, history)
    reliability = reliability_score(commitments, usable, now=now)

    components: dict[str, float | None] = {
        "knowledge": knowledge,
        "transfer_depth": depth,
        "calibration": calibration,
        "topic_breadth": breadth,
        "teaching_impact": teaching,
        "reliability": reliability,
    }
    active_weights = {
        name: weight
        for name, weight in PROOF_SCORE_WEIGHTS.items()
        if components[name] is not None
    }
    weight_total = sum(active_weights.values()) or 1.0
    normalized_weights = {
        name: weight / weight_total for name, weight in active_weights.items()
    }
    composite = sum(
        (components[name] or 0.0) * weight for name, weight in normalized_weights.items()
    )
    composite = clamp(composite, 0.0, 100.0)

    provisional_reasons = _minimum_public_reasons(usable)
    if integrity.state is IntegrityState.HELD:
        state = IntegrityState.HELD
    elif provisional_reasons:
        state = IntegrityState.PROVISIONAL
    else:
        state = IntegrityState.VERIFIED

    verified_topics = {
        event.topic_id.casefold()
        for event in usable
        if event.integrity_state is IntegrityState.VERIFIED
        and event.kind in {EvidenceKind.RETENTION, EvidenceKind.TRANSFER}
    }
    explanations = (
        "Knowledge is an equal-topic estimate of retrieval after delay; repeated attempts have diminishing weight.",
        "Transfer depth comes from independently worded application questions with bounded difficulty adjustment.",
        "Breadth rewards demonstrated proficiency across the syllabus, never the number of attempts on one topic.",
        "Teaching Impact uses the learner's pre/post transfer gain and caps repeated or reciprocal pairs.",
        "Reliability counts only proof-backed learning contracts and is averaged by day.",
        "Study time, message count, note length, text style and presumed AI authorship never add ProofScore.",
    )
    return ScoreBreakdown(
        proof_score=round(composite, 2),
        integrity_state=state,
        leaderboard_eligible=state is IntegrityState.VERIFIED,
        knowledge=round(knowledge, 2),
        transfer_depth=round(depth, 2),
        calibration=round(calibration, 2),
        topic_breadth=round(breadth, 2),
        teaching_impact=round(teaching, 2) if teaching is not None else None,
        reliability=reliability,
        component_weights={name: round(weight, 6) for name, weight in normalized_weights.items()},
        evidence_count=len(usable),
        verified_topic_count=len(verified_topics),
        held_event_ids=integrity.held_event_ids,
        provisional_reasons=tuple(provisional_reasons),
        verification_requests=integrity.verification_requests,
        explanations=explanations,
    )


# Friendly aliases for app code and write-up terminology.
compute_proof_score = calculate_proof_score
calculate_momentum = private_momentum


__all__ = [
    "CommitmentEvent",
    "CommitmentStatus",
    "DailyBonus",
    "EvidenceEvent",
    "EvidenceKind",
    "IntegrityFlag",
    "IntegrityReport",
    "IntegrityState",
    "MomentumSnapshot",
    "ScoreBreakdown",
    "VerificationRequest",
    "assess_integrity",
    "bounded_difficulty",
    "calculate_momentum",
    "calculate_proof_score",
    "compute_proof_score",
    "daily_learning_bonus",
    "difficulty_adjusted_score",
    "diminishing_topic_credit",
    "is_near_copy",
    "near_copy_ratio",
    "normalize_response",
    "private_momentum",
    "reliability_score",
    "teachback_pair_credit_multiplier",
]
