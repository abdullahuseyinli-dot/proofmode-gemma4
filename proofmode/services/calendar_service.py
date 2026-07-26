"""Calendar ingestion, planning, and export helpers for ProofMode.

The public API is intentionally framework independent so it can be used by
Streamlit, a desktop shell, or a background reminder process.  Google Calendar
imports are lazy: importing this module never requires Google SDK packages or
credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from email.utils import parsedate_to_datetime
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_TIMEZONE = "Europe/London"
UTC = timezone.utc

_TIMEZONE_ALIASES = {
    # Common Windows/Outlook identifiers seen in uploaded university calendars.
    "GMT STANDARD TIME": "Europe/London",
    "GREENWICH STANDARD TIME": "Europe/London",
    "W. EUROPE STANDARD TIME": "Europe/Berlin",
    "EASTERN STANDARD TIME": "America/New_York",
    "PACIFIC STANDARD TIME": "America/Los_Angeles",
}


class CalendarError(ValueError):
    """Raised when calendar data cannot be parsed or scheduled."""


class CalendarIntegrationUnavailable(RuntimeError):
    """Raised when the optional Google Calendar integration cannot be used."""


def _timezone(name: str | None) -> tzinfo:
    if not name or name.upper() in {"UTC", "ETC/UTC", "GMT"}:
        return UTC
    name = _TIMEZONE_ALIASES.get(name.upper(), name)
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        # Windows Python installations do not always ship the IANA database.
        # python-dateutil is part of the app environment and carries its own
        # resolver; the import remains optional for standalone use.
        try:
            from dateutil.tz import gettz

            fallback = gettz(name)
            if fallback is not None:
                return fallback
        except ImportError:
            pass
        # UTC is deterministic and still leaves upload/export usable if neither
        # time-zone database is installed.
        return UTC


def _aware(value: datetime, timezone_name: str = DEFAULT_TIMEZONE) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=_timezone(timezone_name))
    return value


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_iso(value: str | datetime | date | None, timezone_name: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _aware(value, timezone_name)
    if isinstance(value, date):
        return datetime.combine(value, time.min, _timezone(timezone_name))
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(cleaned)
    except ValueError:
        try:
            result = parsedate_to_datetime(cleaned)
        except (TypeError, ValueError) as exc:
            raise CalendarError(f"Unsupported date/time value: {value!r}") from exc
    return _aware(result, timezone_name)


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """A normalized calendar event.

    ``end`` is always exclusive, matching RFC 5545.  Timed values are always
    timezone-aware.  Transparent and cancelled events have ``blocks_time`` set
    to ``False`` so availability inference ignores them.
    """

    uid: str
    title: str
    start: datetime
    end: datetime
    description: str = ""
    location: str = ""
    all_day: bool = False
    blocks_time: bool = True
    source: str = "ics"
    recurrence: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_minutes(self) -> int:
        return max(0, round((self.end - self.start).total_seconds() / 60))

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "description": self.description,
            "location": self.location,
            "all_day": self.all_day,
            "blocks_time": self.blocks_time,
            "source": self.source,
            "recurrence": self.recurrence,
            "duration_minutes": self.duration_minutes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], timezone_name: str = DEFAULT_TIMEZONE
    ) -> "CalendarEvent":
        start = _parse_iso(data.get("start"), timezone_name)
        end = _parse_iso(data.get("end"), timezone_name)
        if start is None:
            raise CalendarError("Calendar event is missing a start time")
        all_day = bool(data.get("all_day", False))
        if end is None:
            end = start + (timedelta(days=1) if all_day else timedelta(hours=1))
        if end <= start:
            raise CalendarError("Calendar event end must be after its start")
        return cls(
            uid=str(data.get("uid") or uuid4()),
            title=str(data.get("title") or data.get("summary") or "Untitled event"),
            start=start,
            end=end,
            description=str(data.get("description") or ""),
            location=str(data.get("location") or ""),
            all_day=all_day,
            blocks_time=bool(data.get("blocks_time", True)),
            source=str(data.get("source") or "app"),
            recurrence=data.get("recurrence"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class DetectedAssessment:
    event_uid: str
    title: str
    kind: str
    when: datetime
    confidence: float
    matched_terms: tuple[str, ...] = ()
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_uid": self.event_uid,
            "title": self.title,
            "kind": self.kind,
            "when": self.when.isoformat(),
            "confidence": round(self.confidence, 3),
            "matched_terms": list(self.matched_terms),
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class AvailabilitySlot:
    start: datetime
    end: datetime

    @property
    def duration_minutes(self) -> int:
        return max(0, round((self.end - self.start).total_seconds() / 60))

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_minutes": self.duration_minutes,
        }


@dataclass(frozen=True, slots=True)
class StudyTask:
    topic: str
    estimated_minutes: int
    priority: float = 1.0
    deadline: datetime | None = None
    reason: str = ""
    min_block_minutes: int = 25
    max_block_minutes: int = 60
    task_id: str = field(default_factory=lambda: str(uuid4()))

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], timezone_name: str = DEFAULT_TIMEZONE
    ) -> "StudyTask":
        minutes = data.get("estimated_minutes", data.get("minutes", data.get("remaining_minutes", 60)))
        return cls(
            topic=str(data.get("topic") or data.get("name") or data.get("title") or "Study session"),
            estimated_minutes=max(1, int(minutes)),
            priority=max(0.01, float(data.get("priority", 1.0))),
            deadline=_parse_iso(data.get("deadline") or data.get("exam_date"), timezone_name),
            reason=str(data.get("reason") or ""),
            min_block_minutes=max(10, int(data.get("min_block_minutes", 25))),
            max_block_minutes=max(10, int(data.get("max_block_minutes", 60))),
            task_id=str(data.get("task_id") or data.get("id") or uuid4()),
        )


@dataclass(frozen=True, slots=True)
class StudyBlock:
    uid: str
    topic: str
    title: str
    start: datetime
    end: datetime
    priority: float = 1.0
    reason: str = ""
    task_id: str = ""
    reminders: tuple[int, ...] = (15, 5)

    @property
    def duration_minutes(self) -> int:
        return max(0, round((self.end - self.start).total_seconds() / 60))

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "topic": self.topic,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "priority": round(self.priority, 4),
            "reason": self.reason,
            "task_id": self.task_id,
            "duration_minutes": self.duration_minutes,
            "reminders": list(self.reminders),
        }

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], timezone_name: str = DEFAULT_TIMEZONE
    ) -> "StudyBlock":
        start = _parse_iso(data.get("start"), timezone_name)
        end = _parse_iso(data.get("end"), timezone_name)
        if start is None or end is None or end <= start:
            raise CalendarError("Study block needs valid start and end times")
        topic = str(data.get("topic") or data.get("title") or "Study session")
        reminders = tuple(int(value) for value in data.get("reminders", (15, 5)))
        return cls(
            uid=str(data.get("uid") or uuid4()),
            topic=topic,
            title=str(data.get("title") or f"ProofMode • {topic}"),
            start=start,
            end=end,
            priority=float(data.get("priority", 1.0)),
            reason=str(data.get("reason") or ""),
            task_id=str(data.get("task_id") or ""),
            reminders=reminders,
        )


def _unfold_ics(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    unfolded: list[str] = []
    for line in normalized.split("\n"):
        if line.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return unfolded


def _split_property(line: str) -> tuple[str, dict[str, str], str]:
    head, separator, value = line.partition(":")
    if not separator:
        return head.upper(), {}, ""
    pieces = head.split(";")
    name = pieces[0].upper()
    params: dict[str, str] = {}
    for item in pieces[1:]:
        key, equals, param_value = item.partition("=")
        if equals:
            params[key.upper()] = param_value.strip('"')
    return name, params, value


def _unescape_ics(value: str) -> str:
    return (
        value.replace("\\N", "\n")
        .replace("\\n", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


def _parse_ics_datetime(
    raw: str, params: Mapping[str, str], default_timezone: str
) -> tuple[datetime, bool]:
    value = raw.strip()
    is_date = params.get("VALUE", "").upper() == "DATE" or bool(re.fullmatch(r"\d{8}", value))
    if is_date:
        parsed_date = datetime.strptime(value[:8], "%Y%m%d").date()
        return datetime.combine(parsed_date, time.min, _timezone(params.get("TZID") or default_timezone)), True

    tzid = params.get("TZID")
    if value.endswith("Z"):
        value = value[:-1]
        zone = UTC
    else:
        zone = _timezone(tzid or default_timezone)

    formats = ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M")
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=zone), False
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(value)
        return _aware(parsed, tzid or default_timezone), False
    except ValueError as exc:
        raise CalendarError(f"Invalid ICS date/time: {raw!r}") from exc


_DURATION_PATTERN = re.compile(
    r"^(?P<sign>-)?P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$",
    re.IGNORECASE,
)


def _parse_duration(value: str) -> timedelta | None:
    match = _DURATION_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    amount = timedelta(
        days=int(match.group("days") or 0),
        hours=int(match.group("hours") or 0),
        minutes=int(match.group("minutes") or 0),
        seconds=int(match.group("seconds") or 0),
    )
    return -amount if match.group("sign") else amount


def parse_ics(
    payload: bytes | str, default_timezone: str = DEFAULT_TIMEZONE
) -> list[CalendarEvent]:
    """Parse uploaded RFC 5545 data into normalized events.

    The parser deliberately covers the interoperable subset needed by Google,
    Outlook, and Apple exports without making ``icalendar`` a hard dependency.
    Folded lines, date-only events, time zones, durations, transparency, and the
    common recurrence fields are supported.  Malformed individual VEVENTs are
    skipped only when they have no usable start time.
    """

    if isinstance(payload, bytes):
        text = payload.decode("utf-8-sig", errors="replace")
    elif isinstance(payload, str):
        text = payload.lstrip("\ufeff")
    else:
        raise TypeError("ICS payload must be bytes or text")
    if "BEGIN:VCALENDAR" not in text.upper():
        raise CalendarError("This does not look like an ICS calendar")

    raw_events: list[dict[str, list[tuple[dict[str, str], str]]]] = []
    current: dict[str, list[tuple[dict[str, str], str]]] | None = None
    for line in _unfold_ics(text):
        upper = line.upper()
        if upper == "BEGIN:VEVENT":
            current = {}
            continue
        if upper == "END:VEVENT":
            if current is not None:
                raw_events.append(current)
            current = None
            continue
        if current is None:
            continue
        name, params, value = _split_property(line)
        current.setdefault(name, []).append((params, value))

    events: list[CalendarEvent] = []
    for raw in raw_events:
        if "DTSTART" not in raw:
            continue
        start_params, start_raw = raw["DTSTART"][0]
        try:
            start, all_day = _parse_ics_datetime(start_raw, start_params, default_timezone)
        except CalendarError:
            continue

        end: datetime | None = None
        if raw.get("DTEND"):
            end_params, end_raw = raw["DTEND"][0]
            try:
                end, _ = _parse_ics_datetime(end_raw, end_params, default_timezone)
            except CalendarError:
                end = None
        if end is None and raw.get("DURATION"):
            duration = _parse_duration(raw["DURATION"][0][1])
            if duration and duration > timedelta(0):
                end = start + duration
        if end is None:
            end = start + (timedelta(days=1) if all_day else timedelta(hours=1))
        if end <= start:
            continue

        def value_of(name: str, default: str = "") -> str:
            return _unescape_ics(raw.get(name, [({}, default)])[0][1])

        status = value_of("STATUS").upper()
        transparency = value_of("TRANSP").upper()
        exdates: list[str] = []
        for ex_params, ex_value in raw.get("EXDATE", []):
            for item in ex_value.split(","):
                try:
                    excluded, _ = _parse_ics_datetime(item, ex_params, default_timezone)
                    exdates.append(excluded.isoformat())
                except CalendarError:
                    continue
        metadata: dict[str, Any] = {
            "status": status,
            "transparency": transparency,
        }
        if exdates:
            metadata["exdates"] = exdates
        events.append(
            CalendarEvent(
                uid=value_of("UID", str(uuid4())),
                title=value_of("SUMMARY", "Untitled event"),
                start=start,
                end=end,
                description=value_of("DESCRIPTION"),
                location=value_of("LOCATION"),
                all_day=all_day,
                blocks_time=status != "CANCELLED" and transparency != "TRANSPARENT",
                source="ics",
                recurrence=value_of("RRULE") or None,
                metadata=metadata,
            )
        )
    return sorted(events, key=lambda event: (event.start, event.end, event.title.lower()))


def normalize_events(
    events: Iterable[CalendarEvent | Mapping[str, Any]],
    timezone_name: str = DEFAULT_TIMEZONE,
) -> list[CalendarEvent]:
    """Coerce event dictionaries and dataclasses into one sorted representation."""

    normalized: list[CalendarEvent] = []
    for value in events:
        event = value if isinstance(value, CalendarEvent) else CalendarEvent.from_mapping(value, timezone_name)
        local_zone = _timezone(timezone_name)
        normalized.append(
            replace(event, start=_aware(event.start, timezone_name).astimezone(local_zone), end=_aware(event.end, timezone_name).astimezone(local_zone))
        )
    return sorted(normalized, key=lambda event: (event.start, event.end, event.title.lower()))


_EXAM_TERMS: Mapping[str, float] = {
    "final exam": 0.26,
    "exam": 0.22,
    "midterm": 0.22,
    "mid-term": 0.22,
    "viva": 0.20,
    "practical assessment": 0.20,
    "assessment": 0.11,
    "test": 0.10,
}
_DEADLINE_TERMS: Mapping[str, float] = {
    "submission deadline": 0.27,
    "deadline": 0.24,
    "due date": 0.22,
    "due": 0.16,
    "submit": 0.14,
    "submission": 0.14,
    "coursework": 0.11,
    "assignment": 0.11,
    "essay": 0.08,
    "project": 0.06,
}
_PREPARATION_TERMS = ("revision", "revise", "study", "prep", "prepare", "practice questions")


def _term_hits(text: str, terms: Mapping[str, float]) -> list[tuple[str, float]]:
    hits: list[tuple[str, float]] = []
    for term, weight in terms.items():
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE):
            hits.append((term, weight))
    return hits


def detect_assessments(
    events: Iterable[CalendarEvent | Mapping[str, Any]],
    timezone_name: str = DEFAULT_TIMEZONE,
    min_confidence: float = 0.55,
) -> list[DetectedAssessment]:
    """Detect likely exams and coursework deadlines with explainable heuristics."""

    detections: list[DetectedAssessment] = []
    for event in normalize_events(events, timezone_name):
        title = event.title.lower()
        body = f"{event.title}\n{event.description}".lower()
        exam_hits = _term_hits(body, _EXAM_TERMS)
        deadline_hits = _term_hits(body, _DEADLINE_TERMS)
        if not exam_hits and not deadline_hits:
            continue

        title_exam_hits = _term_hits(title, _EXAM_TERMS)
        title_deadline_hits = _term_hits(title, _DEADLINE_TERMS)
        exam_score = 0.35 + sum(weight for _, weight in exam_hits[:3])
        deadline_score = 0.35 + sum(weight for _, weight in deadline_hits[:3])
        if title_exam_hits:
            exam_score += 0.14
        if title_deadline_hits:
            deadline_score += 0.14
        if any(term in title for term in _PREPARATION_TERMS):
            exam_score -= 0.32
            deadline_score -= 0.16

        if exam_score >= deadline_score:
            kind = "exam"
            score = exam_score
            hits = exam_hits
        else:
            kind = "deadline"
            score = deadline_score
            hits = deadline_hits
        score = min(0.98, max(0.0, score))
        if score < min_confidence:
            continue
        terms = tuple(dict.fromkeys(term for term, _ in hits))
        detections.append(
            DetectedAssessment(
                event_uid=event.uid,
                title=event.title,
                kind=kind,
                when=event.start,
                confidence=score,
                matched_terms=terms,
                evidence=f"Calendar text matched: {', '.join(terms)}",
            )
        )
    return sorted(detections, key=lambda result: (result.when, -result.confidence))


def _rrule_parts(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in value.split(";"):
        key, equals, item = part.partition("=")
        if equals:
            result[key.upper()] = item
    return result


_WEEKDAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}


def expand_recurring_events(
    events: Iterable[CalendarEvent | Mapping[str, Any]],
    range_start: datetime,
    range_end: datetime,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> list[CalendarEvent]:
    """Expand common DAILY and WEEKLY recurrence rules inside a date range."""

    start_bound = _aware(range_start, timezone_name)
    end_bound = _aware(range_end, timezone_name)
    expanded: list[CalendarEvent] = []
    for event in normalize_events(events, timezone_name):
        if not event.recurrence:
            if event.end > start_bound and event.start < end_bound:
                expanded.append(event)
            continue
        rule = _rrule_parts(event.recurrence)
        frequency = rule.get("FREQ", "").upper()
        if frequency not in {"DAILY", "WEEKLY"}:
            if event.end > start_bound and event.start < end_bound:
                expanded.append(event)
            continue
        interval = max(1, int(rule.get("INTERVAL", "1")))
        count_limit = max(0, int(rule["COUNT"])) if rule.get("COUNT", "").isdigit() else None
        until: datetime | None = None
        if rule.get("UNTIL"):
            try:
                until, _ = _parse_ics_datetime(rule["UNTIL"], {}, timezone_name)
            except CalendarError:
                pass
        weekdays = {
            _WEEKDAYS[item[-2:]]
            for item in rule.get("BYDAY", "").split(",")
            if item[-2:] in _WEEKDAYS
        } or {event.start.weekday()}
        excluded = {
            _parse_iso(item, timezone_name).astimezone(UTC).replace(microsecond=0)
            for item in event.metadata.get("exdates", [])
            if _parse_iso(item, timezone_name) is not None
        }
        duration = event.end - event.start
        candidate = event.start
        base_week = event.start.date() - timedelta(days=event.start.weekday())
        occurrence_number = 0
        checked_days = 0
        hard_stop = min(end_bound, event.start + timedelta(days=3660))
        while candidate < hard_stop and checked_days < 4000:
            if frequency == "DAILY":
                qualifies = (candidate.date() - event.start.date()).days % interval == 0
            else:
                week = (candidate.date() - base_week).days // 7
                qualifies = week % interval == 0 and candidate.weekday() in weekdays
            if candidate >= event.start and qualifies:
                occurrence_number += 1
                if count_limit is not None and occurrence_number > count_limit:
                    break
                if until is not None and candidate > until:
                    break
                utc_candidate = candidate.astimezone(UTC).replace(microsecond=0)
                occurrence_end = candidate + duration
                if utc_candidate not in excluded and occurrence_end > start_bound and candidate < end_bound:
                    expanded.append(
                        replace(
                            event,
                            uid=f"{event.uid}#{candidate.isoformat()}",
                            start=candidate,
                            end=occurrence_end,
                            recurrence=None,
                            metadata={**event.metadata, "recurring_parent": event.uid},
                        )
                    )
            candidate += timedelta(days=1)
            checked_days += 1
    return sorted(expanded, key=lambda event: (event.start, event.end))


def _calendar_range(
    start: date | datetime, end: date | datetime, timezone_name: str
) -> tuple[datetime, datetime]:
    zone = _timezone(timezone_name)
    if isinstance(start, datetime):
        range_start = _aware(start, timezone_name).astimezone(zone)
    else:
        range_start = datetime.combine(start, time.min, zone)
    if isinstance(end, datetime):
        range_end = _aware(end, timezone_name).astimezone(zone)
    else:
        # Date ranges are inclusive for the caller and exclusive internally.
        range_end = datetime.combine(end + timedelta(days=1), time.min, zone)
    if range_end <= range_start:
        raise CalendarError("Availability range end must be after start")
    return range_start, range_end


def _merge_intervals(intervals: Sequence[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(intervals, key=lambda value: value[0]):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def infer_study_availability(
    events: Iterable[CalendarEvent | Mapping[str, Any]],
    range_start: date | datetime,
    range_end: date | datetime,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    daily_start: time = time(8, 0),
    daily_end: time = time(22, 0),
    min_slot_minutes: int = 25,
    buffer_minutes: int = 10,
    include_weekends: bool = True,
    block_all_day: bool = False,
) -> list[AvailabilitySlot]:
    """Return usable study windows after subtracting busy calendar events."""

    if daily_end <= daily_start:
        raise CalendarError("daily_end must be later than daily_start")
    if min_slot_minutes < 1 or buffer_minutes < 0:
        raise CalendarError("Slot and buffer durations must be non-negative")
    start_bound, end_bound = _calendar_range(range_start, range_end, timezone_name)
    zone = _timezone(timezone_name)
    occurrences = expand_recurring_events(events, start_bound, end_bound, timezone_name)

    slots: list[AvailabilitySlot] = []
    day = start_bound.date()
    last_day = (end_bound - timedelta(microseconds=1)).date()
    buffer = timedelta(minutes=buffer_minutes)
    while day <= last_day:
        if include_weekends or day.weekday() < 5:
            window_start = max(datetime.combine(day, daily_start, zone), start_bound)
            window_end = min(datetime.combine(day, daily_end, zone), end_bound)
            if window_end > window_start:
                blocked: list[tuple[datetime, datetime]] = []
                for event in occurrences:
                    if not event.blocks_time or (event.all_day and not block_all_day):
                        continue
                    event_start = event.start.astimezone(zone)
                    event_end = event.end.astimezone(zone)
                    busy_start = max(window_start, event_start - (timedelta(0) if event.all_day else buffer))
                    busy_end = min(window_end, event_end + (timedelta(0) if event.all_day else buffer))
                    if busy_end > busy_start:
                        blocked.append((busy_start, busy_end))
                cursor = window_start
                for busy_start, busy_end in _merge_intervals(blocked):
                    if busy_start > cursor and (busy_start - cursor) >= timedelta(minutes=min_slot_minutes):
                        slots.append(AvailabilitySlot(cursor, busy_start))
                    cursor = max(cursor, busy_end)
                if window_end > cursor and (window_end - cursor) >= timedelta(minutes=min_slot_minutes):
                    slots.append(AvailabilitySlot(cursor, window_end))
        day += timedelta(days=1)
    return slots


def schedule_study_blocks(
    tasks: Iterable[StudyTask | Mapping[str, Any]],
    events: Iterable[CalendarEvent | Mapping[str, Any]],
    range_start: date | datetime,
    range_end: date | datetime,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    daily_start: time = time(8, 0),
    daily_end: time = time(22, 0),
    min_block_minutes: int = 25,
    max_block_minutes: int = 60,
    break_minutes: int = 10,
    event_buffer_minutes: int = 10,
    include_weekends: bool = True,
) -> list[StudyBlock]:
    """Allocate prioritized tasks into conflict-free calendar blocks.

    Deadline urgency, task priority, and topic rotation affect selection.  The
    function is deterministic except for generated UIDs and does not mutate the
    supplied tasks or events.
    """

    task_values = [
        value if isinstance(value, StudyTask) else StudyTask.from_mapping(value, timezone_name)
        for value in tasks
    ]
    if not task_values:
        return []
    availability = infer_study_availability(
        events,
        range_start,
        range_end,
        timezone_name=timezone_name,
        daily_start=daily_start,
        daily_end=daily_end,
        min_slot_minutes=max(10, min_block_minutes),
        buffer_minutes=event_buffer_minutes,
        include_weekends=include_weekends,
    )
    state: dict[str, dict[str, int]] = {
        task.task_id: {"remaining": task.estimated_minutes, "sessions": 0}
        for task in task_values
    }
    blocks: list[StudyBlock] = []
    previous_task: str | None = None
    break_delta = timedelta(minutes=max(0, break_minutes))

    for slot in availability:
        cursor = slot.start
        while (slot.end - cursor) >= timedelta(minutes=10):
            capacity = int((slot.end - cursor).total_seconds() // 60)
            candidates: list[tuple[float, StudyTask, int]] = []
            for task in task_values:
                task_state = state[task.task_id]
                remaining = task_state["remaining"]
                if remaining <= 0:
                    continue
                smallest = min(task.min_block_minutes, remaining)
                smallest = max(10, min(smallest, max_block_minutes))
                if capacity < smallest:
                    continue
                if task.deadline is not None:
                    deadline = _aware(task.deadline, timezone_name)
                    if cursor + timedelta(minutes=smallest) > deadline:
                        continue
                    hours_left = max(1.0, (deadline - cursor).total_seconds() / 3600)
                    urgency = 1.0 + min(3.0, 72.0 / hours_left)
                else:
                    urgency = 1.0
                rotation = 0.58 if previous_task == task.task_id and len(task_values) > 1 else 1.0
                fairness = 1.0 + task_state["sessions"] * 0.22
                score = task.priority * urgency * rotation / fairness
                candidates.append((score, task, smallest))
            if not candidates:
                break
            _, selected, selected_minimum = max(
                candidates, key=lambda item: (item[0], -item[1].estimated_minutes, item[1].topic)
            )
            selected_state = state[selected.task_id]
            block_limit = min(max_block_minutes, selected.max_block_minutes)
            minutes = min(capacity, block_limit, selected_state["remaining"])
            if minutes < selected_minimum:
                break
            block_end = cursor + timedelta(minutes=minutes)
            reason = selected.reason or "Scheduled from priority, available time, and deadline urgency."
            blocks.append(
                StudyBlock(
                    uid=f"proofmode-{uuid4()}@local",
                    topic=selected.topic,
                    title=f"ProofMode • {selected.topic}",
                    start=cursor,
                    end=block_end,
                    priority=selected.priority,
                    reason=reason,
                    task_id=selected.task_id,
                )
            )
            selected_state["remaining"] -= minutes
            selected_state["sessions"] += 1
            previous_task = selected.task_id
            cursor = block_end + break_delta
    return blocks


def _escape_ics(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
        .replace(";", "\\;")
        .replace(",", "\\,")
    )


def _fold_ics_line(line: str, limit: int = 75) -> list[str]:
    """Fold a content line without splitting a UTF-8 codepoint."""

    encoded = line.encode("utf-8")
    if len(encoded) <= limit:
        return [line]
    output: list[str] = []
    remaining = line
    first = True
    while remaining:
        prefix = "" if first else " "
        byte_budget = limit - len(prefix.encode("utf-8"))
        consumed = ""
        used = 0
        for char in remaining:
            size = len(char.encode("utf-8"))
            if used + size > byte_budget:
                break
            consumed += char
            used += size
        if not consumed:
            consumed = remaining[0]
        output.append(prefix + consumed)
        remaining = remaining[len(consumed) :]
        first = False
    return output


def export_study_blocks_ics(
    blocks: Iterable[StudyBlock | Mapping[str, Any]],
    *,
    calendar_name: str = "ProofMode Study Plan",
    timezone_name: str = DEFAULT_TIMEZONE,
    default_reminders: Sequence[int] = (15, 5),
) -> bytes:
    """Export study blocks as a standards-compliant calendar with alarms."""

    normalized = [
        value if isinstance(value, StudyBlock) else StudyBlock.from_mapping(value, timezone_name)
        for value in blocks
    ]
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//ProofMode//Study Contract Calendar//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_escape_ics(calendar_name)}",
    ]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for block in sorted(normalized, key=lambda item: item.start):
        start_utc = _aware(block.start, timezone_name).astimezone(UTC)
        end_utc = _aware(block.end, timezone_name).astimezone(UTC)
        description = block.reason or f"Learning contract for {block.topic}. Submit a Learning Receipt when finished."
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{_escape_ics(block.uid)}",
                f"DTSTAMP:{stamp}",
                f"DTSTART:{start_utc.strftime('%Y%m%dT%H%M%SZ')}",
                f"DTEND:{end_utc.strftime('%Y%m%dT%H%M%SZ')}",
                f"SUMMARY:{_escape_ics(block.title)}",
                f"DESCRIPTION:{_escape_ics(description)}",
                "CATEGORIES:STUDY,PROOFMODE",
                f"X-PROOFMODE-TOPIC:{_escape_ics(block.topic)}",
                f"X-PROOFMODE-TASK-ID:{_escape_ics(block.task_id)}",
            ]
        )
        reminders = block.reminders or tuple(default_reminders)
        for minutes in sorted({int(value) for value in reminders if int(value) > 0}, reverse=True):
            lines.extend(
                [
                    "BEGIN:VALARM",
                    f"TRIGGER:-PT{minutes}M",
                    "ACTION:DISPLAY",
                    f"DESCRIPTION:{_escape_ics('Upcoming: ' + block.title)}",
                    "END:VALARM",
                ]
            )
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    folded = [piece for line in lines for piece in _fold_ics_line(line)]
    return ("\r\n".join(folded) + "\r\n").encode("utf-8")


def analyze_calendar(
    payload: bytes | str,
    range_start: date | datetime,
    range_end: date | datetime,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    daily_start: time = time(8, 0),
    daily_end: time = time(22, 0),
) -> dict[str, Any]:
    """One-call UI helper returning imported events, assessments, and free time."""

    events = parse_ics(payload, timezone_name)
    assessments = detect_assessments(events, timezone_name)
    availability = infer_study_availability(
        events,
        range_start,
        range_end,
        timezone_name=timezone_name,
        daily_start=daily_start,
        daily_end=daily_end,
    )
    return {
        "events": [event.to_dict() for event in events],
        "assessments": [assessment.to_dict() for assessment in assessments],
        "availability": [slot.to_dict() for slot in availability],
        "counts": {
            "events": len(events),
            "assessments": len(assessments),
            "available_slots": len(availability),
        },
    }


class GoogleCalendarAdapter:
    """Lazy optional adapter for reading and writing Google Calendar events."""

    SCOPES = ("https://www.googleapis.com/auth/calendar.events",)

    def __init__(
        self,
        credentials_path: str | os.PathLike[str] | None = None,
        token_path: str | os.PathLike[str] | None = None,
        calendar_id: str = "primary",
        timezone_name: str = DEFAULT_TIMEZONE,
    ) -> None:
        credentials_value = credentials_path or os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
        token_value = token_path or os.getenv("GOOGLE_CALENDAR_TOKEN")
        self.credentials_path = Path(credentials_value) if credentials_value else None
        self.token_path = Path(token_value) if token_value else None
        self.calendar_id = calendar_id
        self.timezone_name = timezone_name
        self._service: Any = None

    @property
    def dependencies_available(self) -> bool:
        try:
            return all(
                importlib.util.find_spec(package) is not None
                for package in ("google.auth", "googleapiclient", "google_auth_oauthlib")
            )
        except (ImportError, ModuleNotFoundError, AttributeError):
            return False

    @property
    def is_configured(self) -> bool:
        return bool(
            (self.credentials_path and self.credentials_path.is_file())
            or (self.token_path and self.token_path.is_file())
        )

    def status(self) -> dict[str, Any]:
        return {
            "dependencies_available": self.dependencies_available,
            "configured": self.is_configured,
            "calendar_id": self.calendar_id,
            "timezone": self.timezone_name,
        }

    def _build_service(self) -> Any:
        if self._service is not None:
            return self._service
        if not self.dependencies_available:
            raise CalendarIntegrationUnavailable(
                "Google Calendar support needs google-api-python-client, google-auth, and google-auth-oauthlib"
            )
        if not self.is_configured:
            raise CalendarIntegrationUnavailable(
                "Provide GOOGLE_CALENDAR_CREDENTIALS or GOOGLE_CALENDAR_TOKEN; ICS export remains available"
            )

        # Imports stay here so the core app remains import-safe and offline-first.
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        credentials: Any = None
        if self.token_path and self.token_path.is_file():
            credentials = Credentials.from_authorized_user_file(str(self.token_path), list(self.SCOPES))
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        if not credentials or not credentials.valid:
            if not self.credentials_path or not self.credentials_path.is_file():
                raise CalendarIntegrationUnavailable("Google token is invalid and no OAuth client credentials were provided")
            flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), list(self.SCOPES))
            credentials = flow.run_local_server(port=0)
        if self.token_path:
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(credentials.to_json(), encoding="utf-8")
        self._service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
        return self._service

    def list_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        service = self._build_service()
        response = (
            service.events()
            .list(
                calendarId=self.calendar_id,
                timeMin=_aware(start, self.timezone_name).astimezone(UTC).isoformat(),
                timeMax=_aware(end, self.timezone_name).astimezone(UTC).isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events: list[CalendarEvent] = []
        for raw in response.get("items", []):
            start_data = raw.get("start", {})
            end_data = raw.get("end", {})
            all_day = "date" in start_data
            start_value = start_data.get("dateTime") or start_data.get("date")
            end_value = end_data.get("dateTime") or end_data.get("date")
            if not start_value:
                continue
            parsed_start = _parse_iso(start_value, self.timezone_name)
            parsed_end = _parse_iso(end_value, self.timezone_name)
            if parsed_start is None:
                continue
            if parsed_end is None:
                parsed_end = parsed_start + (timedelta(days=1) if all_day else timedelta(hours=1))
            events.append(
                CalendarEvent(
                    uid=str(raw.get("iCalUID") or raw.get("id") or uuid4()),
                    title=str(raw.get("summary") or "Untitled event"),
                    start=parsed_start,
                    end=parsed_end,
                    description=str(raw.get("description") or ""),
                    location=str(raw.get("location") or ""),
                    all_day=all_day,
                    blocks_time=raw.get("status") != "cancelled" and raw.get("transparency") != "transparent",
                    source="google",
                    recurrence=";".join(raw.get("recurrence", [])) or None,
                    metadata={"google_event_id": raw.get("id"), "html_link": raw.get("htmlLink")},
                )
            )
        return normalize_events(events, self.timezone_name)

    def create_study_block(self, block: StudyBlock | Mapping[str, Any]) -> Mapping[str, Any]:
        value = block if isinstance(block, StudyBlock) else StudyBlock.from_mapping(block, self.timezone_name)
        service = self._build_service()
        body = {
            "summary": value.title,
            "description": value.reason or f"ProofMode Learning Contract for {value.topic}",
            "start": {"dateTime": value.start.isoformat(), "timeZone": self.timezone_name},
            "end": {"dateTime": value.end.isoformat(), "timeZone": self.timezone_name},
            "extendedProperties": {
                "private": {"proofmode_uid": value.uid, "task_id": value.task_id, "topic": value.topic}
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": minute}
                    for minute in sorted({int(item) for item in value.reminders if int(item) > 0}, reverse=True)
                ],
            },
        }
        return service.events().insert(calendarId=self.calendar_id, body=body, sendUpdates="none").execute()

    def create_study_blocks(
        self, blocks: Iterable[StudyBlock | Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        return [self.create_study_block(block) for block in blocks]


__all__ = [
    "AvailabilitySlot",
    "CalendarError",
    "CalendarEvent",
    "CalendarIntegrationUnavailable",
    "DEFAULT_TIMEZONE",
    "DetectedAssessment",
    "GoogleCalendarAdapter",
    "StudyBlock",
    "StudyTask",
    "analyze_calendar",
    "detect_assessments",
    "expand_recurring_events",
    "export_study_blocks_ics",
    "infer_study_availability",
    "normalize_events",
    "parse_ics",
    "schedule_study_blocks",
]
