from __future__ import annotations

from datetime import date, datetime, time, timezone
import unittest

from proofmode.services.calendar_service import (
    CalendarEvent,
    GoogleCalendarAdapter,
    StudyTask,
    detect_assessments,
    export_study_blocks_ics,
    infer_study_availability,
    parse_ics,
    schedule_study_blocks,
)


UTC = timezone.utc


SAMPLE_ICS = """BEGIN:VCALENDAR\r
VERSION:2.0\r
PRODID:-//ProofMode Tests//EN\r
BEGIN:VEVENT\r
UID:lecture-1\r
DTSTART:20260727T100000Z\r
DTEND:20260727T110000Z\r
SUMMARY:Machine Learning lecture\r
DESCRIPTION:Gradient descent and backpropagation\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:exam-1\r
DTSTART;VALUE=DATE:20260807\r
DTEND;VALUE=DATE:20260808\r
SUMMARY:CS404 Final Exam\r
DESCRIPTION:Exam venue MR107\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:deadline-1\r
DTSTART:20260801T160000Z\r
DURATION:PT30M\r
SUMMARY:AI coursework submission deadline\r
DESCRIPTION:Submit the report through MyAberdeen with all source\r
 code attached.\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:transparent-1\r
DTSTART:20260727T120000Z\r
DTEND:20260727T130000Z\r
SUMMARY:Optional event\r
TRANSP:TRANSPARENT\r
END:VEVENT\r
END:VCALENDAR\r
"""


class CalendarParsingTests(unittest.TestCase):
    def test_parse_normalizes_dates_durations_and_folded_lines(self) -> None:
        events = parse_ics(SAMPLE_ICS, "UTC")
        self.assertEqual(4, len(events))
        deadline = next(event for event in events if event.uid == "deadline-1")
        exam = next(event for event in events if event.uid == "exam-1")
        transparent = next(event for event in events if event.uid == "transparent-1")
        self.assertEqual(30, deadline.duration_minutes)
        self.assertIn("sourcecode attached", deadline.description)
        self.assertTrue(exam.all_day)
        self.assertEqual(24 * 60, exam.duration_minutes)
        self.assertFalse(transparent.blocks_time)

    def test_detects_exam_and_deadline_but_not_revision(self) -> None:
        events = parse_ics(SAMPLE_ICS, "UTC")
        events.append(
            CalendarEvent(
                uid="revision",
                title="Exam revision session",
                start=datetime(2026, 8, 2, 10, tzinfo=UTC),
                end=datetime(2026, 8, 2, 11, tzinfo=UTC),
            )
        )
        results = detect_assessments(events, "UTC")
        by_uid = {result.event_uid: result for result in results}
        self.assertEqual("exam", by_uid["exam-1"].kind)
        self.assertEqual("deadline", by_uid["deadline-1"].kind)
        self.assertNotIn("revision", by_uid)
        self.assertGreater(by_uid["exam-1"].confidence, 0.7)


class AvailabilityTests(unittest.TestCase):
    def test_availability_ignores_transparent_and_buffers_busy_event(self) -> None:
        events = parse_ics(SAMPLE_ICS, "UTC")
        slots = infer_study_availability(
            events,
            date(2026, 7, 27),
            date(2026, 7, 27),
            timezone_name="UTC",
            daily_start=time(8),
            daily_end=time(14),
            min_slot_minutes=20,
            buffer_minutes=10,
        )
        self.assertEqual(2, len(slots))
        self.assertEqual(datetime(2026, 7, 27, 8, tzinfo=UTC), slots[0].start)
        self.assertEqual(datetime(2026, 7, 27, 9, 50, tzinfo=UTC), slots[0].end)
        self.assertEqual(datetime(2026, 7, 27, 11, 10, tzinfo=UTC), slots[1].start)
        self.assertEqual(datetime(2026, 7, 27, 14, tzinfo=UTC), slots[1].end)

    def test_weekly_recurrence_blocks_each_occurrence(self) -> None:
        recurring = CalendarEvent(
            uid="weekly",
            title="Tutorial",
            start=datetime(2026, 7, 27, 10, tzinfo=UTC),
            end=datetime(2026, 7, 27, 11, tzinfo=UTC),
            recurrence="FREQ=WEEKLY;COUNT=2",
        )
        slots = infer_study_availability(
            [recurring],
            date(2026, 8, 3),
            date(2026, 8, 3),
            timezone_name="UTC",
            daily_start=time(9),
            daily_end=time(12),
            buffer_minutes=0,
        )
        self.assertEqual(2, len(slots))
        self.assertEqual(time(10), slots[0].end.time())
        self.assertEqual(time(11), slots[1].start.time())


class SchedulingAndExportTests(unittest.TestCase):
    def test_schedule_respects_events_and_prioritizes_urgent_task(self) -> None:
        busy = CalendarEvent(
            uid="busy",
            title="Class",
            start=datetime(2026, 7, 27, 10, tzinfo=UTC),
            end=datetime(2026, 7, 27, 11, tzinfo=UTC),
        )
        tasks = [
            StudyTask("Low priority", 50, priority=1),
            StudyTask(
                "Urgent algorithms",
                50,
                priority=3,
                deadline=datetime(2026, 7, 28, 9, tzinfo=UTC),
            ),
        ]
        blocks = schedule_study_blocks(
            tasks,
            [busy],
            date(2026, 7, 27),
            date(2026, 7, 27),
            timezone_name="UTC",
            daily_start=time(9),
            daily_end=time(13),
            event_buffer_minutes=0,
            break_minutes=5,
        )
        self.assertTrue(blocks)
        self.assertEqual("Urgent algorithms", blocks[0].topic)
        for block in blocks:
            self.assertFalse(block.start < busy.end and block.end > busy.start)

    def test_export_is_round_trippable_and_contains_alarms(self) -> None:
        blocks = schedule_study_blocks(
            [{"name": "Neural networks", "estimated_minutes": 45, "priority": 2}],
            [],
            date(2026, 7, 27),
            date(2026, 7, 27),
            timezone_name="UTC",
            daily_start=time(9),
            daily_end=time(10),
        )
        payload = export_study_blocks_ics(blocks, timezone_name="UTC")
        self.assertTrue(payload.endswith(b"\r\n"))
        self.assertIn(b"BEGIN:VALARM\r\n", payload)
        self.assertIn(b"TRIGGER:-PT15M\r\n", payload)
        imported = parse_ics(payload, "UTC")
        self.assertEqual(1, len(imported))
        self.assertEqual(blocks[0].title, imported[0].title)
        self.assertEqual(45, imported[0].duration_minutes)

    def test_google_adapter_is_safe_without_configuration(self) -> None:
        adapter = GoogleCalendarAdapter(credentials_path="missing.json", token_path="missing-token.json")
        status = adapter.status()
        self.assertFalse(status["configured"])
        self.assertEqual("primary", status["calendar_id"])


if __name__ == "__main__":
    unittest.main()
