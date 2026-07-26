from __future__ import annotations

from datetime import datetime, timedelta


SAMPLE_NOTES = """
Machine Learning Foundations revision outline

Backpropagation
- Forward pass builds a computational graph and produces a loss.
- Reverse-mode automatic differentiation applies the chain rule from the loss backwards.
- Gradients accumulate where a value affects the loss through multiple paths.
- Vanishing gradients can make early layers learn slowly; activation and architecture choices matter.

Bias and variance
- High bias is systematic underfitting; high variance is sensitivity to the training sample.
- Validation data estimates generalisation during model selection.
- Regularisation can reduce variance but too much can increase bias.

Evaluation
- Data leakage makes validation estimates overly optimistic.
- Accuracy can be misleading for imbalanced classes; inspect precision, recall and the cost of errors.
- Cross-validation must preserve the independence of the test set.
""".strip()


def sample_calendar(now: datetime | None = None) -> bytes:
    now = (now or datetime.now().astimezone()).replace(second=0, microsecond=0)
    exam = (now + timedelta(days=12)).replace(hour=14, minute=0)
    lecture = (now + timedelta(days=1)).replace(hour=10, minute=0)
    work = (now + timedelta(days=1)).replace(hour=13, minute=0)
    group = (now + timedelta(days=2)).replace(hour=16, minute=0)

    def stamp(value: datetime) -> str:
        return value.astimezone().strftime("%Y%m%dT%H%M%S")

    content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//ProofMode//Demo Calendar//EN
X-WR-CALNAME:Student calendar
BEGIN:VEVENT
UID:ml-exam-demo
DTSTART;TZID=Europe/London:{stamp(exam)}
DTEND;TZID=Europe/London:{stamp(exam + timedelta(hours=2))}
SUMMARY:CS404 Machine Learning Final Exam
DESCRIPTION:Final written examination. Neural networks, generalisation and evaluation.
END:VEVENT
BEGIN:VEVENT
UID:lecture-demo
DTSTART;TZID=Europe/London:{stamp(lecture)}
DTEND;TZID=Europe/London:{stamp(lecture + timedelta(hours=2))}
SUMMARY:Machine Learning revision lecture
END:VEVENT
BEGIN:VEVENT
UID:work-demo
DTSTART;TZID=Europe/London:{stamp(work)}
DTEND;TZID=Europe/London:{stamp(work + timedelta(hours=3))}
SUMMARY:Part-time work
END:VEVENT
BEGIN:VEVENT
UID:group-demo
DTSTART;TZID=Europe/London:{stamp(group)}
DTEND;TZID=Europe/London:{stamp(group + timedelta(hours=1))}
SUMMARY:Group project meeting
END:VEVENT
END:VCALENDAR
"""
    return content.replace("\n", "\r\n").encode("utf-8")


def fallback_questions(topic: str) -> dict:
    return {
        "mcqs": [
            {
                "question": f"Which activity provides the strongest evidence that you understand {topic}?",
                "options": [
                    "Rereading the heading",
                    "Recognising a familiar definition",
                    "Solving a new example without notes and explaining why",
                    "Copying a worked solution",
                ],
                "correct_index": 2,
                "explanation": "Independent retrieval and transfer provide stronger evidence than familiarity.",
                "skill": "metacognition",
            },
            {
                "question": "What should happen when confidence is high but a retrieval answer is wrong?",
                "options": [
                    "Raise mastery",
                    "Ignore the answer",
                    "Flag a calibration gap and revisit the misconception",
                    "Increase study time without changing the task",
                ],
                "correct_index": 2,
                "explanation": "High-confidence errors are useful signals for targeted correction.",
                "skill": "calibration",
            },
        ],
        "open_question": f"Explain {topic} to a classmate, then apply it to one new example and name one limitation.",
        "open_rubric": ["correct mechanism", "new application", "stated limitation", "clear causal reasoning"],
    }

