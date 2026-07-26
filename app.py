from __future__ import annotations

import html
import hashlib
import json
import re
import secrets
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from proofmode.config import PROJECT_ROOT, settings
from proofmode.database import Database, depth_for_mark
from proofmode.demo_data import SAMPLE_NOTES, fallback_questions, sample_calendar
from proofmode.gemma_client import GemmaClient, GemmaUnavailable, StructuredOutputError
from proofmode.gamification import (
    EvidenceEvent,
    EvidenceKind,
    IntegrityState,
    calculate_proof_score,
    daily_learning_bonus,
)
from proofmode.learning_engine import ambition_gap, update_mastery
from proofmode.services.assessment_service import assess_learning_receipt, generate_questions
from proofmode.services.calendar_service import (
    CalendarIntegrationUnavailable,
    CalendarEvent,
    GoogleCalendarAdapter,
    StudyBlock,
    detect_assessments,
    export_study_blocks_ics,
    parse_ics,
    schedule_study_blocks,
)
from proofmode.services.document_service import StudyDocument, make_document
from proofmode.services.intervention_service import choose_intervention, fallback_intervention
from proofmode.services.planner_service import (
    extract_course_map,
    fallback_course_map,
    learning_contract,
    prioritise_topics,
)
from proofmode.services.research_service import build_evidence_prompt, prepare_research_pack
from proofmode.services.teachback_service import (
    generate_transfer_pair,
    score_teaching_explanation,
    score_transfer,
    teaching_impact,
)
from proofmode.services.verification_service import verify_answer


st.set_page_config(
    page_title="ProofMode · Gemma 4 study companion",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = PROJECT_ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_database() -> Database:
    return Database()


@st.cache_resource
def get_gemma() -> GemmaClient:
    database = get_database()

    def audit(action: str, latency_ms: int, modality: str, payload: dict[str, Any]) -> None:
        database.add_audit(action, settings.model_name, latency_ms, modality, payload)

    return GemmaClient(audit_callback=audit)


def initialise_state() -> None:
    defaults: dict[str, Any] = {
        "course_title": "",
        "exam_summary": "",
        "exam_date": None,
        "topics": [],
        "calendar_events": [],
        "assessments": [],
        "study_blocks": [],
        "documents": [],
        "questions": {},
        "receipt_result": None,
        "intervention": None,
        "research_answer": None,
        "research_pack": None,
        "verification": None,
        "teach_pair": None,
        "teach_step": 1,
        "teach_pre": None,
        "teach_quality": None,
        "teach_result": None,
        "plan_source": "none",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def evidence_payload(event: EvidenceEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["kind"] = event.kind.value
    payload["integrity_state"] = event.integrity_state.value
    payload["occurred_at"] = event.occurred_at.isoformat()
    return payload


def record_proof_event(event: EvidenceEvent) -> None:
    get_database().execute(
        "INSERT INTO proof_events (student_name, topic, event_type, integrity_status, evidence_text, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            event.learner_id,
            event.topic_id,
            event.kind.value,
            event.integrity_state.value,
            event.response_text,
            safe_json(evidence_payload(event)),
            event.occurred_at.isoformat(),
        ),
    )


def load_proof_events(learner_id: str | None = None) -> list[EvidenceEvent]:
    if learner_id:
        rows = get_database().query("SELECT payload FROM proof_events WHERE student_name = ? ORDER BY id", (learner_id,))
    else:
        rows = get_database().query("SELECT payload FROM proof_events ORDER BY id")
    events: list[EvidenceEvent] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
            payload["occurred_at"] = datetime.fromisoformat(payload["occurred_at"])
            events.append(EvidenceEvent(**payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return events


def prompt_fingerprint(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def persist_plan(topics: list[dict[str, Any]], events: list[CalendarEvent], blocks: list[StudyBlock]) -> None:
    database = get_database()
    database.execute("DELETE FROM topics")
    for topic in topics:
        database.save_topic(topic, topic.get("priority", 0))
    database.execute("DELETE FROM calendar_events")
    for event in events:
        database.execute(
            "INSERT OR REPLACE INTO calendar_events (id, title, start_at, end_at, payload) VALUES (?, ?, ?, ?, ?)",
            (event.uid, event.title, event.start.isoformat(), event.end.isoformat(), safe_json(event.to_dict())),
        )
    database.execute("DELETE FROM study_blocks")
    for block in blocks:
        database.execute(
            "INSERT OR REPLACE INTO study_blocks (id, topic, start_at, end_at, contract, status, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                block.uid,
                block.topic,
                block.start.isoformat(),
                block.end.isoformat(),
                block.reason,
                "planned",
                safe_json(block.to_dict()),
            ),
        )


def uploaded_documents(files: list[Any] | None) -> list[StudyDocument]:
    documents: list[StudyDocument] = []
    for file in files or []:
        documents.append(make_document(file.name, file.getvalue(), getattr(file, "type", None)))
    if not documents:
        documents.append(make_document("demo_revision_notes.md", SAMPLE_NOTES.encode("utf-8"), "text/markdown"))
    return documents


def next_exam(assessments: list[Any]) -> datetime:
    now = datetime.now().astimezone()
    future = [item for item in assessments if item.when > now]
    exams = [item.when for item in future if item.kind.lower() == "exam"]
    if exams:
        return min(exams)
    credible_deadlines = [item.when for item in future if item.confidence >= 0.7]
    return min(credible_deadlines) if credible_deadlines else now + timedelta(days=14)


CREATE_BLOCK_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "create_study_block",
            "description": "Approve a deterministic calendar slot and attach a specific learning-proof contract.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["topic", "start", "end", "contract"],
                "properties": {
                    "topic": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "contract": {"type": "string"},
                },
            },
        },
    }
]


def run_autopilot(target_mark: float, note_files: list[Any] | None, calendar_file: Any | None) -> tuple[bool, str]:
    database = get_database()
    client = get_gemma()
    existing_profile = database.one("SELECT name FROM students WHERE id = 1") or {}
    profile = database.set_profile(target_mark, str(existing_profile.get("name") or "Student"))
    depth_mode = profile.get("depth_mode", depth_for_mark(target_mark))
    calendar_bytes = calendar_file.getvalue() if calendar_file else sample_calendar()
    events = parse_ics(calendar_bytes)
    assessments = detect_assessments(events)
    exam_date = next_exam(assessments)
    documents = uploaded_documents(note_files)
    context = "\n".join(
        f"{item.kind}: {item.title} at {item.when.isoformat()} (heuristic confidence {item.confidence:.0%})"
        for item in assessments
    )
    used_fallback = False
    try:
        course_map = extract_course_map(client, documents, context, target_mark, depth_mode)
    except (GemmaUnavailable, StructuredOutputError, ValueError) as error:
        course_map = fallback_course_map()
        used_fallback = True
        fallback_reason = str(error)
    topics = prioritise_topics(course_map.get("topics", []), exam_date, depth_mode)
    for topic in topics:
        topic["exam_date"] = exam_date.isoformat()
        topic["reason"] = learning_contract(topic, depth_mode)
    planning_end = max(exam_date, datetime.now(exam_date.tzinfo) + timedelta(days=2))
    blocks = schedule_study_blocks(
        topics,
        events,
        datetime.now().astimezone(),
        planning_end,
        max_block_minutes=50,
        min_block_minutes=20,
        break_minutes=10,
    )
    # The scheduler owns time arithmetic. Gemma's function call supplies the
    # pedagogical contract, and its arguments are treated as an untrusted proposal.
    if blocks and not used_fallback:
        top = blocks[0]
        try:
            call = client.choose_tool(
                "Use the one allowed function. Keep the deterministic start/end exactly as provided and make the contract observable and closed-book.",
                f"Topic={top.topic}; start={top.start.isoformat()}; end={top.end.isoformat()}; proposed contract={top.reason}",
                CREATE_BLOCK_TOOL,
            )
            if call.tool_calls and call.tool_calls[0]["name"] == "create_study_block":
                args = call.tool_calls[0]["arguments"]
                if args.get("topic") == top.topic and isinstance(args.get("contract"), str):
                    blocks[0] = replace(top, reason=args["contract"][:500])
        except GemmaUnavailable:
            pass
    persist_plan(topics, events, blocks)
    st.session_state.update(
        {
            "course_title": course_map.get("course_title", "Personal study plan"),
            "exam_summary": course_map.get("exam_summary", ""),
            "exam_date": exam_date,
            "topics": topics,
            "calendar_events": events,
            "assessments": assessments,
            "study_blocks": blocks,
            "documents": documents,
            "questions": {},
            "receipt_result": None,
            "plan_source": "fallback" if used_fallback else "gemma",
        }
    )
    if used_fallback:
        return False, f"A robust demo plan was loaded because Gemma was unavailable: {fallback_reason}"
    return True, f"Gemma mapped {len(topics)} topics and ProofMode scheduled {len(blocks)} proof contracts."


def topic_by_name(name: str) -> dict[str, Any]:
    return next((topic for topic in st.session_state.topics if topic["name"] == name), {})


def render_sidebar() -> tuple[float, list[Any] | None, Any | None]:
    client = get_gemma()
    database = get_database()
    profile = database.one("SELECT * FROM students WHERE id = 1") or {}
    with st.sidebar:
        st.markdown("## ◉ ProofMode")
        available = client.available()
        status_class = "" if available else " offline"
        status_text = "Gemma 4 E4B ready" if available else "Gemma server offline"
        st.markdown(f'<span class="status-pill{status_class}">● {status_text}</span>', unsafe_allow_html=True)
        st.caption("Local, private, evidence-first")
        st.divider()
        st.markdown("#### Minimum setup")
        target_mark = st.slider("Desired mark", 40, 100, int(profile.get("target_mark", 85)), 1, help="ProofMode derives learning depth and workload from this.")
        derived = depth_for_mark(target_mark).replace("_", " → ")
        st.caption(f"Derived mode: **{derived}**")
        note_files = st.file_uploader(
            "Notes or syllabus",
            type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="Images are read directly by local Gemma 4. PDF and document text is extracted locally.",
        )
        calendar_file = st.file_uploader("Calendar export (.ics)", type=["ics"], help="Exam dates and free time are inferred automatically.")
        if st.button("Build my autopilot plan", type="primary", width="stretch"):
            with st.spinner("Gemma is mapping topics; ProofMode is resolving your calendar…"):
                ok, message = run_autopilot(target_mark, note_files, calendar_file)
            (st.success if ok else st.warning)(message)
        st.caption("No files yet? The button runs a complete, clearly labelled demo course.")
        with st.expander("Direct Google Calendar (optional)"):
            credentials_path = PROJECT_ROOT / "credentials.json"
            token_path = PROJECT_ROOT / "token.json"
            google = GoogleCalendarAdapter(credentials_path=credentials_path, token_path=token_path)
            if not google.dependencies_available:
                st.caption("Google SDK dependencies are unavailable; `.ics` remains fully functional.")
            elif not google.is_configured:
                st.caption("Place a Desktop OAuth client file at `credentials.json`, then connect. No key is committed.")
            else:
                st.caption("OAuth configuration detected. Access is limited to calendar events.")
            if st.button("Connect & read 30 days", disabled=not google.dependencies_available or not google.is_configured):
                try:
                    now = datetime.now().astimezone()
                    imported = google.list_events(now, now + timedelta(days=30))
                    st.session_state.calendar_events = imported
                    st.session_state.assessments = detect_assessments(imported)
                    st.success(f"Read {len(imported)} events from Google Calendar.")
                except CalendarIntegrationUnavailable as error:
                    st.error(str(error))
            if st.button("Push proof contracts", disabled=not bool(st.session_state.study_blocks) or not google.is_configured):
                try:
                    created = google.create_study_blocks(st.session_state.study_blocks)
                    st.success(f"Created {len(created)} Google Calendar events with reminders.")
                except CalendarIntegrationUnavailable as error:
                    st.error(str(error))
        st.divider()
        st.markdown("**Privacy boundary**")
        st.caption("Notes and assessment records stay on this device. Web research sends only the search question to public search providers.")
    return float(target_mark), note_files, calendar_file


def render_hero() -> None:
    title = st.session_state.course_title or "Your plan should react to proof."
    subtitle = (
        "ProofMode turns calendar intentions into observable learning, then re-plans from retrieval, confidence, notes and peer teaching—not time logged."
    )
    st.markdown(
        f'<div class="proof-hero"><div class="proof-kicker">Gemma 4 · local learning twin</div><h1>{html.escape(title)}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_cockpit(target_mark: float) -> None:
    topics = st.session_state.topics
    blocks = st.session_state.study_blocks
    exam_date = st.session_state.exam_date
    if not topics or not exam_date:
        st.info("Use **Build my autopilot plan**. With no uploads, ProofMode creates a complete demo in one click.")
        return
    average_mastery = sum(float(topic.get("mastery", 0)) for topic in topics) / len(topics)
    required = sum(int(topic.get("estimated_minutes", 45)) for topic in topics)
    available = sum(block.duration_minutes for block in blocks)
    gap = ambition_gap(target_mark, available, required)
    days = max(0, (exam_date.date() - datetime.now().date()).days)
    cols = st.columns(4)
    cols[0].metric("Target", f"{target_mark:.0f}%", depth_for_mark(target_mark).replace("_", " "))
    cols[1].metric("Evidence mastery", f"{average_mastery:.0%}", "not a predicted grade")
    cols[2].metric("Exam horizon", f"{days} days", exam_date.strftime("%d %b · %H:%M"))
    cols[3].metric("Proof contracts", len(blocks), f"{available} min scheduled")
    if gap["status"] == "overcommitted":
        st.warning(gap["message"])
    else:
        st.success(gap["message"])

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Risk-weighted topic map")
        chart = pd.DataFrame(
            {
                "topic": [topic["name"] for topic in topics],
                "priority": [topic["priority"] for topic in topics],
                "mastery": [round(float(topic.get("mastery", 0)) * 100, 1) for topic in topics],
            }
        ).set_index("topic")
        st.bar_chart(chart[["priority", "mastery"]], horizontal=True, color=["#2764FF", "#00A88F"])
        with st.expander("Why these priorities?"):
            st.write("Exam weight × mastery gap × forgetting risk × prerequisite importance × target depth × urgency, adjusted for estimated effort. Gemma proposes factors; deterministic code computes the rank.")
            st.dataframe(
                [
                    {
                        "Topic": topic["name"],
                        "Priority": topic["priority"],
                        "Difficulty": f"{topic.get('difficulty', 0):.0%}",
                        "Uncertainty": topic.get("uncertainty", ""),
                    }
                    for topic in topics
                ],
                hide_index=True,
                width="stretch",
            )
    with right:
        st.markdown("### Next calendar contracts")
        for block in blocks[:5]:
            st.markdown(
                f'<div class="contract-card"><strong>{html.escape(block.topic)}</strong><br>{block.start.strftime("%a %d %b · %H:%M")}–{block.end.strftime("%H:%M")}<br><span class="small-muted">{html.escape(block.reason)}</span></div>',
                unsafe_allow_html=True,
            )
        if blocks:
            st.download_button(
                "Add plan to any calendar (.ics)",
                data=export_study_blocks_ics(blocks),
                file_name="proofmode-study-contracts.ics",
                mime="text/calendar",
                width="stretch",
            )


def fallback_receipt(answer: str, mcq_score: float) -> dict[str, Any]:
    words = len(answer.split())
    open_score = min(0.8, 0.25 + words / 120)
    return {
        "correctness": round((mcq_score + open_score) / 2, 2),
        "depth": round(open_score, 2),
        "evidence_quality": 0.35,
        "coverage": ["Fallback assessment used"],
        "strengths": ["A learning attempt was recorded"],
        "misconceptions": [],
        "feedback": "Gemma was unavailable, so this receipt is provisional and should be reassessed when the local model is running.",
        "next_probe": "Explain the mechanism with one concrete example.",
        "uncertainty": "High — provisional fallback only.",
    }


def reschedule_after_receipt(topic_name: str, new_mastery: float) -> None:
    exam_date = st.session_state.exam_date
    if not exam_date:
        return
    topics = st.session_state.topics
    depth = depth_for_mark((get_database().one("SELECT target_mark FROM students WHERE id=1") or {}).get("target_mark", 70))
    updated = prioritise_topics(topics, exam_date, depth)
    st.session_state.topics = updated
    # Preserve busy imported events, then rebuild remaining contracts. Higher
    # mastery reduces remaining minutes for this topic without erasing review.
    tasks = []
    for topic in updated:
        item = dict(topic)
        remaining_factor = max(0.2, 1 - float(item.get("mastery", 0)))
        item["estimated_minutes"] = max(20, round(int(item.get("estimated_minutes", 45)) * remaining_factor))
        item["reason"] = learning_contract(item, depth)
        item["exam_date"] = exam_date.isoformat()
        tasks.append(item)
    blocks = schedule_study_blocks(tasks, st.session_state.calendar_events, datetime.now().astimezone(), exam_date, max_block_minutes=50)
    st.session_state.study_blocks = blocks
    persist_plan(updated, st.session_state.calendar_events, blocks)


def render_focus(target_mark: float) -> None:
    if not st.session_state.topics:
        st.info("Build a plan first so ProofMode can issue a learning contract.")
        return
    topic_names = [topic["name"] for topic in st.session_state.topics]
    selected = st.selectbox("Focus topic", topic_names, key="focus_topic")
    topic = topic_by_name(selected)
    profile = get_database().one("SELECT * FROM students WHERE id=1") or {}
    contract = learning_contract(topic, profile.get("depth_mode", "apply"))
    st.markdown(
        f'<div class="contract-card"><strong>Today’s proof contract</strong><br>{html.escape(contract)}</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Generate retrieval check", type="primary", width="stretch"):
            with st.spinner("Gemma is generating misconception-sensitive questions…"):
                try:
                    st.session_state.questions[selected] = generate_questions(get_gemma(), topic, target_mark, profile.get("depth_mode", "apply"))
                except (GemmaUnavailable, StructuredOutputError):
                    st.session_state.questions[selected] = fallback_questions(selected)
    with c2:
        st.metric("Current mastery evidence", f"{float(topic.get('mastery', 0)):.0%}", "updated only by proof")

    questions = st.session_state.questions.get(selected)
    if not questions:
        st.caption("Generate the check before looking at notes. The answer key stays hidden until submission.")
    else:
        with st.form("learning_receipt_form"):
            st.markdown("#### Closed-book retrieval")
            answers: list[int] = []
            for index, question in enumerate(questions["mcqs"]):
                choice = st.radio(
                    question["question"],
                    options=list(range(len(question["options"]))),
                    format_func=lambda value, options=question["options"]: options[value],
                    key=f"mcq_{selected}_{index}",
                    index=None,
                )
                answers.append(-1 if choice is None else int(choice))
            st.markdown("#### Transfer proof")
            st.write(questions["open_question"])
            open_answer = st.text_area("Your reasoning", height=140, placeholder="Explain without copying from the notes…")
            confidence = st.slider("How confident are you?", 0, 100, 60, help="Calibration matters: confidence is compared with correctness.")
            receipt_image = st.file_uploader("Optional photo of notes or worked steps", type=["png", "jpg", "jpeg", "webp"], key="receipt_image")
            submitted = st.form_submit_button("Submit Learning Receipt", type="primary", width="stretch")
        if submitted:
            mcq_correct = sum(
                answer == question["correct_index"]
                for answer, question in zip(answers, questions["mcqs"])
            )
            mcq_score = mcq_correct / max(1, len(questions["mcqs"]))
            try:
                with st.spinner("Gemma is checking concepts, reasoning and misconception risk…"):
                    assessment = assess_learning_receipt(
                        get_gemma(),
                        topic,
                        questions["open_question"],
                        open_answer,
                        questions["open_rubric"],
                        receipt_image.getvalue() if receipt_image else None,
                        receipt_image.type if receipt_image else None,
                    )
            except (GemmaUnavailable, StructuredOutputError):
                assessment = fallback_receipt(open_answer, mcq_score)
            correctness = 0.45 * mcq_score + 0.55 * float(assessment["correctness"])
            metrics = update_mastery(
                float(topic.get("mastery", 0)),
                correctness,
                float(assessment["depth"]),
                float(assessment["evidence_quality"]),
                confidence / 100,
            )
            assessment["metrics"] = metrics
            assessment["mcq_score"] = mcq_score
            assessment["confidence"] = confidence / 100
            topic.update({"mastery": metrics["mastery"], "confidence": confidence / 100})
            get_database().execute(
                "INSERT INTO assessments (topic, kind, score, confidence, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (selected, "learning_receipt", correctness, confidence / 100, safe_json(assessment), datetime.now().isoformat()),
            )
            learner = str(profile.get("name") or "Student")
            occurred = datetime.now().astimezone()
            receipt_id = secrets.token_hex(6)
            record_proof_event(
                EvidenceEvent(
                    event_id=f"{receipt_id}-retention",
                    learner_id=learner,
                    topic_id=selected,
                    kind=EvidenceKind.RETENTION,
                    score=mcq_score,
                    occurred_at=occurred,
                    difficulty=float(topic.get("difficulty", 0.5)),
                    confidence=confidence / 100,
                    delay_hours=0,
                    prompt_id=prompt_fingerprint(" | ".join(item["question"] for item in questions["mcqs"])),
                )
            )
            record_proof_event(
                EvidenceEvent(
                    event_id=f"{receipt_id}-transfer",
                    learner_id=learner,
                    topic_id=selected,
                    kind=EvidenceKind.TRANSFER,
                    score=float(assessment["depth"]),
                    occurred_at=occurred,
                    difficulty=float(topic.get("difficulty", 0.5)),
                    confidence=confidence / 100,
                    prompt_id=prompt_fingerprint(questions["open_question"]),
                    response_text=open_answer,
                )
            )
            reschedule_after_receipt(selected, metrics["mastery"])
            st.session_state.receipt_result = assessment
            st.success("Receipt accepted. Mastery evidence and future calendar blocks were updated.")
    result = st.session_state.receipt_result
    if result:
        st.markdown("### Receipt result")
        metrics = result["metrics"]
        cols = st.columns(4)
        cols[0].metric("Knowledge", f"{metrics['knowledge']:.0%}")
        cols[1].metric("Depth", f"{metrics['depth']:.0%}")
        cols[2].metric("Calibration", f"{metrics['calibration']:.0%}")
        cols[3].metric("Evidence", f"{metrics['evidence_quality']:.0%}")
        st.write(result["feedback"])
        if result.get("misconceptions"):
            st.warning("Misconception watch: " + "; ".join(result["misconceptions"]))
        st.caption("Uncertainty: " + result.get("uncertainty", "Not reported"))

    st.divider()
    st.markdown("### Missed the block? Run a recovery, not a guilt loop.")
    friction = st.radio(
        "What blocked you?",
        ["overwhelmed", "confused", "tired", "distracted", "time conflict"],
        index=0,
        horizontal=True,
        key="friction",
    )
    if st.button("Demo time machine: mark missed & recover"):
        try:
            intervention = choose_intervention(get_gemma(), topic, friction or "overwhelmed")
        except GemmaUnavailable:
            intervention = fallback_intervention(topic, friction or "overwhelmed")
        st.session_state.intervention = intervention
    if st.session_state.intervention:
        action = st.session_state.intervention
        args = action["arguments"]
        st.success(f"{action['name'].replace('_', ' ').title()}: {args.get('first_action', '')}")
        st.caption(args.get("reason", ""))


RESEARCH_ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["search_query", "preferred_domain", "risk"],
    "properties": {
        "search_query": {"type": "string"},
        "preferred_domain": {"type": "string"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}


def research_route(question: str) -> dict[str, Any]:
    try:
        result = get_gemma().structured(
            "You prepare a web search for a fact-checking tutor. search_query must contain at most 10 precise terms, never a full question or generic 'best practices'. preferred_domain is one likely primary/official domain such as scikit-learn.org, ai.google.dev or pubmed.ncbi.nlm.nih.gov; use an empty string only if genuinely unknown. Mark risk high for medical, legal, financial, current, statistical, or safety-sensitive questions. Keep every JSON value extremely short.",
            f"Student question: {question}",
            RESEARCH_ROUTE_SCHEMA,
            schema_name="research_route",
            max_tokens=450,
        )
        return result.payload or {"search_query": question, "preferred_domain": "", "risk": "medium"}
    except (GemmaUnavailable, StructuredOutputError):
        return {"search_query": question, "preferred_domain": "", "risk": "medium"}


def render_research() -> None:
    st.markdown("### Evidence-backed tutor")
    st.write("Gemma prepares a bounded research pack, teaches from those sources, then a second pass checks every cited claim. No model weights are changed.")
    question = st.text_area("What should Gemma teach or verify?", placeholder="Why can cross-validation still leak test information?", height=100)
    if st.button("Research, teach & verify", type="primary", disabled=not question.strip()):
        route = research_route(question)
        with st.spinner(f"Preparing {route['risk']}-risk evidence sources…"):
            candidate_domain = str(route.get("preferred_domain", "")).lower().removeprefix("www.")
            preferred = candidate_domain if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", candidate_domain) else ""
            focused_query = f"site:{preferred} {route['search_query']}" if preferred else route["search_query"]
            pack = prepare_research_pack(focused_query, max_results=5)
            if not pack.sources and preferred:
                pack = prepare_research_pack(route["search_query"], max_results=5)
        st.session_state.research_pack = pack
        if not pack.sources:
            st.session_state.research_answer = None
            st.session_state.verification = None
            st.warning("Live sources were unavailable, so ProofMode refused to improvise a factual answer.")
        else:
            prompt = build_evidence_prompt(question, pack, max_chars_per_source=3500)
            try:
                with st.spinner("Gemma is teaching from the research pack…"):
                    answer = get_gemma().chat(
                        "You are an evidence-grounded tutor. Follow the evidence rules exactly. Use clear explanations, but never add uncited factual material.",
                        prompt,
                        max_tokens=1100,
                    ).content

                def verifier_callback(verification_prompt: str) -> str:
                    return get_gemma().chat(
                        "You are a strict claim verifier. Return only the requested JSON; never use outside knowledge.",
                        verification_prompt,
                        max_tokens=900,
                    ).content

                with st.spinner("A separate Gemma pass is checking claim support…"):
                    report = verify_answer(answer, pack, llm_callback=verifier_callback)
                if not report.safe_to_show:
                    repair_prompt = (
                        build_evidence_prompt(question, pack, max_chars_per_source=2500)
                        + "\n\nHELD DRAFT\n"
                        + answer
                        + "\n\nVERIFIER REPORT\n"
                        + safe_json(report.as_dict())
                        + "\n\nRewrite the answer once. Remove unsupported claims, qualify uncertain ones as unverified, and preserve only citations that directly support each sentence. Do not discuss this repair process."
                    )
                    with st.spinner("Unsupported claims found · Gemma is repairing from evidence only…"):
                        answer = get_gemma().chat(
                            "You repair evidence-grounded answers. Use only the research pack, remove unsupported material, and cite every remaining factual claim.",
                            repair_prompt,
                            max_tokens=1000,
                        ).content
                        report = verify_answer(answer, pack, llm_callback=verifier_callback)
                st.session_state.research_answer = answer
                st.session_state.verification = report
            except GemmaUnavailable as error:
                st.error(str(error))
    pack = st.session_state.research_pack
    answer = st.session_state.research_answer
    report = st.session_state.verification
    if pack:
        st.caption(f"Research status: {pack.status} · {pack.method}")
        for warning in pack.warnings:
            st.caption("⚠ " + warning)
    if answer and report:
        if report.safe_to_show:
            st.success(report.summary)
            st.markdown(answer)
        else:
            st.error(report.summary)
            st.write("ProofMode held the answer instead of presenting uncertain claims as fact.")
            with st.expander("Inspect held draft and verifier report"):
                st.markdown(answer)
                st.json(report.as_dict())
    if pack and pack.sources:
        with st.expander("Inspect research pack", expanded=bool(answer)):
            for source in pack.sources:
                st.markdown(f"**[{source.source_id}] [{source.title}]({source.url})** · `{source.domain}` · authority signal {source.authority_score:.1f}")
                st.caption(source.snippet[:500] or source.text[:500])


def render_teachback() -> None:
    if not st.session_state.topics:
        st.info("Build a plan first; Teach-Back uses a weak topic from the learner model.")
        return
    st.markdown("### Teach-Back Arena")
    st.write("Teaching Impact is earned only when the learner improves on a new transfer question—not for popularity or message volume.")
    name_cols = st.columns(2)
    teacher_name = name_cols[0].text_input("Teacher nickname", value="Sam", key="teacher_name")
    learner_name = name_cols[1].text_input("Learner nickname", value="Alex", key="learner_name")
    topic_name = st.selectbox("Arena topic", [topic["name"] for topic in st.session_state.topics], key="teach_topic")
    topic = topic_by_name(topic_name)
    if st.button("Create fair pre/post challenge", type="primary"):
        try:
            st.session_state.teach_pair = generate_transfer_pair(get_gemma(), topic)
        except (GemmaUnavailable, StructuredOutputError):
            st.session_state.teach_pair = {
                "pre_question": f"Apply {topic_name} to one concrete scenario and justify your choice.",
                "post_question": f"Apply the same principle behind {topic_name} to a different scenario and identify a limitation.",
                "rubric": ["correct principle", "application", "causal justification", "limitation"],
                "concept_invariant": topic_name,
            }
        st.session_state.teach_step = 1
        st.session_state.teach_pre = None
        st.session_state.teach_quality = None
        st.session_state.teach_result = None
    pair = st.session_state.teach_pair
    if not pair:
        return
    if st.session_state.teach_step == 1:
        st.markdown("#### 1 · Learner baseline")
        st.write(pair["pre_question"])
        pre_answer = st.text_area("Learner’s private answer", key="teach_pre_answer")
        if st.button("Lock baseline", disabled=not pre_answer.strip()):
            try:
                st.session_state.teach_pre = score_transfer(get_gemma(), pair["pre_question"], pre_answer, pair["rubric"])
            except (GemmaUnavailable, StructuredOutputError):
                st.session_state.teach_pre = {"score": min(0.75, len(pre_answer.split()) / 80), "feedback": "Provisional offline score"}
            st.session_state.teach_step = 2
            st.rerun()
    elif st.session_state.teach_step == 2:
        st.markdown("#### 2 · Friend teaches")
        explanation = st.text_area("Friend’s explanation", key="teach_explanation", height=150, placeholder="Explain mechanism, example and likely misconception…")
        if st.button("Share explanation", disabled=not explanation.strip()):
            try:
                st.session_state.teach_quality = score_teaching_explanation(get_gemma(), topic_name, explanation)
            except (GemmaUnavailable, StructuredOutputError):
                st.session_state.teach_quality = {"score": min(0.75, len(explanation.split()) / 100), "feedback": "Provisional offline score"}
            st.session_state.teach_explanation_text = explanation
            st.session_state.teach_step = 3
            st.rerun()
    else:
        st.markdown("#### 3 · New transfer, same underlying skill")
        st.write(pair["post_question"])
        post_answer = st.text_area("Learner’s new answer", key="teach_post_answer")
        if st.button("Measure Teaching Impact", disabled=not post_answer.strip()):
            try:
                post = score_transfer(get_gemma(), pair["post_question"], post_answer, pair["rubric"])
            except (GemmaUnavailable, StructuredOutputError):
                post = {"score": min(0.9, len(post_answer.split()) / 80), "feedback": "Provisional offline score"}
            pre = st.session_state.teach_pre or {"score": 0}
            quality = st.session_state.teach_quality or {"score": 0}
            impact = teaching_impact(float(pre["score"]), float(post["score"]), float(quality["score"]))
            result = {"pre": pre, "post": post, "quality": quality, "impact": impact}
            st.session_state.teach_result = result
            room = secrets.token_hex(3).upper()
            record_proof_event(
                EvidenceEvent(
                    event_id=f"teach-{room}",
                    learner_id=teacher_name.strip() or "Teacher",
                    topic_id=topic_name,
                    kind=EvidenceKind.TEACHBACK,
                    score=float(post["score"]),
                    occurred_at=datetime.now().astimezone(),
                    difficulty=float(topic.get("difficulty", 0.5)),
                    prompt_id=prompt_fingerprint(pair["concept_invariant"] + pair["post_question"]),
                    response_text=st.session_state.get("teach_explanation_text", ""),
                    partner_id=learner_name.strip() or "Learner",
                    pre_score=float(pre["score"]),
                    post_score=float(post["score"]),
                    teaching_quality=float(quality["score"]),
                )
            )
            get_database().execute(
                "INSERT INTO teaching_scores (room_code, teacher_name, topic, pre_score, post_score, impact, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (room, teacher_name.strip() or "Teacher", topic_name, float(pre["score"]), float(post["score"]), impact["impact"], safe_json(result), datetime.now().isoformat()),
            )
        result = st.session_state.teach_result
        if result:
            impact = result["impact"]
            cols = st.columns(4)
            cols[0].metric("Before", f"{impact['pre']:.0f}%")
            cols[1].metric("After", f"{impact['post']:.0f}%")
            cols[2].metric("Transfer gain", f"{impact['gain']:+.0f} pts")
            cols[3].metric("Teaching Impact", f"{impact['impact']:.0f}")
            if impact["gain"] > 0:
                st.success("The learner improved on a new problem. Teaching Impact earned.")
            else:
                st.info("No impact points yet. ProofMode rewards learning lift, not activity volume.")
    leaderboard = get_database().query("SELECT teacher_name, topic, impact, created_at FROM teaching_scores ORDER BY impact DESC LIMIT 8")
    if leaderboard:
        st.markdown("#### Opt-in Teaching Impact board")
        st.dataframe(leaderboard, hide_index=True, width="stretch")


def demo_league_events(learner: str, strength: float, topics: list[dict[str, Any]], seed: int) -> list[EvidenceEvent]:
    """Transparent in-memory examples so judges can see a populated fair league."""
    now = datetime.now().astimezone()
    selected = (topics or fallback_course_map()["topics"])[:3]
    while len(selected) < 2:
        selected.append(dict(selected[0]))
    values = [max(0.2, min(0.98, strength - 0.04)), max(0.2, min(0.98, strength + 0.03))]
    events: list[EvidenceEvent] = []
    for index, topic in enumerate(selected[:2]):
        occurred = now - timedelta(days=4 - index)
        events.append(
            EvidenceEvent(
                event_id=f"demo-{seed}-{index}-r",
                learner_id=learner,
                topic_id=topic["name"],
                kind=EvidenceKind.RETENTION,
                score=values[index],
                occurred_at=occurred,
                difficulty=float(topic.get("difficulty", 0.5)),
                confidence=max(0.0, values[index] - 0.03),
                delay_hours=24 + 24 * index,
                prompt_id=f"demo-delayed-{seed}-{index}",
                response_text=f"Independent delayed retrieval example {seed} {index} with a distinct explanation of {topic['name']}.",
            )
        )
        events.append(
            EvidenceEvent(
                event_id=f"demo-{seed}-{index}-t",
                learner_id=learner,
                topic_id=topic["name"],
                kind=EvidenceKind.TRANSFER,
                score=max(0.2, values[index] - 0.05),
                occurred_at=occurred + timedelta(hours=1),
                difficulty=float(topic.get("difficulty", 0.5)),
                confidence=values[index],
                prompt_id=f"demo-transfer-{seed}-{index}",
                response_text=f"Distinct isomorphic transfer response {seed} {index} applying {topic['name']} in a new scenario.",
            )
        )
    events.append(
        EvidenceEvent(
            event_id=f"demo-{seed}-teach",
            learner_id=learner,
            topic_id=selected[0]["name"],
            kind=EvidenceKind.TEACHBACK,
            score=strength,
            occurred_at=now - timedelta(days=1),
            partner_id=f"Learner-{seed}",
            pre_score=max(0.1, strength - 0.35),
            post_score=max(0.2, strength - 0.05),
            teaching_quality=strength,
            prompt_id=f"demo-teach-{seed}",
            response_text=f"Unique teaching explanation {seed} with mechanism, example, and misconception check.",
        )
    )
    return events


def rename_learner(old_name: str, new_name: str) -> None:
    database = get_database()
    new_name = new_name.strip() or "Student"
    if old_name == new_name:
        return
    rows = database.query("SELECT id, payload FROM proof_events WHERE student_name = ?", (old_name,))
    for row in rows:
        try:
            payload = json.loads(row["payload"])
            payload["learner_id"] = new_name
            database.execute(
                "UPDATE proof_events SET student_name = ?, payload = ? WHERE id = ?",
                (new_name, safe_json(payload), row["id"]),
            )
        except (json.JSONDecodeError, KeyError):
            continue
    database.execute("UPDATE teaching_scores SET teacher_name = ? WHERE teacher_name = ?", (new_name, old_name))
    database.execute("UPDATE students SET name = ? WHERE id = 1", (new_name,))


def schedule_delayed_checks(topics: list[dict[str, Any]]) -> None:
    if not topics:
        return
    now = datetime.now().astimezone()
    new_blocks: list[StudyBlock] = []
    for index, topic in enumerate(topics[:2], 1):
        start = (now + timedelta(days=index)).replace(hour=18, minute=0, second=0, microsecond=0)
        new_blocks.append(
            StudyBlock(
                uid=f"proofmode-delayed-{secrets.token_hex(4)}@local",
                topic=topic["name"],
                title=f"ProofMode · delayed proof · {topic['name']}",
                start=start,
                end=start + timedelta(minutes=20),
                priority=float(topic.get("priority", 1)),
                reason="Fresh closed-book retrieval and one new transfer question; required for leaderboard verification.",
                task_id=f"delayed-{prompt_fingerprint(topic['name'])}",
            )
        )
    st.session_state.study_blocks = sorted([*st.session_state.study_blocks, *new_blocks], key=lambda block: block.start)
    persist_plan(st.session_state.topics, st.session_state.calendar_events, st.session_state.study_blocks)


def render_league() -> None:
    database = get_database()
    profile = database.one("SELECT * FROM students WHERE id=1") or {"name": "Student"}
    current_name = str(profile.get("name") or "Student")
    st.markdown("### Proof League")
    st.write("A high score means broad, retained and transferable learning—not who clicked, typed or studied the longest.")
    identity_cols = st.columns([2, 1])
    nickname = identity_cols[0].text_input("Competition nickname (opt-in)", value=current_name, key="league_nickname")
    if identity_cols[1].button("Save identity", width="stretch"):
        rename_learner(current_name, nickname)
        current_name = nickname.strip() or "Student"
        database.execute(
            "INSERT OR REPLACE INTO league_profiles (student_name, opted_in, league, created_at) VALUES (?, 1, ?, ?)",
            (current_name, st.session_state.course_title or "Local demo league", datetime.now().isoformat()),
        )
        st.success("Opt-in identity saved locally.")

    all_real_events = load_proof_events()
    real_events = [event for event in all_real_events if event.learner_id == current_name]
    curriculum = [topic["name"] for topic in st.session_state.topics]
    score = calculate_proof_score(
        real_events,
        learner_id=current_name,
        curriculum_topics=curriculum or None,
        comparison_history=[event for event in all_real_events if event.learner_id != current_name],
    )
    bonus = daily_learning_bonus(real_events, datetime.now().date())
    status_label = score.integrity_state.value.title() if score.evidence_count else "No evidence yet"
    cols = st.columns(4)
    cols[0].metric("ProofScore", f"{score.proof_score:.1f}/100" if score.evidence_count else "—", status_label)
    cols[1].metric("Verified topics", score.verified_topic_count, f"{score.evidence_count} usable checks")
    cols[2].metric("Spark", f"+{bonus.points}", "private cosmetic reward")
    cols[3].metric("Leaderboard", "Eligible" if score.leaderboard_eligible else "Provisional", "evidence gate")
    component_values = {
        "Knowledge": score.knowledge,
        "Transfer": score.transfer_depth,
        "Calibration": score.calibration,
        "Breadth": score.topic_breadth,
    }
    if score.teaching_impact is not None:
        component_values["Teaching"] = score.teaching_impact
    if score.reliability is not None:
        component_values["Reliability"] = score.reliability
    st.bar_chart(pd.DataFrame({"component": component_values.keys(), "score": component_values.values()}).set_index("component"), horizontal=True, color="#2764FF")
    if score.provisional_reasons:
        st.info("To enter the public board:\n\n- " + "\n- ".join(score.provisional_reasons))
        if st.session_state.topics and st.button("Schedule the missing delayed checks"):
            schedule_delayed_checks(st.session_state.topics)
            st.success("Fresh delayed checks were added to the calendar. They earn score only after completion evidence.")
    if score.verification_requests:
        st.warning("Some evidence overlaps an earlier response and has been held—not punished.")
        for request in score.verification_requests:
            st.write(f"**Fresh check for {request.topic_id}:** {request.instruction}")

    st.markdown("#### Course league")
    demo_specs = [("Maya", 0.86, 11), ("Jon", 0.76, 22), ("Lena", 0.68, 33)]
    rows: list[dict[str, Any]] = []
    for name, strength, seed in demo_specs:
        events = demo_league_events(name, strength, st.session_state.topics, seed)
        result = calculate_proof_score(events, learner_id=name, curriculum_topics=curriculum or None)
        rows.append({"Learner": name, "ProofScore": result.proof_score, "Topics": result.verified_topic_count, "Integrity": result.integrity_state.value, "Source": "labelled demo history"})
    if score.leaderboard_eligible:
        rows.append({"Learner": current_name, "ProofScore": score.proof_score, "Topics": score.verified_topic_count, "Integrity": score.integrity_state.value, "Source": "live local evidence"})
    leaderboard = pd.DataFrame(rows).sort_values(["ProofScore", "Topics"], ascending=False).reset_index(drop=True)
    leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
    st.dataframe(leaderboard, hide_index=True, width="stretch")
    st.caption("Demo rivals are clearly labelled; they make the scoring behaviour visible without pretending activity occurred live.")

    with st.expander("Why this resists reward hacking"):
        st.markdown(
            """
- Raw minutes, note length, message count and number of clicks are not score inputs.
- Same-topic attempts have diminishing weight and cannot manufacture breadth.
- Difficulty gives only a small bounded adjustment, so asking for impossible questions cannot inflate points.
- Delayed retrieval and a fresh transfer question are required before public eligibility.
- Near-copied answers are held for a new isomorphic challenge; no unreliable AI-authorship detector is used.
- Repeated or reciprocal teaching pairs rapidly lose credit, and teaching never scores without learner pre/post gain.
- Cosmetic Spark and private streaks are capped and never alter ProofScore or mastery.
            """
        )
        st.json(score.as_dict())


def render_audit() -> None:
    st.markdown("### AI Audit")
    st.write("Every important model action is inspectable. Calendar writes and scores remain validated by deterministic code.")
    rows = get_database().query("SELECT action, model, latency_ms, modality, payload, created_at FROM audit_log ORDER BY id DESC LIMIT 30")
    if not rows:
        st.info("Run the planner, retrieval check or evidence tutor to populate the audit trail.")
        return
    display = [
        {
            "When": row["created_at"].split("T")[-1][:8],
            "Gemma action": row["action"],
            "Model": row["model"],
            "Modality": row["modality"],
            "Latency": f"{row['latency_ms']} ms",
        }
        for row in rows
    ]
    st.dataframe(display, hide_index=True, width="stretch")
    selected = st.selectbox("Inspect an event", range(len(rows)), format_func=lambda index: f"{rows[index]['action']} · {rows[index]['created_at']}")
    try:
        st.json(json.loads(rows[selected]["payload"]))
    except json.JSONDecodeError:
        st.code(rows[selected]["payload"])
    st.caption("Raw hidden reasoning is never displayed or stored. The audit contains final structured results and tool calls only.")


load_css()
initialise_state()
get_database().initialize()
target_mark, _, _ = render_sidebar()
render_hero()

cockpit, focus, research, teachback, league, audit = st.tabs(
    ["Cockpit", "Focus + Proof", "Verified Tutor", "Teach-Back Arena", "Proof League", "AI Audit"]
)
with cockpit:
    render_cockpit(target_mark)
with focus:
    render_focus(target_mark)
with research:
    render_research()
with teachback:
    render_teachback()
with league:
    render_league()
with audit:
    render_audit()
