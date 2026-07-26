# ProofMode: Schedule proof, not promises

## A local Gemma 4 learning twin that replans from evidence

Most AI study tools answer questions or generate a timetable. Both are easy to ignore. ProofMode closes the loop between intention and actual learning: it reads a student’s calendar and notes, schedules observable “proof contracts,” checks what the student can retrieve and transfer, then changes the plan.

### The problem

Procrastination is rarely solved by another chatbot. Learners over-plan, confuse familiarity with mastery, miss a session, feel guilty, and abandon the timetable. Study time is also a weak proxy for learning: two hours of rereading can produce less retention than ten minutes of retrieval.

ProofMode asks for only the desired mark, notes/syllabus, and a calendar export. Exam dates, existing commitments and available time are inferred. The target mark derives a depth mode—from core recall through independent transfer and teaching—without a long onboarding questionnaire.

### The closed loop

Gemma 4 first reads text, PDFs or note images and returns a schema-constrained topic/prerequisite map with difficulty, likely weighting and explicit uncertainty. Deterministic Python calculates a transparent priority from exam weight, mastery gap, forgetting risk, prerequisite centrality, target depth, urgency and effort. A calendar engine then finds real free time and creates conflict-free blocks with reminders.

Every block contains a Learning Contract such as: “Solve a new example without notes, justify each step, and finish with a confidence-rated retrieval question.” Completion requires a Learning Receipt: MCQs, an open transfer answer, confidence, and optionally a photo of worked notes. Gemma grades against a visible rubric, checking concepts and causal links rather than handwriting or length. Python updates knowledge, depth, calibration and mastery, then rebuilds the future calendar.

When a block is missed, Gemma uses native function calling to choose one allowlisted intervention: create a two-to-fifteen-minute rescue, reschedule, or start a missing prerequisite. The student selects only the friction—confused, tired, distracted, overwhelmed or a time conflict. The response is deliberately small and non-judgmental.

### Teach-Back Arena and fair competition

Peer chat normally rewards confidence or popularity. ProofMode measures whether teaching worked:

1. The learner answers a private pre-question.
2. A friend explains the concept.
3. The learner receives a different, isomorphic transfer question.
4. The friend earns Teaching Impact only when the learner improves; explanation accuracy modifies the award.

The public ProofScore is normalized to 0–100 from delayed retention, transfer depth, confidence calibration, topic breadth and Teaching Impact. It is not cumulative XP. Raw hours, note length, message volume, repeated easy quizzes and negative peer gains award nothing. Near-duplicate answers, repeated partner loops and same-topic farming become provisional and trigger a fresh transfer challenge. We do not use unreliable AI-authorship detection.

### Evidence-backed teaching

If Gemma needs broader or current knowledge, ProofMode does preparation rather than pretending to “train” the model. It creates a focused web query, ranks sources toward official and academic material, safely extracts a bounded research pack, and instructs Gemma to cite every factual claim. Code rejects unknown citations. A separate Gemma verification pass labels each claim supported, unsupported or uncertain using only the supplied evidence. Answers with unsupported or uncertain claims are held instead of presented as fact. The student can inspect every source and verifier decision.

### Why Gemma 4 is essential

This prototype runs Gemma 4 E4B QAT Q4 locally through llama.cpp on an NVIDIA laptop GPU. Gemma is not decorative: it performs multimodal curriculum extraction, question generation, rubric assessment, intervention tool selection, research-query preparation, cited explanation, claim verification and peer-teaching evaluation.

The AI Audit visibly records the exact Gemma model, modality, schema, tool call and latency. Raw hidden reasoning is never displayed or stored. Python retains authority over calendar writes, arithmetic, persistence and leaderboard values.

Local inference creates a meaningful privacy advantage: notes, voice, study behaviour and assessment history stay on the student’s device. Web research sends only the search query to public providers. The server binds to localhost.

### Engineering under sprint constraints

The application is one Streamlit process with SQLite state and an OpenAI-compatible local inference client. Calendar ingestion supports Outlook, Google and Apple `.ics` files, recurring events, time zones and two reminder alarms. An optional Google Calendar adapter is isolated so credentials cannot break the demo.

We chose prompt engineering and retrieval over fine-tuning because the product needs personal, changing material and verifiable current facts. Strict JSON schemas, allowlisted functions, bounded network access and deterministic scoring produce a stronger one-day foundation than an opaque end-to-end agent.

A Windows launcher starts or reuses Gemma and Streamlit, opens an app-style window, and provides a draggable always-on-top assistant bubble. The complete demo works locally without hosted-model quota or a login.

### Impact

ProofMode changes the key question from “Did you spend time?” to “What can you now retrieve, apply, and teach?” That makes planning adaptive, reminders purposeful, peer competition defensible, and Gemma central to a behaviour-changing learning loop.

