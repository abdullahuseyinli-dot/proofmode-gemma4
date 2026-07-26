from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from proofmode.config import settings


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'Student',
    target_mark REAL NOT NULL DEFAULT 70,
    depth_mode TEXT NOT NULL DEFAULT 'apply',
    timezone TEXT NOT NULL DEFAULT 'Europe/London',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    payload TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    UNIQUE(student_id, name)
);

CREATE TABLE IF NOT EXISTS calendar_events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    start_at TEXT NOT NULL,
    end_at TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS study_blocks (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    start_at TEXT NOT NULL,
    end_at TEXT NOT NULL,
    contract TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned',
    payload TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    kind TEXT NOT NULL,
    score REAL NOT NULL,
    confidence REAL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS study_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS peer_rooms (
    code TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    learner_name TEXT NOT NULL,
    teacher_name TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS peer_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_code TEXT NOT NULL,
    author TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teaching_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_code TEXT NOT NULL,
    teacher_name TEXT NOT NULL,
    topic TEXT NOT NULL,
    pre_score REAL NOT NULL,
    post_score REAL NOT NULL,
    impact REAL NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proof_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT NOT NULL,
    topic TEXT NOT NULL,
    event_type TEXT NOT NULL,
    integrity_status TEXT NOT NULL DEFAULT 'verified',
    evidence_text TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS league_profiles (
    student_name TEXT PRIMARY KEY,
    opted_in INTEGER NOT NULL DEFAULT 0,
    league TEXT NOT NULL DEFAULT 'Local demo league',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    modality TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO students (id, name, target_mark, depth_mode, timezone, created_at) VALUES (1, ?, ?, ?, ?, ?)",
                ("Student", 70, "apply", "Europe/London", datetime.now().isoformat()),
            )

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> int:
        with self.connect() as connection:
            cursor = connection.execute(sql, parameters)
            return int(cursor.lastrowid or 0)

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]

    def one(self, sql: str, parameters: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, parameters)
        return rows[0] if rows else None

    def set_profile(self, target_mark: float, name: str = "Student") -> dict[str, Any]:
        depth = depth_for_mark(target_mark)
        self.execute(
            "UPDATE students SET name = ?, target_mark = ?, depth_mode = ? WHERE id = 1",
            (name.strip() or "Student", float(target_mark), depth),
        )
        return self.one("SELECT * FROM students WHERE id = 1") or {}

    def save_topic(self, payload: dict[str, Any], priority: float) -> None:
        now = datetime.now().isoformat()
        self.execute(
            """
            INSERT INTO topics (student_id, name, payload, priority, updated_at)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(student_id, name) DO UPDATE SET
                payload=excluded.payload, priority=excluded.priority, updated_at=excluded.updated_at
            """,
            (payload["name"], json.dumps(payload), float(priority), now),
        )

    def load_topics(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT payload, priority FROM topics ORDER BY priority DESC")
        result = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["priority"] = row["priority"]
            result.append(payload)
        return result

    def add_audit(self, action: str, model: str, latency_ms: int, modality: str, payload: dict[str, Any]) -> None:
        self.execute(
            "INSERT INTO audit_log (action, model, latency_ms, modality, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (action, model, int(latency_ms), modality, json.dumps(payload), datetime.now().isoformat()),
        )


def depth_for_mark(target_mark: float) -> str:
    if target_mark >= 85:
        return "teach_research"
    if target_mark >= 70:
        return "transfer"
    if target_mark >= 60:
        return "apply"
    return "core"


def json_payload(row: dict[str, Any], key: str = "payload") -> dict[str, Any]:
    try:
        return json.loads(row.get(key, "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
