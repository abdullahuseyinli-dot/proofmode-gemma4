from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Topic:
    name: str
    description: str = ""
    exam_weight: float = 0.5
    difficulty: float = 0.5
    mastery: float = 0.2
    confidence: float = 0.3
    prerequisite_centrality: float = 0.5
    estimated_minutes: int = 45
    prerequisites: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StudyBlock:
    topic: str
    start: datetime
    end: datetime
    contract: str
    status: str = "planned"
    id: str | None = None
    source: str = "proofmode"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start"] = self.start.isoformat()
        data["end"] = self.end.isoformat()
        return data


@dataclass
class AuditRecord:
    action: str
    model: str
    latency_ms: int
    modality: str = "text"
    tool_name: str | None = None
    structured: bool = True
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Source:
    source_id: str
    title: str
    url: str
    snippet: str
    domain: str = ""
    authoritative_score: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

