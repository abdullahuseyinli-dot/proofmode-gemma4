from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from proofmode.gemma_client import GemmaClient
from proofmode.services.teachback_service import score_transfer


class ScriptedCompletions:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.call_count = 0

    def create(self, **_: Any) -> SimpleNamespace:
        self.call_count += 1
        content = self.contents.pop(0)
        message = SimpleNamespace(content=content, tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_failed_structured_attempt_and_retry_are_both_audited() -> None:
    valid = json.dumps(
        {
            "score": 0.8,
            "accurate_points": ["Correct invariant"],
            "missing_points": [],
            "misconceptions": [],
            "feedback": "Good transfer.",
            "uncertainty": "Low",
        }
    )
    completions = ScriptedCompletions(["{truncated", valid])
    events: list[dict[str, Any]] = []
    client = GemmaClient.__new__(GemmaClient)
    client.base_url = "http://127.0.0.1:8080/v1"
    client.model = "test-gemma"
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.audit_callback = lambda action, latency, modality, payload: events.append(
        {
            "action": action,
            "latency_ms": latency,
            "modality": modality,
            "payload": payload,
        }
    )

    result = score_transfer(client, "Apply the invariant", "It is preserved", ["Name invariant"])

    assert result["score"] == 0.8
    assert completions.call_count == 2
    assert [event["action"] for event in events] == ["transfer_score", "transfer_score_retry"]
    assert [event["payload"]["outcome"] for event in events] == ["invalid_output", "success"]
    assert all(event["latency_ms"] >= 0 for event in events)
    assert events[0]["payload"]["output_chars"] == len("{truncated")
    assert "result" not in events[0]["payload"]
