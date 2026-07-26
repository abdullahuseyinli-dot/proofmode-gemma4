from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from proofmode.gemma_client import GemmaClient


INTERVENTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_rescue_block",
            "description": "Create a very small immediate starter when overwhelm, anxiety or low energy is blocking action.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["topic", "minutes", "first_action", "reason"],
                "properties": {
                    "topic": {"type": "string"},
                    "minutes": {"type": "integer", "minimum": 2, "maximum": 15},
                    "first_action": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_block",
            "description": "Move a missed block when there is a genuine time conflict or fatigue and retain a small commitment now.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["topic", "new_start", "minutes", "first_action", "reason"],
                "properties": {
                    "topic": {"type": "string"},
                    "new_start": {"type": "string", "description": "ISO 8601 local datetime"},
                    "minutes": {"type": "integer", "minimum": 10, "maximum": 120},
                    "first_action": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_prerequisite",
            "description": "Switch to a short prerequisite explanation when confusion is the main blocker.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["topic", "prerequisite", "minutes", "first_action", "reason"],
                "properties": {
                    "topic": {"type": "string"},
                    "prerequisite": {"type": "string"},
                    "minutes": {"type": "integer", "minimum": 5, "maximum": 20},
                    "first_action": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
]


def choose_intervention(client: GemmaClient, topic: dict[str, Any], friction: str) -> dict[str, Any]:
    now = datetime.now().astimezone()
    result = client.choose_tool(
        "You are an autonomy-supportive anti-procrastination coach. Choose exactly one allowlisted action. Make it concrete, tiny, non-judgmental, and proportional. Do not diagnose mental health or use shame.",
        f"A student missed a block on {topic['name']}. They selected friction={friction!r}. Topic={topic}. Current local time={now.isoformat()}. Choose the best recovery action.",
        INTERVENTION_TOOLS,
    )
    if result.tool_calls:
        call = result.tool_calls[0]
        if call["name"] in {"create_rescue_block", "reschedule_block", "start_prerequisite"}:
            return call
    return fallback_intervention(topic, friction, now)


def fallback_intervention(topic: dict[str, Any], friction: str, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now().astimezone()
    name = topic.get("name", "this topic")
    if friction == "confused" and topic.get("prerequisites"):
        return {
            "name": "start_prerequisite",
            "arguments": {
                "topic": name,
                "prerequisite": topic["prerequisites"][0],
                "minutes": 7,
                "first_action": "Write one thing you know and one precise point of confusion.",
                "reason": "Removing the prerequisite gap makes the original task smaller.",
            },
        }
    if friction in {"tired", "time conflict"}:
        return {
            "name": "reschedule_block",
            "arguments": {
                "topic": name,
                "new_start": (now + timedelta(days=1)).replace(hour=10, minute=0).isoformat(),
                "minutes": 25,
                "first_action": "Open the notes and mark the first worked example now.",
                "reason": "Preserve momentum now and move the demanding work to a clearer slot.",
            },
        }
    return {
        "name": "create_rescue_block",
        "arguments": {
            "topic": name,
            "minutes": 7,
            "first_action": f"Explain the central idea of {name} in three imperfect sentences.",
            "reason": "A deliberately small start reduces the activation barrier.",
        },
    }

