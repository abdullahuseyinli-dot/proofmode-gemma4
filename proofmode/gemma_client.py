from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from openai import APIConnectionError, APIError, OpenAI

from proofmode.config import settings


class GemmaUnavailable(RuntimeError):
    """Raised when the local Gemma server cannot be reached."""


class StructuredOutputError(RuntimeError):
    """Raised when Gemma's result cannot be parsed or validated."""


@dataclass
class GemmaResult:
    content: str
    latency_ms: int
    model: str
    payload: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None


def _strip_json_fence(value: str) -> str:
    value = value.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if match:
        value = match.group(1).strip()
    if not value.startswith("{"):
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    return value


def _validate_required(payload: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(payload, dict):
            raise StructuredOutputError(f"{path} must be an object")
        for key in schema.get("required", []):
            if key not in payload:
                raise StructuredOutputError(f"{path}.{key} is required")
        for key, child in schema.get("properties", {}).items():
            if key in payload:
                _validate_required(payload[key], child, f"{path}.{key}")
    elif expected == "array":
        if not isinstance(payload, list):
            raise StructuredOutputError(f"{path} must be an array")
        child = schema.get("items", {})
        for index, item in enumerate(payload):
            _validate_required(item, child, f"{path}[{index}]")
    elif expected == "string" and not isinstance(payload, str):
        raise StructuredOutputError(f"{path} must be a string")
    elif expected == "number" and not isinstance(payload, (int, float)):
        raise StructuredOutputError(f"{path} must be a number")
    elif expected == "integer" and not isinstance(payload, int):
        raise StructuredOutputError(f"{path} must be an integer")
    elif expected == "boolean" and not isinstance(payload, bool):
        raise StructuredOutputError(f"{path} must be a boolean")


class GemmaClient:
    """Small, audited client for the local llama.cpp OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        audit_callback: Callable[[str, int, str, dict[str, Any]], None] | None = None,
    ):
        self.base_url = (base_url or settings.gemma_base_url).rstrip("/")
        self.model = model or settings.model_name
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=settings.gemma_api_key,
            timeout=settings.request_timeout_seconds,
        )
        self.audit_callback = audit_callback

    def available(self) -> bool:
        try:
            response = httpx.get(self.base_url.replace("/v1", "/health"), timeout=2)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _audit(self, action: str, latency_ms: int, modality: str, payload: dict[str, Any]) -> None:
        if self.audit_callback:
            self.audit_callback(action, latency_ms, modality, payload)

    @staticmethod
    def file_part(data: bytes, mime_type: str) -> dict[str, Any]:
        encoded = base64.b64encode(data).decode("ascii")
        if mime_type.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}
        if mime_type.startswith("audio/"):
            fmt = "wav" if "wav" in mime_type else "mp3"
            return {"type": "input_audio", "input_audio": {"data": encoded, "format": fmt}}
        raise ValueError(f"Unsupported multimodal type: {mime_type}")

    def chat(
        self,
        system: str,
        user: str | list[dict[str, Any]],
        *,
        max_tokens: int = 700,
        modality: str = "text",
        temperature: float = 1.0,
    ) -> GemmaResult:
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                extra_body={"top_k": 64},
            )
        except (APIConnectionError, APIError) as error:
            raise GemmaUnavailable(
                "Local Gemma is not reachable. Start it with C:\\Users\\DELL\\gemma4\\start-server.cmd."
            ) from error
        latency = int((time.perf_counter() - start) * 1000)
        content = response.choices[0].message.content or ""
        self._audit("chat", latency, modality, {"preview": content[:280]})
        return GemmaResult(content=content, latency_ms=latency, model=self.model)

    def structured(
        self,
        system: str,
        user: str | list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        schema_name: str,
        max_tokens: int = 1200,
        modality: str = "text",
    ) -> GemmaResult:
        start = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": system
                + "\nReturn only valid JSON matching the supplied schema. Do not include markdown or hidden reasoning.",
            },
            {"role": "user", "content": user},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=1.0,
                top_p=0.95,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },  # type: ignore[arg-type]
                extra_body={"top_k": 64},
            )
        except (APIConnectionError, APIError) as error:
            raise GemmaUnavailable(
                "Local Gemma is not reachable. Start it with C:\\Users\\DELL\\gemma4\\start-server.cmd."
            ) from error
        latency = int((time.perf_counter() - start) * 1000)
        raw = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(_strip_json_fence(raw))
            _validate_required(payload, schema)
        except (json.JSONDecodeError, StructuredOutputError) as error:
            raise StructuredOutputError(f"Gemma returned invalid {schema_name} JSON: {raw[:300]}") from error
        self._audit(schema_name, latency, modality, {"schema": schema_name, "result": payload})
        return GemmaResult(content=raw, latency_ms=latency, model=self.model, payload=payload)

    def choose_tool(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 500,
    ) -> GemmaResult:
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                tools=tools,  # type: ignore[arg-type]
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=1.0,
                top_p=0.95,
                extra_body={"top_k": 64},
            )
        except (APIConnectionError, APIError) as error:
            raise GemmaUnavailable("Gemma tool selection failed because the local server is unavailable.") from error
        latency = int((time.perf_counter() - start) * 1000)
        message = response.choices[0].message
        calls: list[dict[str, Any]] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
        self._audit("tool_selection", latency, "text", {"tool_calls": calls})
        return GemmaResult(
            content=message.content or "",
            latency_ms=latency,
            model=self.model,
            tool_calls=calls,
        )

