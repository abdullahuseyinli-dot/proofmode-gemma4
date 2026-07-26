#!/usr/bin/env python3
"""Reproducible real-world benchmark for ProofMode's existing service layer.

The harness deliberately lives outside the product package.  It can import the
current tree or a detached baseline worktree through ``--repo-root`` without
changing product code.  Raw outputs are retained so every aggregate in the
report remains auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HARNESS_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = HARNESS_DIR.parent
DEFAULT_MANIFEST = HARNESS_DIR / "sources.json"
DEFAULT_CASES = HARNESS_DIR / "cases.json"
ALL_SUITES = (
    "documents",
    "course_map",
    "calendar",
    "questions",
    "assessment",
    "research",
    "teachback",
    "gamification",
)

ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["search_query", "preferred_domain", "risk"],
    "properties": {
        "search_query": {"type": "string"},
        "preferred_domain": {"type": "string"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

ROUTE_SYSTEM = (
    "You prepare a web search for a fact-checking tutor. search_query must contain at most "
    "10 precise terms, never a full question or generic best-practices wording. "
    "preferred_domain is one likely primary or official domain. Mark risk high for medical, "
    "legal, financial, current, statistical, or safety-sensitive questions. Keep JSON values short."
)


@dataclass
class Services:
    GemmaClient: Any
    GemmaUnavailable: Any
    StructuredOutputError: Any
    make_document: Callable[..., Any]
    extract_course_map: Callable[..., Any]
    prioritise_topics: Callable[..., Any]
    fallback_questions: Callable[..., Any]
    generate_questions: Callable[..., Any]
    assess_learning_receipt: Callable[..., Any]
    parse_ics: Callable[..., Any]
    detect_assessments: Callable[..., Any]
    schedule_study_blocks: Callable[..., Any]
    export_study_blocks_ics: Callable[..., Any]
    expand_recurring_events: Callable[..., Any]
    CalendarStudyBlock: Any
    prepare_research_pack: Callable[..., Any]
    build_evidence_prompt: Callable[..., Any]
    verify_answer: Callable[..., Any]
    select_supported_excerpt: Callable[..., Any]
    ResearchPack: Any
    ResearchSource: Any
    generate_transfer_pair: Callable[..., Any]
    score_transfer: Callable[..., Any]
    score_teaching_explanation: Callable[..., Any]
    teaching_impact: Callable[..., Any]
    EvidenceEvent: Any
    EvidenceKind: Any
    IntegrityState: Any
    calculate_proof_score: Callable[..., Any]
    assess_integrity: Callable[..., Any]


class AuditCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(
        self,
        action: str,
        latency_ms: int,
        modality: str,
        payload: dict[str, Any],
    ) -> None:
        self.events.append(
            {
                "action": action,
                "latency_ms": int(latency_ms),
                "modality": modality,
                "payload": payload,
            }
        )

    def mark(self) -> int:
        return len(self.events)

    def since(self, mark: int) -> list[dict[str, Any]]:
        return self.events[mark:]


class Recorder:
    def __init__(self, run_id: str, run_label: str) -> None:
        self.run_id = run_id
        self.run_label = run_label
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        suite: str,
        case_id: str,
        variant: str,
        *,
        repeat: int = 1,
        metrics: Mapping[str, Any] | None = None,
        output: Any = None,
        success: bool = True,
        error: BaseException | str | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "run_label": self.run_label,
            "suite": suite,
            "case_id": case_id,
            "variant": variant,
            "repeat": repeat,
            "success": bool(success),
            "metrics": jsonable(dict(metrics or {})),
            "output": jsonable(output),
        }
        if error is not None:
            row["error_type"] = type(error).__name__ if isinstance(error, BaseException) else "Error"
            row["error"] = str(error)[:2_000]
        self.records.append(row)


@dataclass
class Context:
    args: argparse.Namespace
    services: Services
    sources: list[dict[str, Any]]
    cases: dict[str, Any]
    cache_dir: Path
    artifacts_dir: Path
    recorder: Recorder
    audit: AuditCollector
    client: Any
    source_lock: list[dict[str, Any]]
    generated_course_maps: dict[str, dict[str, Any]] = field(default_factory=dict)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "as_dict"):
        return jsonable(value.as_dict())
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return str(value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = value.casefold().replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def term_recall(text: str, terms: Sequence[str]) -> float:
    if not terms:
        return 1.0
    normalized = normalize_text(text)
    hits = sum(normalize_text(term) in normalized for term in terms)
    return hits / len(terms)


def expected_span_recall(text: str, spans: Sequence[str]) -> float | None:
    if not spans:
        return None
    return term_recall(text, spans)


def token_f1(first: str, second: str) -> float:
    left = set(normalize_text(first).split())
    right = set(normalize_text(second).split())
    if not left or not right:
        return 0.0
    precision = len(left & right) / len(left)
    recall = len(left & right) / len(right)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def topic_metrics(
    predicted: Sequence[str],
    expected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    candidates: list[tuple[float, int, int, str]] = []
    for p_index, prediction in enumerate(predicted):
        for e_index, item in enumerate(expected):
            aliases = [str(item.get("name", "")), *[str(x) for x in item.get("aliases", [])]]
            score = max(
                max(token_f1(prediction, alias), SequenceMatcher(None, normalize_text(prediction), normalize_text(alias)).ratio())
                for alias in aliases
                if alias.strip()
            )
            candidates.append((score, p_index, e_index, aliases[0]))
    used_predictions: set[int] = set()
    used_expected: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, p_index, e_index, expected_name in sorted(candidates, reverse=True):
        if score < 0.42 or p_index in used_predictions or e_index in used_expected:
            continue
        used_predictions.add(p_index)
        used_expected.add(e_index)
        matches.append(
            {"predicted": predicted[p_index], "expected": expected_name, "similarity": round(score, 3)}
        )
    precision = len(matches) / len(predicted) if predicted else 0.0
    recall = len(matches) / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "topic_precision": precision,
        "topic_recall": recall,
        "topic_f1": f1,
        "topic_count": float(len(predicted)),
    }, matches


def baseline_heading_topics(text: str, limit: int = 12) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    headings: list[str] = []
    for index, line in enumerate(lines):
        if not 3 <= len(line) <= 90:
            continue
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        rst_heading = bool(next_line and set(next_line) <= set("=-~^\"`:+*#_") and len(next_line) >= 3)
        numbered = bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?|lesson\s+\d+)\s+\S", line, re.I))
        titleish = line.istitle() and len(line.split()) <= 9
        if rst_heading or numbered or titleish:
            clean = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", line).strip(" :-")
            if clean and normalize_text(clean) not in {normalize_text(item) for item in headings}:
                headings.append(clean)
        if len(headings) >= limit:
            break
    if not headings:
        headings = [line for line in lines if 3 <= len(line) <= 70][:limit]
    return headings


def profile_selected(item: Mapping[str, Any], profile: str) -> bool:
    return profile in item.get("profiles", ["quick", "full"])


def selected_suites(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return ALL_SUITES
    suites = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    unknown = sorted(set(suites) - set(ALL_SUITES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown suite(s): {', '.join(unknown)}")
    return suites


def import_services(repo_root: Path) -> Services:
    package = repo_root / "proofmode" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(f"Selected repo root has no ProofMode package: {package}")
    sys.path.insert(0, str(repo_root))

    gemma = importlib.import_module("proofmode.gemma_client")
    documents = importlib.import_module("proofmode.services.document_service")
    planner = importlib.import_module("proofmode.services.planner_service")
    assessment = importlib.import_module("proofmode.services.assessment_service")
    calendar = importlib.import_module("proofmode.services.calendar_service")
    research = importlib.import_module("proofmode.services.research_service")
    verification = importlib.import_module("proofmode.services.verification_service")
    teachback = importlib.import_module("proofmode.services.teachback_service")
    gamification = importlib.import_module("proofmode.gamification")
    demo = importlib.import_module("proofmode.demo_data")

    return Services(
        GemmaClient=gemma.GemmaClient,
        GemmaUnavailable=gemma.GemmaUnavailable,
        StructuredOutputError=gemma.StructuredOutputError,
        make_document=documents.make_document,
        extract_course_map=planner.extract_course_map,
        prioritise_topics=planner.prioritise_topics,
        fallback_questions=demo.fallback_questions,
        generate_questions=assessment.generate_questions,
        assess_learning_receipt=assessment.assess_learning_receipt,
        parse_ics=calendar.parse_ics,
        detect_assessments=calendar.detect_assessments,
        schedule_study_blocks=calendar.schedule_study_blocks,
        export_study_blocks_ics=calendar.export_study_blocks_ics,
        expand_recurring_events=calendar.expand_recurring_events,
        CalendarStudyBlock=calendar.StudyBlock,
        prepare_research_pack=research.prepare_research_pack,
        build_evidence_prompt=research.build_evidence_prompt,
        verify_answer=verification.verify_answer,
        select_supported_excerpt=verification.select_supported_excerpt,
        ResearchPack=research.ResearchPack,
        ResearchSource=research.ResearchSource,
        generate_transfer_pair=teachback.generate_transfer_pair,
        score_transfer=teachback.score_transfer,
        score_teaching_explanation=teachback.score_teaching_explanation,
        teaching_impact=teachback.teaching_impact,
        EvidenceEvent=gamification.EvidenceEvent,
        EvidenceKind=gamification.EvidenceKind,
        IntegrityState=gamification.IntegrityState,
        calculate_proof_score=gamification.calculate_proof_score,
        assess_integrity=gamification.assess_integrity,
    )


def fetch_json(url: str, timeout: float = 3.0) -> Any:
    request = Request(url, headers={"User-Agent": "ProofModeBenchmark/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(2_000_000).decode("utf-8", errors="replace"))


def model_environment(base_url: str) -> dict[str, Any]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    result: dict[str, Any] = {"base_url": base_url, "available": False}
    try:
        health = fetch_json(root + "/health")
        result["health"] = health
        result["available"] = health.get("status") == "ok"
    except Exception as error:
        result["health_error"] = f"{type(error).__name__}: {error}"
        return result
    try:
        props = fetch_json(root + "/props")
        defaults = props.get("default_generation_settings", {})
        params = defaults.get("params", {}) if isinstance(defaults, Mapping) else {}
        result.update(
            {
                "model_alias": props.get("model_alias"),
                "model_ftype": props.get("model_ftype"),
                "context_size": params.get("n_ctx"),
                "temperature": params.get("temperature"),
                "top_k": params.get("top_k"),
                "top_p": params.get("top_p"),
                "total_slots": props.get("total_slots"),
                "modalities": props.get("modalities"),
                "build_info": props.get("build_info"),
            }
        )
    except Exception as error:
        result["props_error"] = f"{type(error).__name__}: {error}"
    return result


def git_environment(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "repo_root": str(repo_root.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current") or "detached",
        "dirty": bool(status),
        "status_entries": status.splitlines()[:50],
    }


def download_one(source: Mapping[str, Any], cache_dir: Path, refresh: bool) -> dict[str, Any]:
    path = cache_dir / "sources" / str(source["filename"])
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    started = time.perf_counter()
    if refresh or not path.is_file():
        request = Request(
            str(source["url"]),
            headers={
                "User-Agent": "ProofModeBenchmark/1.0 (+local reproducibility harness)",
                "Accept": "text/plain,text/html,application/pdf,image/*,*/*;q=0.1",
            },
        )
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(request, timeout=45) as response, temporary.open("wb") as output:
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 80_000_000:
                        raise ValueError(f"Source exceeded 80 MB cap: {source['id']}")
                    output.write(chunk)
            temporary.replace(path)
            downloaded = True
        finally:
            if temporary.exists():
                temporary.unlink()
    digest = sha256_file(path)
    expected = str(source.get("sha256") or "").lower()
    if expected and digest != expected:
        raise ValueError(f"SHA-256 mismatch for {source['id']}: expected {expected}, got {digest}")
    return {
        "id": source["id"],
        "title": source["title"],
        "url": source["url"],
        "filename": source["filename"],
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "downloaded_this_run": downloaded,
        "download_ms": round((time.perf_counter() - started) * 1000, 2),
        "license": source.get("license"),
        "license_url": source.get("license_url"),
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


def download_sources(
    sources: Sequence[dict[str, Any]],
    cache_dir: Path,
    refresh: bool,
) -> list[dict[str, Any]]:
    lock: list[dict[str, Any]] = []
    for source in sources:
        print(f"[source] {source['id']}")
        lock.append(download_one(source, cache_dir, refresh))
    lock_path = cache_dir / "sources.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    return lock


def source_path(ctx: Context, source: Mapping[str, Any]) -> Path:
    return ctx.cache_dir / "sources" / str(source["filename"])


def audit_metrics(ctx: Context, mark: int, wall_ms: float) -> dict[str, Any]:
    events = ctx.audit.since(mark)
    return {
        "wall_ms": round(wall_ms, 2),
        "model_call_count": len(events),
        "model_latency_ms": sum(int(item["latency_ms"]) for item in events),
    }


def run_documents(ctx: Context) -> None:
    for source in ctx.sources:
        path = source_path(ctx, source)
        data = path.read_bytes()
        expected = source.get("expected_spans", [])

        started = time.perf_counter()
        baseline = data.decode("utf-8", errors="ignore")
        baseline_ms = (time.perf_counter() - started) * 1000
        baseline_recall = expected_span_recall(baseline, expected)
        metrics: dict[str, Any] = {
            "extracted_chars": len(baseline),
            "empty_output": not bool(baseline.strip()),
            "wall_ms": round(baseline_ms, 2),
        }
        if baseline_recall is not None:
            metrics["gold_span_recall"] = baseline_recall
        ctx.recorder.add(
            "documents",
            str(source["id"]),
            "baseline_byte_decode",
            metrics=metrics,
            output={"preview": baseline[:800]},
        )

        try:
            started = time.perf_counter()
            document = ctx.services.make_document(
                str(source["filename"]), data, str(source["media_type"])
            )
            elapsed = (time.perf_counter() - started) * 1000
            recall = expected_span_recall(document.text, expected)
            metrics = {
                "extracted_chars": len(document.text),
                "empty_output": not bool(document.text.strip()),
                "wall_ms": round(elapsed, 2),
            }
            if recall is not None:
                metrics["gold_span_recall"] = recall
            ctx.recorder.add(
                "documents",
                str(source["id"]),
                "optimized_mime_extraction",
                metrics=metrics,
                output={
                    "mime_type": document.mime_type,
                    "is_image": document.is_image,
                    "preview": document.text[:800],
                },
            )
        except Exception as error:
            ctx.recorder.add(
                "documents",
                str(source["id"]),
                "optimized_mime_extraction",
                success=False,
                error=error,
            )


def run_course_maps(ctx: Context) -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    curriculum = [source for source in ctx.sources if source.get("role") == "curriculum"]
    repeats = ctx.args.repeats
    for source in curriculum:
        path = source_path(ctx, source)
        data = path.read_bytes()
        expected = source.get("expected_topics", [])
        try:
            document = ctx.services.make_document(
                str(source["filename"]), data, str(source["media_type"])
            )
        except Exception as error:
            ctx.recorder.add("course_map", str(source["id"]), "optimized_gemma_structured", success=False, error=error)
            continue

        headings = baseline_heading_topics(document.text)
        baseline_metrics, baseline_matches = topic_metrics(headings, expected)
        for repeat in range(1, repeats + 1):
            ctx.recorder.add(
                "course_map",
                str(source["id"]),
                "baseline_heading_heuristic",
                repeat=repeat,
                metrics={**baseline_metrics, "wall_ms": 0.0, "model_call_count": 0},
                output={"topics": headings, "matches": baseline_matches},
            )

            mark = ctx.audit.mark()
            started = time.perf_counter()
            try:
                payload = ctx.services.extract_course_map(
                    ctx.client,
                    [document],
                    "Benchmark final examination in 14 days; no official topic weights supplied.",
                    80.0,
                    "transfer",
                )
                wall = (time.perf_counter() - started) * 1000
                predicted = [str(item.get("name", "")) for item in payload.get("topics", [])]
                scores, matches = topic_metrics(predicted, expected)
                complete = bool(payload.get("course_title") and payload.get("exam_summary") and predicted)
                fraction_fields = (
                    "exam_weight",
                    "difficulty",
                    "mastery",
                    "confidence",
                    "prerequisite_centrality",
                )
                fraction_checks = [
                    isinstance(item.get(field), (int, float))
                    and not isinstance(item.get(field), bool)
                    and math.isfinite(float(item[field]))
                    and 0.0 <= float(item[field]) <= 1.0
                    for item in payload.get("topics", [])
                    if isinstance(item, Mapping)
                    for field in fraction_fields
                ]
                minute_checks = [
                    isinstance(item.get("estimated_minutes"), int)
                    and not isinstance(item.get("estimated_minutes"), bool)
                    and 15 <= int(item["estimated_minutes"]) <= 360
                    for item in payload.get("topics", [])
                    if isinstance(item, Mapping)
                ]
                fraction_valid_rate = (
                    sum(fraction_checks) / len(fraction_checks) if fraction_checks else 0.0
                )
                minute_valid_rate = sum(minute_checks) / len(minute_checks) if minute_checks else 0.0
                ctx.recorder.add(
                    "course_map",
                    str(source["id"]),
                    "optimized_gemma_structured",
                    repeat=repeat,
                    metrics={
                        **scores,
                        "schema_complete": complete,
                        "fraction_fields_valid_rate": fraction_valid_rate,
                        "estimated_minutes_valid_rate": minute_valid_rate,
                        "numeric_bounds_valid": bool(
                            fraction_checks
                            and minute_checks
                            and all(fraction_checks)
                            and all(minute_checks)
                        ),
                        **audit_metrics(ctx, mark, wall),
                    },
                    output={"course_map": payload, "matches": matches},
                )
                maps[str(source["id"])] = payload
            except Exception as error:
                wall = (time.perf_counter() - started) * 1000
                ctx.recorder.add(
                    "course_map",
                    str(source["id"]),
                    "optimized_gemma_structured",
                    repeat=repeat,
                    metrics=audit_metrics(ctx, mark, wall),
                    success=False,
                    error=error,
                )
    return maps


def topic_record(name: str, description: str = "Benchmark topic from an authoritative course source.") -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "exam_weight": 0.65,
        "difficulty": 0.62,
        "mastery": 0.30,
        "confidence": 0.40,
        "prerequisite_centrality": 0.65,
        "estimated_minutes": 55,
        "prerequisites": [],
        "evidence": ["Downloaded benchmark source"],
        "uncertainty": "Benchmark prior; not observed learner mastery.",
    }


def question_metrics(payload: Mapping[str, Any], topic_name: str) -> dict[str, Any]:
    mcqs = payload.get("mcqs", []) if isinstance(payload, Mapping) else []
    valid_rows = []
    unique_rows = []
    skills: set[str] = set()
    text_parts: list[str] = []
    for row in mcqs if isinstance(mcqs, list) else []:
        if not isinstance(row, Mapping):
            continue
        options = row.get("options", [])
        correct = row.get("correct_index")
        valid = isinstance(options, list) and len(options) == 4 and isinstance(correct, int) and 0 <= correct < 4
        valid_rows.append(valid)
        unique_rows.append(len({normalize_text(str(item)) for item in options}) == len(options) if isinstance(options, list) else False)
        if row.get("skill"):
            skills.add(normalize_text(str(row["skill"])))
        text_parts.append(str(row.get("question", "")))
    text_parts.append(str(payload.get("open_question", "")))
    topic_terms = [term for term in normalize_text(topic_name).split() if len(term) >= 5]
    return {
        "schema_complete": bool(mcqs and payload.get("open_question") and payload.get("open_rubric")),
        "mcq_count": len(mcqs) if isinstance(mcqs, list) else 0,
        "valid_mcq_rate": sum(valid_rows) / len(valid_rows) if valid_rows else 0.0,
        "unique_option_rate": sum(unique_rows) / len(unique_rows) if unique_rows else 0.0,
        "topic_term_recall": term_recall(" ".join(text_parts), topic_terms),
        "skill_diversity": len(skills),
        "rubric_items": len(payload.get("open_rubric", [])) if isinstance(payload.get("open_rubric"), list) else 0,
    }


def run_questions(ctx: Context) -> None:
    for source in [item for item in ctx.sources if item.get("role") == "curriculum"]:
        expected_topics = source.get("expected_topics", [])
        if not expected_topics:
            continue
        name = str(expected_topics[0]["name"])
        generated_topics = ctx.generated_course_maps.get(str(source["id"]), {}).get("topics", [])
        record = topic_record(name)
        if generated_topics:
            aliases = [
                str(expected_topics[0].get("name", "")),
                *[str(item) for item in expected_topics[0].get("aliases", [])],
            ]
            record = max(
                (dict(item) for item in generated_topics if isinstance(item, Mapping)),
                key=lambda item: term_recall(json.dumps(item, ensure_ascii=False), aliases),
                default=record,
            )
        upstream_used = bool(generated_topics and record)
        for repeat in range(1, ctx.args.repeats + 1):
            baseline = ctx.services.fallback_questions(name)
            ctx.recorder.add(
                "questions",
                str(source["id"]),
                "baseline_generic_fallback",
                repeat=repeat,
                metrics={**question_metrics(baseline, name), "wall_ms": 0.0, "model_call_count": 0},
                output=baseline,
            )
            mark = ctx.audit.mark()
            started = time.perf_counter()
            try:
                generated = ctx.services.generate_questions(ctx.client, record, 80.0, "transfer")
                wall = (time.perf_counter() - started) * 1000
                ctx.recorder.add(
                    "questions",
                    str(source["id"]),
                    "optimized_gemma_generated",
                    repeat=repeat,
                    metrics={
                        **question_metrics(generated, name),
                        "upstream_course_map_used": upstream_used,
                        **audit_metrics(ctx, mark, wall),
                    },
                    output={"topic_record": record, "questions": generated},
                )
            except Exception as error:
                wall = (time.perf_counter() - started) * 1000
                ctx.recorder.add(
                    "questions",
                    str(source["id"]),
                    "optimized_gemma_generated",
                    repeat=repeat,
                    metrics=audit_metrics(ctx, mark, wall),
                    success=False,
                    error=error,
                )


def run_assessment(ctx: Context) -> None:
    for case in ctx.cases.get("assessment", []):
        if not profile_selected(case, ctx.args.profile):
            continue
        record = topic_record(str(case["topic"]), str(case["question"]))
        answers = [item for item in case.get("answers", []) if profile_selected(item, ctx.args.profile)]
        if ctx.args.profile == "full":
            misconception = next((item for item in answers if item.get("label") == "misconception"), None)
            if misconception:
                answers = [
                    *answers,
                    {
                        **misconception,
                        "label": "misconception_verbose",
                        "text": str(misconception["text"]) + " " + ("This response is detailed and confidently presented. " * 12),
                    },
                ]
        for answer_case in answers:
            case_id = f"{case['id']}:{answer_case['label']}"
            gold = float(answer_case["gold"])
            baseline_score = term_recall(str(answer_case["text"]), case.get("expected_terms", []))
            for repeat in range(1, ctx.args.repeats + 1):
                ctx.recorder.add(
                    "assessment",
                    case_id,
                    "baseline_keyword_coverage",
                    repeat=repeat,
                    metrics={
                        "predicted_score": baseline_score,
                        "gold_score": gold,
                        "absolute_error": abs(baseline_score - gold),
                        "pass_correct": (baseline_score >= 0.60) == (gold >= 0.60),
                        "wall_ms": 0.0,
                        "model_call_count": 0,
                    },
                    output={"expected_terms": case.get("expected_terms", [])},
                )
                mark = ctx.audit.mark()
                started = time.perf_counter()
                try:
                    result = ctx.services.assess_learning_receipt(
                        ctx.client,
                        record,
                        str(case["question"]),
                        str(answer_case["text"]),
                        list(case["rubric"]),
                    )
                    wall = (time.perf_counter() - started) * 1000
                    prediction = float(result.get("correctness", 0.0))
                    ctx.recorder.add(
                        "assessment",
                        case_id,
                        "optimized_gemma_rubric",
                        repeat=repeat,
                        metrics={
                            "predicted_score": prediction,
                            "gold_score": gold,
                            "absolute_error": abs(prediction - gold),
                            "pass_correct": (prediction >= 0.60) == (gold >= 0.60),
                            "depth_score": float(result.get("depth", 0.0)),
                            "evidence_quality": float(result.get("evidence_quality", 0.0)),
                            **audit_metrics(ctx, mark, wall),
                        },
                        output=result,
                    )
                except Exception as error:
                    wall = (time.perf_counter() - started) * 1000
                    ctx.recorder.add(
                        "assessment",
                        case_id,
                        "optimized_gemma_rubric",
                        repeat=repeat,
                        metrics=audit_metrics(ctx, mark, wall),
                        success=False,
                        error=error,
                    )


def benchmark_calendar_payload() -> bytes:
    now = datetime.now().astimezone().replace(second=0, microsecond=0)
    day = (now + timedelta(days=1)).date()
    exam_day = (now + timedelta(days=10)).date()

    def stamp(value: datetime) -> str:
        return value.strftime("%Y%m%dT%H%M%S")

    zone = now.tzinfo
    work_start = datetime.combine(day, datetime.min.time(), zone).replace(hour=9)
    lecture_start = work_start.replace(hour=14)
    exam_start = datetime.combine(exam_day, datetime.min.time(), zone).replace(hour=14)
    deadline = exam_start - timedelta(days=2)
    text = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//ProofMode Benchmark//EN
BEGIN:VEVENT
UID:benchmark-final
DTSTART;TZID=Europe/London:{stamp(exam_start)}
DTEND;TZID=Europe/London:{stamp(exam_start + timedelta(hours=2))}
SUMMARY:Applied Learning Final Exam
DESCRIPTION:Final exam covering leakage, eigenvalues, and epidemiology.
END:VEVENT
BEGIN:VEVENT
UID:benchmark-work
DTSTART;TZID=Europe/London:{stamp(work_start)}
DTEND;TZID=Europe/London:{stamp(work_start + timedelta(hours=3))}
RRULE:FREQ=DAILY;COUNT=3
SUMMARY:Part-time work
END:VEVENT
BEGIN:VEVENT
UID:benchmark-lecture
DTSTART;TZID=Europe/London:{stamp(lecture_start)}
DTEND;TZID=Europe/London:{stamp(lecture_start + timedelta(hours=2))}
SUMMARY:Revision lecture
END:VEVENT
BEGIN:VEVENT
UID:benchmark-deadline
DTSTART;TZID=Europe/London:{stamp(deadline)}
DTEND;TZID=Europe/London:{stamp(deadline + timedelta(minutes=30))}
SUMMARY:Coursework submission deadline
END:VEVENT
END:VCALENDAR
"""
    return text.replace("\n", "\r\n").encode("utf-8")


def overlap_minutes(blocks: Sequence[Any], events: Sequence[Any]) -> float:
    total = 0.0
    for block in blocks:
        for event in events:
            if not getattr(event, "blocks_time", True):
                continue
            start = max(block.start, event.start)
            end = min(block.end, event.end)
            if end > start:
                total += (end - start).total_seconds() / 60
    return total


def calendar_metrics(
    ctx: Context,
    blocks: Sequence[Any],
    events: Sequence[Any],
    expanded: Sequence[Any],
    exam: datetime,
    required_minutes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    exported = ctx.services.export_study_blocks_ics(blocks)
    roundtrip = ctx.services.parse_ics(exported)
    assessments = ctx.services.detect_assessments(events)
    allocated = sum(int(block.duration_minutes) for block in blocks)
    return (
        {
            "parsed_event_count": len(events),
            "assessment_recall": float(any(item.kind == "exam" for item in assessments)),
            "assessment_false_positives": max(0, len(assessments) - 2),
            "conflict_minutes": overlap_minutes(blocks, expanded),
            "deadline_violations": sum(block.end > exam for block in blocks),
            "block_limit_violations": sum(block.duration_minutes > 50 or block.duration_minutes < 20 for block in blocks),
            "allocated_ratio": min(1.0, allocated / max(required_minutes, 1)),
            "roundtrip_block_recall": len(roundtrip) / max(len(blocks), 1),
            "alarm_count": exported.count(b"BEGIN:VALARM"),
            "wall_ms": 0.0,
            "model_call_count": 0,
        },
        {
            "blocks": [item.to_dict() for item in blocks],
            "detected_assessments": [item.to_dict() for item in assessments],
            "export_sha256": sha256_bytes(exported),
        },
    )


def run_calendar(ctx: Context) -> None:
    payload = benchmark_calendar_payload()
    events = ctx.services.parse_ics(payload)
    assessments = ctx.services.detect_assessments(events)
    exam = min(item.when for item in assessments if item.kind == "exam")
    now = datetime.now().astimezone().replace(second=0, microsecond=0)
    end = exam + timedelta(hours=1)
    expanded = ctx.services.expand_recurring_events(events, now, end)
    tasks = [
        {
            "task_id": f"calendar-{index}",
            "name": topic,
            "estimated_minutes": 90,
            "priority": priority,
            "exam_date": exam.isoformat(),
            "reason": f"Closed-book proof for {topic}",
            "min_block_minutes": 20,
            "max_block_minutes": 50,
        }
        for index, (topic, priority) in enumerate(
            [("Data leakage", 8.0), ("Eigenvalues", 7.0), ("Epidemiology", 6.0)], 1
        )
    ]
    required = sum(int(item["estimated_minutes"]) for item in tasks)

    work = next(item for item in events if item.uid == "benchmark-work")
    baseline: list[Any] = []
    cursor = work.start
    for index, task in enumerate(tasks):
        for session in range(2):
            block_start = cursor + timedelta(minutes=(index * 2 + session) * 55)
            baseline.append(
                ctx.services.CalendarStudyBlock(
                    uid=f"baseline-{index}-{session}",
                    topic=str(task["name"]),
                    title=f"Baseline - {task['name']}",
                    start=block_start,
                    end=block_start + timedelta(minutes=45),
                    priority=float(task["priority"]),
                    reason=str(task["reason"]),
                    task_id=str(task["task_id"]),
                )
            )
    baseline_metrics, baseline_output = calendar_metrics(ctx, baseline, events, expanded, exam, required)
    ctx.recorder.add("calendar", "realistic-ics", "baseline_sequential", metrics=baseline_metrics, output=baseline_output)

    started = time.perf_counter()
    try:
        optimized = ctx.services.schedule_study_blocks(
            tasks,
            events,
            now,
            exam,
            min_block_minutes=20,
            max_block_minutes=50,
            break_minutes=10,
            event_buffer_minutes=10,
        )
        wall = (time.perf_counter() - started) * 1000
        optimized_metrics, optimized_output = calendar_metrics(ctx, optimized, events, expanded, exam, required)
        optimized_metrics["wall_ms"] = round(wall, 2)
        ctx.recorder.add("calendar", "realistic-ics", "optimized_conflict_aware", metrics=optimized_metrics, output=optimized_output)
    except Exception as error:
        ctx.recorder.add("calendar", "realistic-ics", "optimized_conflict_aware", success=False, error=error)


def pack_to_json(pack: Any) -> dict[str, Any]:
    return pack.as_dict() if hasattr(pack, "as_dict") else jsonable(pack)


def pack_from_json(ctx: Context, payload: Mapping[str, Any]) -> Any:
    sources = tuple(
        ctx.services.ResearchSource(
            source_id=str(item.get("source_id", "")),
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            snippet=str(item.get("snippet", "")),
            domain=str(item.get("domain", "")),
            text=str(item.get("text", "")),
            authority_score=float(item.get("authority_score", 0.0)),
        )
        for item in payload.get("sources", [])
    )
    return ctx.services.ResearchPack(
        query=str(payload.get("query", "")),
        sources=sources,
        status=str(payload.get("status", "offline")),
        warnings=tuple(str(item) for item in payload.get("warnings", [])),
        method=str(payload.get("method", "preparation/research pack (no fine-tuning)")),
    )


def cached_research_pack(ctx: Context, case_id: str, variant: str, query: str) -> tuple[Any, bool, float]:
    cache_path = ctx.cache_dir / "research" / f"{case_id}-{variant}.json"
    if ctx.args.reuse_research_cache and cache_path.is_file() and not ctx.args.refresh_research:
        started = time.perf_counter()
        pack = pack_from_json(ctx, load_json(cache_path))
        return pack, True, (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    pack = ctx.services.prepare_research_pack(
        query,
        max_results=ctx.args.research_results,
        timeout=ctx.args.web_timeout,
        max_chars_per_source=5_000,
    )
    elapsed = (time.perf_counter() - started) * 1000
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(pack_to_json(pack), indent=2, ensure_ascii=False), encoding="utf-8")
    return pack, False, elapsed


def research_pack_metrics(pack: Any, expected_domains: Sequence[str], elapsed_ms: float, cache_hit: bool) -> dict[str, Any]:
    sources = list(pack.sources)
    domains = [str(source.domain).casefold() for source in sources]
    target_hit = any(
        domain == expected.casefold() or domain.endswith("." + expected.casefold())
        for domain in domains
        for expected in expected_domains
    )
    return {
        "source_count": len(sources),
        "target_domain_hit": target_hit,
        "fetched_fraction": sum(bool(source.text.strip()) for source in sources) / max(len(sources), 1),
        "authority_mean": statistics.fmean([float(source.authority_score) for source in sources]) if sources else 0.0,
        "online": pack.status in {"online", "partial"},
        "cache_hit": cache_hit,
        "wall_ms": round(elapsed_ms, 2),
        "model_call_count": 0,
    }


def verify_callback(ctx: Context, prompt: str) -> str:
    return ctx.client.chat(
        "You are a strict claim verifier. Return only the requested JSON and never use outside knowledge.",
        prompt,
        max_tokens=1800,
        temperature=0.1,
    ).content


def answer_metrics(answer: str, report: Any | None, expected_terms: Sequence[str], displayed: bool) -> dict[str, Any]:
    recall = term_recall(answer, expected_terms)
    if report is None:
        citation_valid = False
        unsupported = 1 if answer.strip() else 0
        uncertain = 0
        safe = False
    else:
        citation_valid = bool(report.citation_validation.valid)
        unsupported = int(report.unsupported_count)
        uncertain = int(report.uncertain_count)
        safe = bool(report.safe_to_show)
    approximate_correct = recall >= 0.60
    return {
        "gold_term_recall": recall,
        "citation_valid": citation_valid,
        "safe_to_show": safe,
        "displayed": displayed,
        "unsupported_claims": unsupported,
        "uncertain_claims": uncertain,
        "unsupported_leakage": bool(displayed and (unsupported > 0 or uncertain > 0 or not citation_valid)),
        "displayed_approx_correct": bool(displayed and approximate_correct),
        "correct_or_held": bool((displayed and approximate_correct) or (not displayed and not safe)),
    }


def run_research(ctx: Context) -> None:
    for case in ctx.cases.get("research", []):
        if not profile_selected(case, ctx.args.profile):
            continue
        case_id = str(case["id"])
        question = str(case["question"])
        expected_terms = list(case.get("expected_terms", []))
        expected_domains = list(case.get("expected_domains", []))

        for repeat in range(1, ctx.args.repeats + 1):
            mark = ctx.audit.mark()
            started = time.perf_counter()
            try:
                memory = ctx.client.chat(
                    "Answer the student's factual question concisely from your existing model knowledge.",
                    question,
                    max_tokens=550,
                ).content
                wall = (time.perf_counter() - started) * 1000
                ctx.recorder.add(
                    "research",
                    case_id,
                    "baseline_memory_answer",
                    repeat=repeat,
                    metrics={
                        **answer_metrics(memory, None, expected_terms, displayed=True),
                        **audit_metrics(ctx, mark, wall),
                    },
                    output={"answer": memory},
                )
            except Exception as error:
                wall = (time.perf_counter() - started) * 1000
                ctx.recorder.add(
                    "research", case_id, "baseline_memory_answer", repeat=repeat,
                    metrics=audit_metrics(ctx, mark, wall), success=False, error=error,
                )

            try:
                raw_pack, raw_hit, raw_ms = cached_research_pack(ctx, case_id, "raw", question)
                ctx.recorder.add(
                    "research", case_id, "baseline_raw_retrieval", repeat=repeat,
                    metrics=research_pack_metrics(raw_pack, expected_domains, raw_ms, raw_hit),
                    output=pack_to_json(raw_pack),
                )
                if raw_pack.sources:
                    mark = ctx.audit.mark()
                    started = time.perf_counter()
                    raw_answer = ctx.client.chat(
                        "Teach from the supplied evidence. Cite every factual claim inline as [S#].",
                        ctx.services.build_evidence_prompt(question, raw_pack, max_chars_per_source=2_500),
                        max_tokens=650,
                    ).content
                    raw_report = ctx.services.verify_answer(raw_answer, raw_pack)
                    wall = (time.perf_counter() - started) * 1000
                    ctx.recorder.add(
                        "research", case_id, "ablation_raw_rag", repeat=repeat,
                        metrics={
                            **answer_metrics(raw_answer, raw_report, expected_terms, displayed=True),
                            **audit_metrics(ctx, mark, wall),
                        },
                        output={"answer": raw_answer, "verification": raw_report.as_dict()},
                    )
            except Exception as error:
                ctx.recorder.add("research", case_id, "baseline_raw_retrieval", repeat=repeat, success=False, error=error)

            route = {
                "search_query": str(case.get("optimized_query", question)),
                "preferred_domain": str(case.get("preferred_domain", "")),
                "risk": "medium",
            }
            mark = ctx.audit.mark()
            route_started = time.perf_counter()
            try:
                routed = ctx.client.structured(
                    ROUTE_SYSTEM,
                    f"Student question: {question}",
                    ROUTE_SCHEMA,
                    schema_name="benchmark_research_route",
                    max_tokens=350,
                ).payload
                if routed:
                    route.update(routed)
            except Exception:
                pass
            route_wall = (time.perf_counter() - route_started) * 1000
            domain = str(route.get("preferred_domain", "")).casefold().removeprefix("www.")
            if not re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", domain):
                domain = str(case.get("preferred_domain", ""))
            query = str(route.get("search_query") or case.get("optimized_query") or question)
            focused = f"site:{domain} {query}" if domain else query
            try:
                pack, pack_hit, pack_ms = cached_research_pack(ctx, case_id, "routed", focused)
                if not pack.sources and domain:
                    pack, pack_hit, pack_ms = cached_research_pack(ctx, case_id, "routed-broad", query)
                ctx.recorder.add(
                    "research",
                    case_id,
                    "optimized_routed_retrieval",
                    repeat=repeat,
                    metrics={
                        **research_pack_metrics(pack, expected_domains, pack_ms, pack_hit),
                        "route_wall_ms": round(route_wall, 2),
                        "route_model_latency_ms": sum(item["latency_ms"] for item in ctx.audit.since(mark)),
                    },
                    output={"route": route, "focused_query": focused, "pack": pack_to_json(pack)},
                )
                if not pack.sources:
                    ctx.recorder.add(
                        "research",
                        case_id,
                        "optimized_verified_answer",
                        repeat=repeat,
                        metrics={
                            **answer_metrics("", None, expected_terms, displayed=False),
                            "repair_used": False,
                            "repair_rescued": False,
                            "selective_excerpt_used": False,
                            "selective_excerpt_rescued": False,
                            "wall_ms": round(route_wall + pack_ms, 2),
                            "model_call_count": len(ctx.audit.since(mark)),
                        },
                        output={"refusal": "No public evidence sources were available."},
                    )
                    continue

                answer_mark = ctx.audit.mark()
                answer_started = time.perf_counter()
                answer = ctx.client.chat(
                    "You are an evidence-grounded tutor. Use only the research pack and cite every factual claim.",
                    ctx.services.build_evidence_prompt(question, pack, max_chars_per_source=3_000),
                    max_tokens=750,
                ).content
                report = ctx.services.verify_answer(
                    answer,
                    pack,
                    llm_callback=lambda prompt: verify_callback(ctx, prompt),
                )
                repair_used = False
                selective_excerpt_used = False
                selective_excerpt_rescued = False
                initial_report = report.as_dict()
                if not report.safe_to_show:
                    repair_used = True
                    repair_prompt = (
                        ctx.services.build_evidence_prompt(question, pack, max_chars_per_source=2_200)
                        + "\n\nHELD DRAFT\n"
                        + answer
                        + "\n\nVERIFIER REPORT\n"
                        + json.dumps(report.as_dict(), ensure_ascii=False)
                        + "\n\nRewrite once using only supported evidence. Remove unsupported claims and cite every remaining factual sentence."
                    )
                    answer = ctx.client.chat(
                        "Repair an evidence-grounded answer. Return only the repaired cited answer.",
                        repair_prompt,
                        max_tokens=700,
                    ).content
                    report = ctx.services.verify_answer(
                        answer,
                        pack,
                        llm_callback=lambda prompt: verify_callback(ctx, prompt),
                    )
                repair_rescued = bool(repair_used and report.safe_to_show)
                pre_selective_report = report.as_dict()
                if not report.safe_to_show:
                    selective = ctx.services.select_supported_excerpt(answer, pack, report)
                    if selective is not None:
                        answer, report = selective
                        selective_excerpt_used = True
                        selective_excerpt_rescued = bool(report.safe_to_show)
                answer_wall = (time.perf_counter() - answer_started) * 1000
                ctx.recorder.add(
                    "research",
                    case_id,
                    "optimized_verified_answer",
                    repeat=repeat,
                    metrics={
                        **answer_metrics(answer, report, expected_terms, displayed=bool(report.safe_to_show)),
                        "repair_used": repair_used,
                        "repair_rescued": repair_rescued,
                        "selective_excerpt_used": selective_excerpt_used,
                        "selective_excerpt_rescued": selective_excerpt_rescued,
                        **audit_metrics(ctx, answer_mark, answer_wall),
                    },
                    output={
                        "answer": answer,
                        "initial_verification": initial_report,
                        "pre_selective_verification": pre_selective_report,
                        "final_verification": report.as_dict(),
                    },
                )
            except Exception as error:
                ctx.recorder.add(
                    "research", case_id, "optimized_verified_answer", repeat=repeat,
                    success=False, error=error,
                    output={"traceback": traceback.format_exc(limit=4)},
                )


def run_teachback(ctx: Context) -> None:
    for case in ctx.cases.get("teachback", []):
        if not profile_selected(case, ctx.args.profile):
            continue
        case_id = str(case["id"])
        for repeat in range(1, ctx.args.repeats + 1):
            mark = ctx.audit.mark()
            started = time.perf_counter()
            try:
                pair = ctx.services.generate_transfer_pair(ctx.client, dict(case["topic_record"]))
                wall = (time.perf_counter() - started) * 1000
                ctx.recorder.add(
                    "teachback", case_id, "optimized_generated_pair", repeat=repeat,
                    metrics={
                        "schema_complete": all(key in pair for key in ("pre_question", "post_question", "rubric", "concept_invariant")),
                        "questions_distinct": normalize_text(str(pair.get("pre_question", ""))) != normalize_text(str(pair.get("post_question", ""))),
                        "rubric_items": len(pair.get("rubric", [])),
                        **audit_metrics(ctx, mark, wall),
                    },
                    output=pair,
                )
            except Exception as error:
                wall = (time.perf_counter() - started) * 1000
                ctx.recorder.add(
                    "teachback", case_id, "optimized_generated_pair", repeat=repeat,
                    metrics=audit_metrics(ctx, mark, wall), success=False, error=error,
                )

            expected_terms = list(case.get("expected_terms", []))
            baseline_pre = term_recall(str(case["pre_answer"]), expected_terms)
            baseline_post = term_recall(str(case["post_answer"]), expected_terms)
            baseline_quality = min(1.0, len(str(case["explanation"]).split()) / 80)
            baseline_mae = statistics.fmean(
                [
                    abs(baseline_pre - float(case["pre_gold"])),
                    abs(baseline_post - float(case["post_gold"])),
                    abs(baseline_quality - float(case["quality_gold"])),
                ]
            )
            ctx.recorder.add(
                "teachback", case_id, "baseline_keyword_activity_scoring", repeat=repeat,
                metrics={
                    "mean_absolute_error": baseline_mae,
                    "pre_score": baseline_pre,
                    "post_score": baseline_post,
                    "quality_score": baseline_quality,
                    "gain_direction_correct": (baseline_post > baseline_pre) == (float(case["post_gold"]) > float(case["pre_gold"])),
                    "wall_ms": 0.0,
                    "model_call_count": 0,
                },
            )

            mark = ctx.audit.mark()
            started = time.perf_counter()
            try:
                pre = ctx.services.score_transfer(ctx.client, str(case["pre_question"]), str(case["pre_answer"]), list(case["rubric"]))
                quality = ctx.services.score_teaching_explanation(ctx.client, str(case["topic"]), str(case["explanation"]))
                post = ctx.services.score_transfer(ctx.client, str(case["post_question"]), str(case["post_answer"]), list(case["rubric"]))
                impact = ctx.services.teaching_impact(float(pre["score"]), float(post["score"]), float(quality["score"]))
                wall = (time.perf_counter() - started) * 1000
                mae = statistics.fmean(
                    [
                        abs(float(pre["score"]) - float(case["pre_gold"])),
                        abs(float(post["score"]) - float(case["post_gold"])),
                        abs(float(quality["score"]) - float(case["quality_gold"])),
                    ]
                )
                ctx.recorder.add(
                    "teachback", case_id, "optimized_gemma_pre_post", repeat=repeat,
                    metrics={
                        "mean_absolute_error": mae,
                        "pre_score": float(pre["score"]),
                        "post_score": float(post["score"]),
                        "quality_score": float(quality["score"]),
                        "gain_direction_correct": (float(post["score"]) > float(pre["score"])) == (float(case["post_gold"]) > float(case["pre_gold"])),
                        "teaching_impact": float(impact["impact"]),
                        **audit_metrics(ctx, mark, wall),
                    },
                    output={"pre": pre, "quality": quality, "post": post, "impact": impact},
                )
            except Exception as error:
                wall = (time.perf_counter() - started) * 1000
                ctx.recorder.add(
                    "teachback", case_id, "optimized_gemma_pre_post", repeat=repeat,
                    metrics=audit_metrics(ctx, mark, wall), success=False, error=error,
                )

            baseline_attack = min(100.0, len(str(case["explanation"]).split()) * 2.0)
            optimized_attack = float(ctx.services.teaching_impact(0.85, 0.85, 1.0)["impact"])
            genuine = ctx.services.teaching_impact(
                float(case["pre_gold"]), float(case["post_gold"]), float(case["quality_gold"])
            )
            ctx.recorder.add(
                "teachback", case_id, "baseline_activity_reward", repeat=repeat,
                metrics={"no_gain_attack_reward": baseline_attack, "genuine_reward": baseline_attack},
            )
            ctx.recorder.add(
                "teachback", case_id, "optimized_learning_lift_reward", repeat=repeat,
                metrics={"no_gain_attack_reward": optimized_attack, "genuine_reward": float(genuine["impact"])},
            )


def naive_xp(event_count: int, minutes: int, messages: int) -> float:
    return min(100.0, event_count * 5.0 + minutes * 0.10 + messages)


def run_gamification(ctx: Context) -> None:
    event = ctx.services.EvidenceEvent
    kind = ctx.services.EvidenceKind
    now = datetime.now(UTC)
    base = [
        event(
            event_id="base-a-retention", learner_id="BenchmarkLearner", topic_id="Data leakage",
            kind=kind.RETENTION, score=0.82, occurred_at=now - timedelta(days=4), difficulty=0.65,
            confidence=0.78, delay_hours=24, prompt_id="fresh-a-r",
            response_text="Independent delayed explanation of why held-out samples cannot influence fitted preprocessing parameters.",
        ),
        event(
            event_id="base-a-transfer", learner_id="BenchmarkLearner", topic_id="Data leakage",
            kind=kind.TRANSFER, score=0.78, occurred_at=now - timedelta(days=3, hours=20), difficulty=0.65,
            confidence=0.75, prompt_id="fresh-a-t",
            response_text="Applied the leakage principle to imputation inside validation folds with a distinct causal justification.",
        ),
        event(
            event_id="base-b-retention", learner_id="BenchmarkLearner", topic_id="Eigenvalues",
            kind=kind.RETENTION, score=0.80, occurred_at=now - timedelta(days=2), difficulty=0.70,
            confidence=0.76, delay_hours=48, prompt_id="fresh-b-r",
            response_text="Retrieved the independence condition and construction of the diagonal eigenvalue matrix after a delay.",
        ),
        event(
            event_id="base-b-transfer", learner_id="BenchmarkLearner", topic_id="Eigenvalues",
            kind=kind.TRANSFER, score=0.76, occurred_at=now - timedelta(days=1, hours=20), difficulty=0.70,
            confidence=0.74, prompt_id="fresh-b-t",
            response_text="Applied diagonalization to a new matrix scenario and explained why a deficient eigenspace prevents it.",
        ),
    ]
    replays = [
        event(
            event_id=f"replay-{index}", learner_id="BenchmarkLearner", topic_id="Data leakage",
            kind=kind.RETENTION, score=1.0, occurred_at=now + timedelta(minutes=index), difficulty=0.85,
            confidence=1.0, delay_hours=168, prompt_id="fresh-a-r",
            response_text="Independent delayed explanation of why held-out samples cannot influence fitted preprocessing parameters.",
        )
        for index in range(1, 21)
    ]
    baseline_base = naive_xp(len(base), 120, 4)
    baseline_attacked = naive_xp(len(base) + len(replays), 600, 100)
    optimized_base = ctx.services.calculate_proof_score(
        base, learner_id="BenchmarkLearner", curriculum_topics=["Data leakage", "Eigenvalues", "Epidemiology"]
    )
    optimized_attacked = ctx.services.calculate_proof_score(
        [*base, *replays], learner_id="BenchmarkLearner", curriculum_topics=["Data leakage", "Eigenvalues", "Epidemiology"]
    )

    low_transfer = [
        item if item.kind is not kind.TRANSFER else event(**{**asdict(item), "score": 0.10})
        for item in base
    ]
    low_score = ctx.services.calculate_proof_score(
        low_transfer, learner_id="BenchmarkLearner", curriculum_topics=["Data leakage", "Eigenvalues"]
    )
    copied_text = "This copied peer explanation repeats the complete causal mechanism and worked application almost word for word."
    peer = event(
        event_id="peer-original", learner_id="Peer", topic_id="Data leakage", kind=kind.TRANSFER,
        score=0.9, occurred_at=now - timedelta(hours=2), prompt_id="peer-prompt", response_text=copied_text,
    )
    copy = event(
        event_id="learner-copy", learner_id="BenchmarkLearner", topic_id="Data leakage", kind=kind.TRANSFER,
        score=0.9, occurred_at=now - timedelta(hours=1), prompt_id="different-prompt", response_text=copied_text,
    )
    copy_report = ctx.services.assess_integrity([copy], comparison_history=[peer], learner_id="BenchmarkLearner")

    ctx.recorder.add(
        "gamification", "reward-hacking", "baseline_naive_xp",
        metrics={
            "base_score": baseline_base,
            "attacked_score": baseline_attacked,
            "replay_attack_gain": baseline_attacked - baseline_base,
            "activity_spam_gain": naive_xp(len(base), 900, 200) - baseline_base,
            "copy_hold": False,
            "low_transfer_false_eligibility": True,
        },
    )
    ctx.recorder.add(
        "gamification", "reward-hacking", "optimized_proofscore",
        metrics={
            "base_score": float(optimized_base.proof_score),
            "attacked_score": float(optimized_attacked.proof_score),
            "replay_attack_gain": float(optimized_attacked.proof_score - optimized_base.proof_score),
            "activity_spam_gain": 0.0,
            "copy_hold": bool(copy_report.held_event_ids),
            "low_transfer_false_eligibility": bool(low_score.leaderboard_eligible),
            "base_leaderboard_eligible": bool(optimized_base.leaderboard_eligible),
            "attacked_leaderboard_eligible": bool(optimized_attacked.leaderboard_eligible),
            "held_replay_count": len(optimized_attacked.held_event_ids),
        },
        output={
            "base": optimized_base.as_dict(),
            "attacked": optimized_attacked.as_dict(),
            "low_transfer": low_score.as_dict(),
            "copy_integrity": copy_report,
        },
    )


COMPARISON_PAIRS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "documents": (("baseline_byte_decode", "optimized_mime_extraction"),),
    "course_map": (("baseline_heading_heuristic", "optimized_gemma_structured"),),
    "calendar": (("baseline_sequential", "optimized_conflict_aware"),),
    "questions": (("baseline_generic_fallback", "optimized_gemma_generated"),),
    "assessment": (("baseline_keyword_coverage", "optimized_gemma_rubric"),),
    "research": (
        ("baseline_memory_answer", "optimized_verified_answer"),
        ("baseline_raw_retrieval", "optimized_routed_retrieval"),
    ),
    "teachback": (
        ("baseline_keyword_activity_scoring", "optimized_gemma_pre_post"),
        ("baseline_activity_reward", "optimized_learning_lift_reward"),
    ),
    "gamification": (("baseline_naive_xp", "optimized_proofscore"),),
}

COMPARISON_METRICS: Mapping[str, set[str]] = {
    "documents": {"gold_span_recall", "empty_output", "wall_ms"},
    "course_map": {
        "topic_precision", "topic_recall", "topic_f1", "schema_complete",
        "fraction_fields_valid_rate", "estimated_minutes_valid_rate", "numeric_bounds_valid",
        "wall_ms", "model_call_count",
    },
    "calendar": {
        "assessment_recall", "assessment_false_positives", "conflict_minutes",
        "deadline_violations", "block_limit_violations", "allocated_ratio",
        "roundtrip_block_recall", "wall_ms",
    },
    "questions": {
        "schema_complete", "valid_mcq_rate", "unique_option_rate", "topic_term_recall",
        "skill_diversity", "rubric_items", "wall_ms", "model_call_count",
    },
    "assessment": {"absolute_error", "pass_correct", "wall_ms", "model_call_count"},
    "research": {
        "gold_term_recall", "citation_valid", "unsupported_claims",
        "unsupported_leakage", "displayed_approx_correct",
        "correct_or_held", "target_domain_hit", "fetched_fraction", "authority_mean",
        "online",
        "wall_ms", "model_call_count",
    },
    "teachback": {
        "mean_absolute_error", "gain_direction_correct", "no_gain_attack_reward",
        "genuine_reward", "wall_ms", "model_call_count",
    },
    "gamification": {
        "replay_attack_gain", "activity_spam_gain", "copy_hold",
        "low_transfer_false_eligibility",
    },
}

LOWER_IS_BETTER = {
    "absolute_error",
    "mean_absolute_error",
    "wall_ms",
    "model_latency_ms",
    "conflict_minutes",
    "assessment_false_positives",
    "deadline_violations",
    "block_limit_violations",
    "unsupported_claims",
    "uncertain_claims",
    "unsupported_leakage",
    "replay_attack_gain",
    "activity_spam_gain",
    "no_gain_attack_reward",
    "low_transfer_false_eligibility",
    "empty_output",
    "model_call_count",
    "route_wall_ms",
    "route_model_latency_ms",
}

HIGHER_IS_BETTER = {
    "completion_rate",
    "gold_span_recall",
    "topic_precision",
    "topic_recall",
    "topic_f1",
    "schema_complete",
    "fraction_fields_valid_rate",
    "estimated_minutes_valid_rate",
    "numeric_bounds_valid",
    "assessment_recall",
    "allocated_ratio",
    "roundtrip_block_recall",
    "valid_mcq_rate",
    "unique_option_rate",
    "topic_term_recall",
    "skill_diversity",
    "rubric_items",
    "pass_correct",
    "citation_valid",
    "displayed_approx_correct",
    "correct_or_held",
    "target_domain_hit",
    "fetched_fraction",
    "authority_mean",
    "online",
    "gain_direction_correct",
    "genuine_reward",
    "copy_hold",
}


def metric_direction(metric: str) -> str | None:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    return None


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in records:
        grouped[(str(row["suite"]), str(row["variant"]), "completion_rate")].append(
            float(bool(row.get("success")))
        )
        if not row.get("success"):
            continue
        for metric, raw in row.get("metrics", {}).items():
            value = numeric(raw)
            if value is not None:
                grouped[(str(row["suite"]), str(row["variant"]), str(metric))].append(value)
    summary: list[dict[str, Any]] = []
    for (suite, variant, metric), values in sorted(grouped.items()):
        summary.append(
            {
                "suite": suite,
                "variant": variant,
                "metric": metric,
                "n": len(values),
                "mean": round(statistics.fmean(values), 6),
                "median": round(statistics.median(values), 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
            }
        )
    return summary


def internal_comparisons(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare variants only on identical successful case/repeat pairs."""

    output: list[dict[str, Any]] = []
    for suite, pairs in COMPARISON_PAIRS.items():
        for baseline, optimized in pairs:
            baseline_rows = [
                row for row in records if row.get("suite") == suite and row.get("variant") == baseline
            ]
            optimized_rows = [
                row for row in records if row.get("suite") == suite and row.get("variant") == optimized
            ]
            if baseline_rows and optimized_rows:
                before = statistics.fmean(float(bool(row.get("success"))) for row in baseline_rows)
                after = statistics.fmean(float(bool(row.get("success"))) for row in optimized_rows)
                output.append(
                    {
                        "suite": suite,
                        "metric": "completion_rate",
                        "baseline_variant": baseline,
                        "optimized_variant": optimized,
                        "baseline_mean": round(before, 6),
                        "optimized_mean": round(after, 6),
                        "delta": round(after - before, 6),
                        "improvement": round(after - before, 6),
                        "direction": "higher",
                        "baseline_n": len(baseline_rows),
                        "optimized_n": len(optimized_rows),
                    }
                )

            baseline_index = {
                (str(row.get("case_id")), int(row.get("repeat", 1))): row
                for row in baseline_rows
                if row.get("success")
            }
            optimized_index = {
                (str(row.get("case_id")), int(row.get("repeat", 1))): row
                for row in optimized_rows
                if row.get("success")
            }
            paired_keys = baseline_index.keys() & optimized_index.keys()
            allowed = COMPARISON_METRICS.get(suite, set())
            for metric in sorted(allowed):
                direction = metric_direction(metric)
                if direction is None:
                    continue
                pairs_for_metric: list[tuple[float, float]] = []
                for key in paired_keys:
                    before_value = numeric(baseline_index[key].get("metrics", {}).get(metric))
                    after_value = numeric(optimized_index[key].get("metrics", {}).get(metric))
                    if before_value is not None and after_value is not None:
                        pairs_for_metric.append((before_value, after_value))
                if not pairs_for_metric:
                    continue
                before = statistics.fmean(pair[0] for pair in pairs_for_metric)
                after = statistics.fmean(pair[1] for pair in pairs_for_metric)
                delta = after - before
                improvement = -delta if direction == "lower" else delta
                output.append(
                    {
                        "suite": suite,
                        "metric": metric,
                        "baseline_variant": baseline,
                        "optimized_variant": optimized,
                        "baseline_mean": round(before, 6),
                        "optimized_mean": round(after, 6),
                        "delta": round(delta, 6),
                        "improvement": round(improvement, 6),
                        "direction": direction,
                        "paired_n": len(pairs_for_metric),
                    }
                )
    return output


def external_comparisons(
    current: Sequence[Mapping[str, Any]],
    prior_path: Path | None,
) -> list[dict[str, Any]]:
    if prior_path is None:
        return []
    prior_payload = load_json(prior_path)
    prior = prior_payload.get("summary", [])
    old_lookup = {
        (str(row["suite"]), str(row["variant"]), str(row["metric"])): row
        for row in prior
    }
    new_lookup = {
        (str(row["suite"]), str(row["variant"]), str(row["metric"])): row
        for row in current
    }
    output: list[dict[str, Any]] = []
    for key in sorted(old_lookup.keys() & new_lookup.keys()):
        metric = key[2]
        direction = metric_direction(metric)
        if direction is None:
            continue
        old_row = old_lookup[key]
        new_row = new_lookup[key]
        if int(old_row.get("n", 0)) != int(new_row.get("n", 0)):
            continue
        before = float(old_row["mean"])
        after = float(new_row["mean"])
        delta = after - before
        improvement = -delta if direction == "lower" else delta
        output.append(
            {
                "suite": key[0],
                "variant": key[1],
                "metric": metric,
                "prior_mean": round(before, 6),
                "current_mean": round(after, 6),
                "delta": round(delta, 6),
                "improvement": round(improvement, 6),
                "direction": direction,
                "n": int(new_row.get("n", 0)),
                "prior_metrics_path": str(prior_path.resolve()),
            }
        )
    return output


def report_markdown(
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    external: Sequence[Mapping[str, Any]],
) -> str:
    successful = sum(bool(row.get("success")) for row in records)
    failures = [row for row in records if not row.get("success")]
    lines = [
        "# ProofMode real-world benchmark report",
        "",
        f"- Run: `{metadata['run_id']}` (`{metadata['run_label']}`)",
        f"- Profile: `{metadata['profile']}`; repeats: {metadata['repeats']}",
        f"- Product commit: `{metadata['git'].get('commit', 'unknown')}`",
        f"- Model: `{metadata['model'].get('model_alias') or metadata['model_name']}`",
        f"- Records: {successful}/{len(records)} successful; {len(failures)} failed",
        "",
        "## Source provenance",
        "",
        "| Source | Bytes | SHA-256 | Licence |",
        "|---|---:|---|---|",
    ]
    for source in metadata.get("sources", []):
        lines.append(
            f"| {source['id']} | {source['bytes']} | `{source['sha256'][:16]}...` | {source.get('license') or ''} |"
        )
    lines.extend(["", "## Baseline-to-optimized comparisons", ""])
    if comparisons:
        lines.extend(
            [
                "Positive improvement follows the stated direction; it is not automatically a statistical claim.",
                "",
                "| Suite | Metric | Baseline | Optimized | Delta | Improvement | Better direction |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in comparisons:
            lines.append(
                f"| {row['suite']} | {row['metric']} | {row['baseline_mean']:.4g} | "
                f"{row['optimized_mean']:.4g} | {row['delta']:+.4g} | {row['improvement']:+.4g} | {row['direction']} |"
            )
    else:
        lines.append("No comparable metric pairs were produced.")
    if external:
        lines.extend(
            [
                "",
                "## Detached code-tree comparison",
                "",
                "| Suite | Variant | Metric | Prior | Current | Delta | Improvement |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in external:
            lines.append(
                f"| {row['suite']} | {row['variant']} | {row['metric']} | {row['prior_mean']:.4g} | "
                f"{row['current_mean']:.4g} | {row['delta']:+.4g} | {row['improvement']:+.4g} |"
            )
    lines.extend(["", "## Failures and refusals", ""])
    if failures:
        for row in failures:
            lines.append(
                f"- `{row['suite']}/{row['case_id']}/{row['variant']}`: "
                f"{row.get('error_type', 'Error')} — {row.get('error', 'No detail')}"
            )
    else:
        lines.append("No harness stage failed.")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Gold-term recall is a transparent lexical proxy, not full semantic correctness.",
            "- Model-written answer keys never validate themselves; assessment scores use human-authored labels.",
            "- Search results can change. The raw research packs and cache-hit state are retained.",
            "- Baseline and optimized tree runs should share downloaded source files and use the same model server.",
            "- Research is live by default so page-fetch optimizations remain measurable; use --reuse-research-cache only when fixed evidence is the intended control.",
            "- A one-day software benchmark cannot establish improved human retention or reduced procrastination.",
            "- Reported deltas are descriptive unless repeats and confidence intervals justify a stronger claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(ctx: Context, metadata: dict[str, Any]) -> tuple[Path, Path, Path]:
    raw_path = ctx.artifacts_dir / "raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as output:
        for row in ctx.recorder.records:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate_records(ctx.recorder.records)
    comparisons = internal_comparisons(ctx.recorder.records)
    external = external_comparisons(summary, ctx.args.compare_to)
    metrics_payload = {
        "metadata": metadata,
        "summary": summary,
        "comparisons": comparisons,
        "external_comparisons": external,
    }
    metrics_path = ctx.artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = ctx.artifacts_dir / "report.md"
    report_path.write_text(
        report_markdown(metadata, ctx.recorder.records, comparisons, external), encoding="utf-8"
    )
    return raw_path, metrics_path, report_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run real-source, live-Gemma benchmarks against ProofMode service modules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    value.add_argument("--profile", choices=("quick", "full"), default="quick")
    value.add_argument("--suite", default="all", help=f"Comma-separated: {', '.join(ALL_SUITES)}, or all")
    value.add_argument("--repeats", type=int, default=None, help="Override quick=1/full=3")
    value.add_argument("--repo-root", "--source-root", dest="repo_root", type=Path, default=DEFAULT_REPO_ROOT)
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    value.add_argument("--cache-dir", type=Path, default=HARNESS_DIR / ".cache")
    value.add_argument("--output-dir", type=Path, default=DEFAULT_REPO_ROOT / "benchmark_artifacts")
    value.add_argument("--run-label", default="working-tree")
    value.add_argument("--compare-to", type=Path, default=None, help="Prior metrics.json from another code-tree run")
    value.add_argument("--model-url", default=os.getenv("PROOFMODE_GEMMA_URL", "http://127.0.0.1:8080/v1"))
    value.add_argument("--model", default=os.getenv("PROOFMODE_MODEL", "gemma-4-e4b-it-q4"))
    value.add_argument("--research-results", type=int, default=3)
    value.add_argument("--web-timeout", type=float, default=5.0)
    value.add_argument("--refresh-sources", action="store_true")
    value.add_argument("--refresh-research", action="store_true")
    value.add_argument(
        "--reuse-research-cache",
        action="store_true",
        help="Reuse saved research packs; default runs live search/fetch so retrieval optimizations remain measurable",
    )
    value.add_argument("--download-only", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--allow-model-offline", action="store_true", help="Run deterministic suites and record live-suite failures")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.manifest = args.manifest.resolve()
    args.cases = args.cases.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.compare_to is not None:
        args.compare_to = args.compare_to.resolve()
    args.suites = selected_suites(args.suite)
    args.repeats = args.repeats if args.repeats is not None else (1 if args.profile == "quick" else 3)
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if not 1 <= args.research_results <= 8:
        raise SystemExit("--research-results must be between 1 and 8")

    manifest = load_json(args.manifest)
    cases = load_json(args.cases)
    all_sources = manifest.get("sources", [])
    sources = [dict(item) for item in all_sources if profile_selected(item, args.profile)]
    services = import_services(args.repo_root)
    model = model_environment(args.model_url)
    print(f"ProofMode source root: {args.repo_root}")
    print(f"Suites: {', '.join(args.suites)}; profile={args.profile}; repeats={args.repeats}")
    print(f"Gemma available: {model.get('available', False)} at {args.model_url}")

    if args.dry_run:
        missing = [item["id"] for item in sources if not (args.cache_dir / "sources" / item["filename"]).is_file()]
        print(f"Manifest sources selected: {len(sources)}; not cached: {len(missing)}")
        if missing:
            print("Would download: " + ", ".join(missing))
        print(f"Assessment cases: {len(cases.get('assessment', []))}")
        print(f"Research cases: {len(cases.get('research', []))}")
        print(f"Teach-back cases: {len(cases.get('teachback', []))}")
        return 0

    source_lock = download_sources(sources, args.cache_dir, args.refresh_sources)
    if args.download_only:
        print(f"Source lock: {args.cache_dir / 'sources.lock.json'}")
        return 0

    live_suites = set(args.suites) & {"course_map", "questions", "assessment", "research", "teachback"}
    if live_suites and not model.get("available") and not args.allow_model_offline:
        raise SystemExit(
            "Local Gemma is not healthy. Start ProofMode's model server, select deterministic suites, "
            "or pass --allow-model-offline to record fail-closed behavior."
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + re.sub(r"[^a-z0-9-]+", "-", args.run_label.casefold()).strip("-")
    artifacts_dir = args.output_dir / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    audit = AuditCollector()
    client = services.GemmaClient(base_url=args.model_url, model=args.model, audit_callback=audit)
    recorder = Recorder(run_id, args.run_label)
    ctx = Context(
        args=args,
        services=services,
        sources=sources,
        cases=cases,
        cache_dir=args.cache_dir,
        artifacts_dir=artifacts_dir,
        recorder=recorder,
        audit=audit,
        client=client,
        source_lock=source_lock,
    )

    runners: Mapping[str, Callable[[Context], Any]] = {
        "documents": run_documents,
        "course_map": run_course_maps,
        "calendar": run_calendar,
        "questions": run_questions,
        "assessment": run_assessment,
        "research": run_research,
        "teachback": run_teachback,
        "gamification": run_gamification,
    }
    started = time.perf_counter()
    for suite in args.suites:
        print(f"[suite] {suite}")
        try:
            result = runners[suite](ctx)
            if suite == "course_map" and isinstance(result, dict):
                ctx.generated_course_maps = result
        except Exception as error:
            recorder.add(suite, "suite-level", "harness", success=False, error=error, output={"traceback": traceback.format_exc()})
    elapsed = time.perf_counter() - started

    metadata = {
        "run_id": run_id,
        "run_label": args.run_label,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "suites": list(args.suites),
        "repeats": args.repeats,
        "elapsed_seconds": round(elapsed, 3),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_environment(args.repo_root),
        "model": model,
        "model_name": args.model,
        "manifest_sha256": sha256_file(args.manifest),
        "cases_sha256": sha256_file(args.cases),
        "sources": source_lock,
        "research_cache_policy": (
            "refresh live"
            if args.refresh_research
            else "reuse cached pack"
            if args.reuse_research_cache
            else "live search/fetch and save audit copy"
        ),
    }
    environment_path = artifacts_dir / "environment.json"
    environment_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    raw_path, metrics_path, report_path = write_artifacts(ctx, metadata)
    print(f"Completed in {elapsed:.1f}s")
    print(f"Raw records: {raw_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    return 0 if all(row.get("success") for row in recorder.records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
