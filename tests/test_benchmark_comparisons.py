from __future__ import annotations

import json

from benchmarks.run_benchmark import (
    aggregate_records,
    external_comparisons,
    internal_comparisons,
)


def _row(case_id: str, variant: str, *, success: bool = True, **metrics: float) -> dict:
    return {
        "suite": "documents",
        "case_id": case_id,
        "variant": variant,
        "repeat": 1,
        "success": success,
        "metrics": metrics,
    }


def test_internal_comparison_is_matched_and_reports_completion() -> None:
    rows = [
        _row("shared", "baseline_byte_decode", gold_span_recall=0.2, wall_ms=5),
        _row("baseline-only", "baseline_byte_decode", gold_span_recall=1.0, wall_ms=5),
        _row("shared", "optimized_mime_extraction", gold_span_recall=0.8, wall_ms=3),
        _row("failed", "optimized_mime_extraction", success=False),
    ]

    comparisons = internal_comparisons(rows)
    recall = next(row for row in comparisons if row["metric"] == "gold_span_recall")
    completion = next(row for row in comparisons if row["metric"] == "completion_rate")

    assert recall["baseline_mean"] == 0.2
    assert recall["optimized_mean"] == 0.8
    assert recall["paired_n"] == 1
    assert completion["baseline_mean"] == 1.0
    assert completion["optimized_mean"] == 0.5


def test_aggregate_includes_failures_in_completion_rate() -> None:
    summary = aggregate_records(
        [
            _row("one", "optimized_mime_extraction", gold_span_recall=0.8),
            _row("two", "optimized_mime_extraction", success=False),
        ]
    )

    completion = next(row for row in summary if row["metric"] == "completion_rate")
    assert completion["mean"] == 0.5
    assert completion["n"] == 2


def test_external_comparison_omits_unknown_and_unmatched_metrics(tmp_path) -> None:
    prior = {
        "summary": [
            {"suite": "x", "variant": "v", "metric": "model_call_count", "mean": 3, "n": 2},
            {"suite": "x", "variant": "v", "metric": "attacked_score", "mean": 4, "n": 2},
            {"suite": "x", "variant": "v", "metric": "wall_ms", "mean": 10, "n": 3},
        ]
    }
    prior_path = tmp_path / "metrics.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    current = [
        {"suite": "x", "variant": "v", "metric": "model_call_count", "mean": 2, "n": 2},
        {"suite": "x", "variant": "v", "metric": "attacked_score", "mean": 9, "n": 2},
        {"suite": "x", "variant": "v", "metric": "wall_ms", "mean": 5, "n": 2},
    ]

    comparisons = external_comparisons(current, prior_path)

    assert [row["metric"] for row in comparisons] == ["model_call_count"]
    assert comparisons[0]["direction"] == "lower"
    assert comparisons[0]["improvement"] == 1.0
