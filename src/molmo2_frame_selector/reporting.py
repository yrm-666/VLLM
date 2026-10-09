"""Aggregate JSONL experiment records into accuracy-efficiency summaries."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _numeric(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if record.get("correct") is not None]
    successful = [record for record in records if record.get("status", "ok") == "ok"]
    output: dict[str, Any] = {
        "examples": len(records),
        "scored_examples": len(scored),
        "successful_examples": len(successful),
        "failed_examples": len(records) - len(successful),
        "accuracy": (
            sum(bool(record["correct"]) for record in scored) / len(scored)
            if scored
            else None
        ),
    }
    for key in (
        "candidate_frames",
        "selected_frames",
        "estimated_visual_tokens",
        "actual_visual_tokens",
        "input_tokens",
        "output_tokens",
        "selected_time_span_fraction",
        "temporal_bins_covered",
        "temporal_bins_available",
        "cache_hits",
        "cache_misses",
        "decode_seconds",
        "selector_seconds",
        "text_encoding_seconds",
        "image_encoding_seconds",
        "scoring_selection_seconds",
        "processor_seconds",
        "ttft_seconds",
        "prefill_seconds",
        "generation_seconds",
        "end_to_end_seconds",
        "peak_vram_gb",
    ):
        values = _numeric(successful, key)
        output[f"mean_{key}"] = mean(values) if values else None
        output[f"median_{key}"] = median(values) if values else None
    return output


def summarize_jsonl(path: str | Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(payload)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("mode")), str(record.get("task")))].append(record)
    return {
        "overall": _summarize_group(records),
        "groups": {
            f"{mode}/{task}": _summarize_group(group_records)
            for (mode, task), group_records in sorted(grouped.items())
        },
    }


def compare_jsonl_files(paths: list[str | Path], baseline_mode: str = "uniform") -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for path in paths:
        before = len(seen)
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                mode = str(record.get("mode"))
                example_id = str(record.get("example_id"))
                identity = (mode, example_id)
                if identity in seen:
                    raise ValueError(
                        f"duplicate example for mode {mode}: {example_id} "
                        f"({path}, line {line_number})"
                    )
                seen.add(identity)
                by_mode[mode].append(record)
        if len(seen) == before:
            raise ValueError(f"result file contains no records: {path}")

    summaries = {mode: _summarize_group(records) for mode, records in sorted(by_mode.items())}
    if baseline_mode not in by_mode:
        raise ValueError(f"baseline mode not found: {baseline_mode}")
    baseline = {str(record["example_id"]): record for record in by_mode[baseline_mode]}
    for mode, records in by_mode.items():
        if {str(record["example_id"]) for record in records} != set(baseline):
            raise ValueError(f"mode {mode} has a different sample set from baseline")
        for record in records:
            reference = baseline[str(record["example_id"])]
            for key in ("query", "gold_label", "task", "model_id", "dtype", "max_fps",
                        "max_new_tokens", "start_seconds", "end_seconds",
                        "resolved_model_revision", "resolved_selector_revision",
                        "attention_backend", "sampling_policy"):
                if key == "resolved_selector_revision" and (
                    record.get(key) is None or reference.get(key) is None
                ):
                    # Uniform has no selector checkpoint; compare selector
                    # revisions below only among methods that use one.
                    continue
                if key in record and key in reference and record[key] != reference[key]:
                    raise ValueError(f"paired comparison differs in {key}: {record['example_id']}")
    selector_revisions = {
        record["resolved_selector_revision"]
        for records in by_mode.values() for record in records
        if record.get("resolved_selector_revision") is not None
    }
    if len(selector_revisions) > 1:
        raise ValueError("paired selector modes use different checkpoint revisions")
    baseline_accuracy = summaries.get(baseline_mode, {}).get("accuracy")
    for mode, summary in summaries.items():
        accuracy = summary.get("accuracy")
        summary["accuracy_delta_vs_baseline"] = (
            accuracy - baseline_accuracy
            if accuracy is not None and baseline_accuracy is not None
            else None
        )
    return {"baseline_mode": baseline_mode, "modes": summaries}
