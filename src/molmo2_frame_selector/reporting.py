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
    output: dict[str, Any] = {
        "examples": len(records),
        "scored_examples": len(scored),
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
        values = _numeric(records, key)
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

    summaries = {mode: _summarize_group(records) for mode, records in sorted(by_mode.items())}
    baseline_accuracy = summaries.get(baseline_mode, {}).get("accuracy")
    for mode, summary in summaries.items():
        accuracy = summary.get("accuracy")
        summary["accuracy_delta_vs_baseline"] = (
            accuracy - baseline_accuracy
            if accuracy is not None and baseline_accuracy is not None
            else None
        )
    return {"baseline_mode": baseline_mode, "modes": summaries}
