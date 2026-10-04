"""Command-line adapter for selecting frames from precomputed features."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .config import SelectorConfig
from .selector import QueryAwareFrameSelector


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select query-aware video frames from precomputed features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file containing relevance_scores and frame_embeddings.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file overriding SelectorConfig defaults.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write result JSON here; stdout is used when omitted.",
    )
    return parser


def run(input_payload: dict[str, Any], config_payload: dict[str, Any] | None) -> dict[str, Any]:
    config = SelectorConfig(**(config_payload or {}))
    selector = QueryAwareFrameSelector(config)
    result = selector.select(
        input_payload["relevance_scores"],
        input_payload["frame_embeddings"],
        timestamps=input_payload.get("timestamps"),
        candidate_indices=input_payload.get("candidate_indices"),
    )
    return {
        "query": input_payload.get("query"),
        "config": asdict(config),
        "candidate_count": result.candidate_count,
        "requested_count": result.requested_count,
        "actual_selected_count": len(result.selected_indices),
        "selected_positions": list(result.selected_positions),
        "selected_indices": list(result.selected_indices),
        "selected_timestamps": list(result.selected_timestamps),
        "selection_steps": [asdict(step) for step in result.selection_steps],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_payload = _load_json(args.input)
    if not isinstance(input_payload, dict):
        raise ValueError("input JSON must be an object")

    config_payload = _load_json(args.config) if args.config is not None else None
    if config_payload is not None and not isinstance(config_payload, dict):
        raise ValueError("config JSON must be an object")

    output_payload = run(input_payload, config_payload)
    serialized = json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

