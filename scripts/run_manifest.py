"""Run a JSONL video-QA manifest while keeping both models loaded."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from time import perf_counter

from molmo2_frame_selector.baselines import ALL_MODES, SELECTOR_MODES
from molmo2_frame_selector.cache import SQLiteEmbeddingCache
from molmo2_frame_selector.encoders import Siglip2Encoder
from molmo2_frame_selector.experiment import VideoQAExperiment, load_jsonl_manifest
from molmo2_frame_selector.hf_runner import Molmo2HFRunner
from molmo2_frame_selector.metrics import JsonlWriter
from molmo2_frame_selector.video import DecordCandidateDecoder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=ALL_MODES, required=True)
    parser.add_argument("--num-selected", type=int, default=32)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--official-max-frames", type=int, default=384)
    parser.add_argument("--max-fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-id", default="allenai/Molmo2-4B")
    parser.add_argument("--selector-model-id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--selector-revision", default="main")
    parser.add_argument("--selector-device", default=None)
    parser.add_argument("--selector-batch-size", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--cache", type=Path, default=Path("outputs/embedding_cache.sqlite3"))
    parser.add_argument("--measure-ttft", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--cache-state", choices=("reuse", "cold", "warm"), default="reuse")
    parser.add_argument("--continue-on-error", action="store_true")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--overwrite", action="store_true")
    output_group.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 0 or (args.limit is not None and args.limit < 0):
        raise ValueError("warmup and limit must be non-negative")
    if min(args.num_selected, args.num_candidates, args.official_max_frames,
           args.max_new_tokens, args.selector_batch_size) <= 0:
        raise ValueError("frame/token budgets and selector batch size must be positive")
    if args.num_selected > args.num_candidates:
        raise ValueError("num_selected cannot exceed num_candidates")
    examples = load_jsonl_manifest(args.manifest)
    if args.limit is not None:
        examples = examples[: args.limit]

    configuration = {
        "schema_version": 2, "mode": args.mode,
        "model_id": args.model_id, "model_revision": args.model_revision,
        "selector_model_id": args.selector_model_id if args.mode in SELECTOR_MODES else None,
        "selector_revision": args.selector_revision if args.mode in SELECTOR_MODES else None,
        "num_selected_requested": args.num_selected, "num_candidates_requested": args.num_candidates,
        "official_max_frames": args.official_max_frames, "max_fps": args.max_fps,
        "max_new_tokens": args.max_new_tokens, "dtype": args.dtype,
        "attention_backend": args.attention_backend, "cache_state": args.cache_state,
        "measure_ttft": args.measure_ttft, "warmup": args.warmup,
        "selector_batch_size": args.selector_batch_size, "selector_device": args.selector_device,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selector_implementation": "incremental-v2", "sampling_policy": "legacy-round-v1",
    }
    previous_records = []
    completed_ids: set[str] = set()
    if args.output.exists() and args.output.stat().st_size > 0:
        if args.overwrite:
            args.output.write_text("", encoding="utf-8")
        elif args.resume:
            with args.output.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        record = json.loads(line)
                        for key, value in configuration.items():
                            if record.get(key) != value:
                                raise ValueError(f"cannot resume with changed/missing {key}; use a new output file")
                        previous_records.append(record)
                        completed_ids.add(str(record["example_id"]))
            examples = [example for example in examples if example.example_id not in completed_ids]
        else:
            raise FileExistsError(
                f"output already exists: {args.output}; use --resume or --overwrite"
            )

    if not examples:
        print(json.dumps({"remaining_examples": 0}))
        return 1 if any(row.get("status") == "error" for row in previous_records) else 0

    decoder = DecordCandidateDecoder(max_fps=args.max_fps)
    setup_start = perf_counter()
    runner = Molmo2HFRunner(
        model_id=args.model_id,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        revision=args.model_revision,
    )
    encoder = None
    cache = None
    if args.mode in SELECTOR_MODES:
        encoder = Siglip2Encoder(
            model_id=args.selector_model_id,
            device=args.selector_device,
            batch_size=args.selector_batch_size,
            revision=args.selector_revision,
        )
        if args.cache_state != "cold":
            cache = SQLiteEmbeddingCache(args.cache)

    configuration.update(
        resolved_model_revision=runner.resolved_revision,
        resolved_selector_revision=encoder.resolved_revision if encoder is not None else None,
    )
    for record in previous_records:
        for key in ("resolved_model_revision", "resolved_selector_revision"):
            if record.get(key) != configuration[key]:
                if cache is not None:
                    cache.close()
                raise ValueError(f"cannot resume after checkpoint change: {key}")
    setup_seconds = perf_counter() - setup_start

    try:
        experiment = VideoQAExperiment(
            mode=args.mode,
            decoder=decoder,
            molmo_runner=runner,
            encoder=encoder,
            embedding_cache=cache,
            num_selected=args.num_selected,
            num_candidates=args.num_candidates,
            official_max_frames=args.official_max_frames,
            max_new_tokens=args.max_new_tokens,
            measure_ttft=args.measure_ttft,
        )
        writer = JsonlWriter(args.output)
        # Warm-cache prepopulation is outside measured per-example latency.
        warmup_examples = examples if args.cache_state == "warm" else examples[:args.warmup]
        warmup_start = perf_counter()
        for example in warmup_examples:
            warmup_outcome = experiment.run_safely(example)
            if warmup_outcome.record.status != "ok":
                print(json.dumps({"event": "warmup_error", "example_id": example.example_id,
                                  "error": warmup_outcome.record.error_message}), flush=True)
        metadata = {**configuration, "setup_seconds": setup_seconds,
                    "warmup_seconds": perf_counter() - warmup_start,
                    "measured_examples": len(examples), "resumed_examples": len(completed_ids)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".run.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        correct = 0
        scored = 0
        failed = sum(row.get("status") == "error" for row in previous_records)
        for index, example in enumerate(examples, start=1):
            outcome = experiment.run_safely(example)
            payload = outcome.as_dict()
            payload.update(configuration)
            payload.update(start_seconds=example.start_seconds, end_seconds=example.end_seconds)
            writer.append(payload)
            if outcome.record.correct is not None:
                scored += 1
                correct += int(outcome.record.correct)
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(examples)}",
                        "example_id": example.example_id,
                        "correct": outcome.record.correct,
                        "status": outcome.record.status,
                        "error": outcome.record.error_message,
                        "running_accuracy": correct / scored if scored else None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if outcome.record.status == "error":
                failed += 1
                if not args.continue_on_error:
                    return 1
                runner.release_unused_memory()
    finally:
        if cache is not None:
            cache.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
