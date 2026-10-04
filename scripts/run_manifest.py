"""Run a JSONL video-QA manifest while keeping both models loaded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--overwrite", action="store_true")
    output_group.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    examples = load_jsonl_manifest(args.manifest)
    if args.limit is not None:
        examples = examples[: args.limit]

    completed_ids: set[str] = set()
    if args.output.exists() and args.output.stat().st_size > 0:
        if args.overwrite:
            args.output.write_text("", encoding="utf-8")
        elif args.resume:
            with args.output.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        completed_ids.add(str(json.loads(line)["example_id"]))
            examples = [example for example in examples if example.example_id not in completed_ids]
        else:
            raise FileExistsError(
                f"output already exists: {args.output}; use --resume or --overwrite"
            )

    decoder = DecordCandidateDecoder(max_fps=args.max_fps)
    runner = Molmo2HFRunner(
        model_id=args.model_id,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
    )
    encoder = None
    cache = None
    if args.mode in SELECTOR_MODES:
        encoder = Siglip2Encoder(
            model_id=args.selector_model_id,
            device=args.selector_device,
            batch_size=args.selector_batch_size,
        )
        cache = SQLiteEmbeddingCache(args.cache)

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
        correct = 0
        scored = 0
        for index, example in enumerate(examples, start=1):
            outcome = experiment.run(example)
            payload = outcome.as_dict()
            payload.update(
                {
                    "model_id": args.model_id,
                    "selector_model_id": (
                        args.selector_model_id if args.mode in SELECTOR_MODES else None
                    ),
                    "num_selected_requested": args.num_selected,
                    "num_candidates_requested": args.num_candidates,
                    "official_max_frames": args.official_max_frames,
                    "max_fps": args.max_fps,
                    "dtype": args.dtype,
                    "attention_backend": args.attention_backend,
                }
            )
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
                        "running_accuracy": correct / scored if scored else None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if cache is not None:
            cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
