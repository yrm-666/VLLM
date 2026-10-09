"""Single-video smoke/evaluation entry point for the GPU server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from molmo2_frame_selector.baselines import ALL_MODES, SELECTOR_MODES
from molmo2_frame_selector.cache import SQLiteEmbeddingCache
from molmo2_frame_selector.encoders import Siglip2Encoder
from molmo2_frame_selector.hf_runner import Molmo2HFRunner
from molmo2_frame_selector.experiment import VideoQAExample, VideoQAExperiment
from molmo2_frame_selector.video import DecordCandidateDecoder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--mode", choices=ALL_MODES, default="query_aware")
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
    parser.add_argument("--cache-state", choices=("reuse", "cold", "warm"), default="reuse")
    parser.add_argument("--start-seconds", type=float, default=None)
    parser.add_argument("--end-seconds", type=float, default=None)
    parser.add_argument("--measure-ttft", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.num_selected <= 0 or args.num_candidates <= 0 or args.official_max_frames <= 0:
        raise ValueError("frame budgets must be positive")
    if args.num_selected > args.num_candidates:
        raise ValueError("num-selected cannot exceed num-candidates")
    if args.max_new_tokens <= 0 or args.selector_batch_size <= 0:
        raise ValueError("token budget and selector batch size must be positive")
    decoder = DecordCandidateDecoder(max_fps=args.max_fps)
    example = VideoQAExample(
        example_id="single-video", video_path=args.video.expanduser().resolve(strict=True),
        question=args.question, options=tuple(args.option),
        start_seconds=args.start_seconds, end_seconds=args.end_seconds,
    )
    process_start = perf_counter()
    runner = Molmo2HFRunner(
        model_id=args.model_id, dtype=args.dtype,
        attention_backend=args.attention_backend, revision=args.model_revision,
    )
    encoder = None
    cache = None
    if args.mode in SELECTOR_MODES:
        encoder = Siglip2Encoder(
            model_id=args.selector_model_id, device=args.selector_device,
            batch_size=args.selector_batch_size, revision=args.selector_revision,
        )
        if args.cache_state != "cold":
            cache = SQLiteEmbeddingCache(args.cache)
    setup_seconds = perf_counter() - process_start
    try:
        experiment = VideoQAExperiment(
            mode=args.mode, decoder=decoder,
            molmo_runner=runner, encoder=encoder, embedding_cache=cache,
            num_selected=args.num_selected, num_candidates=args.num_candidates,
            official_max_frames=args.official_max_frames, max_new_tokens=args.max_new_tokens,
            measure_ttft=args.measure_ttft,
        )
        warmup_start = perf_counter()
        if args.cache_state == "warm":
            experiment.run(example)
        warmup_seconds = perf_counter() - warmup_start
        outcome = experiment.run(example)
        payload = outcome.as_dict()
        payload.update(
            schema_version=2, video=str(example.video_path),
            model_id=args.model_id, resolved_model_revision=runner.resolved_revision,
            selector_model_id=args.selector_model_id if encoder is not None else None,
            resolved_selector_revision=encoder.resolved_revision if encoder is not None else None,
            num_selected_requested=args.num_selected, num_candidates_requested=args.num_candidates,
            official_max_frames=args.official_max_frames, max_fps=args.max_fps,
            max_new_tokens=args.max_new_tokens, dtype=args.dtype,
            attention_backend=args.attention_backend, cache_state=args.cache_state,
            start_seconds=args.start_seconds, end_seconds=args.end_seconds,
            setup_seconds=setup_seconds, warmup_seconds=warmup_seconds,
            total_process_seconds=perf_counter() - process_start,
            selector_implementation="incremental-v2", sampling_policy="legacy-round-v1",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        if cache is not None:
            cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
