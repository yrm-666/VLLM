"""Single-video smoke/evaluation entry point for the GPU server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from molmo2_frame_selector.baselines import ALL_MODES, SELECTOR_MODES, selector_config_for_mode
from molmo2_frame_selector.cache import (
    SQLiteEmbeddingCache,
    cache_namespace,
    video_file_identity,
)
from molmo2_frame_selector.encoders import Siglip2Encoder
from molmo2_frame_selector.hf_runner import Molmo2HFRunner
from molmo2_frame_selector.metrics import estimate_visual_tokens
from molmo2_frame_selector.pipeline import QueryAwareSelectionPipeline
from molmo2_frame_selector.query import build_query_text
from molmo2_frame_selector.selector import QueryAwareFrameSelector
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
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.num_selected <= 0 or args.num_candidates <= 0:
        raise ValueError("frame budgets must be positive")
    if args.num_selected > args.num_candidates:
        raise ValueError("num-selected cannot exceed num-candidates")

    query = build_query_text(args.question, args.option or None)
    decoder = DecordCandidateDecoder(max_fps=args.max_fps)
    total_start = perf_counter()

    selector_seconds = 0.0
    text_encoding_seconds = 0.0
    image_encoding_seconds = 0.0
    scoring_selection_seconds = 0.0
    cache_hits = 0
    cache_misses = 0
    decode_start = perf_counter()
    if args.mode == "official_original":
        decoded = decoder.decode(args.video, args.official_max_frames)
        selected_frames = tuple(decoded.frames)
        selected_indices = decoded.frame_indices
        candidate_count = len(decoded.frame_indices)
    elif args.mode == "uniform":
        decoded = decoder.decode(args.video, args.num_selected)
        selected_frames = tuple(decoded.frames)
        selected_indices = decoded.frame_indices
        candidate_count = len(decoded.frame_indices)
    else:
        decoded = decoder.decode(args.video, args.num_candidates)
        config = selector_config_for_mode(
            args.mode,
            num_selected=args.num_selected,
            num_candidates=args.num_candidates,
        )
        encoder = Siglip2Encoder(
            model_id=args.selector_model_id,
            device=args.selector_device,
            batch_size=args.selector_batch_size,
        )
        with SQLiteEmbeddingCache(args.cache) as embedding_cache:
            namespace = cache_namespace(
                encoder_fingerprint=encoder.fingerprint,
                video_identity=video_file_identity(args.video),
            )
            pipeline = QueryAwareSelectionPipeline(
                encoder,
                QueryAwareFrameSelector(config),
                embedding_cache=embedding_cache,
            )
            selection = pipeline.select(
                query,
                decoded.frames,
                timestamps=decoded.timestamps,
                frame_indices=decoded.frame_indices,
                cache_namespace=namespace,
            )
        selected_frames = selection.selected_frames
        selected_indices = selection.selection.selected_indices
        candidate_count = selection.selection.candidate_count
        selector_seconds = selection.timings.total_seconds
        text_encoding_seconds = selection.timings.text_encoding_seconds
        image_encoding_seconds = selection.timings.image_encoding_seconds
        scoring_selection_seconds = selection.timings.scoring_and_selection_seconds
        cache_hits = selection.cache_hits
        cache_misses = selection.cache_misses
    decode_and_selection_seconds = perf_counter() - decode_start
    decode_seconds = max(0.0, decode_and_selection_seconds - selector_seconds)

    runner = Molmo2HFRunner(
        model_id=args.model_id,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
    )
    generation = runner.generate(
        query,
        selected_frames,
        decoded.metadata_for(selected_indices),
        max_new_tokens=args.max_new_tokens,
        measure_ttft=args.measure_ttft,
    )
    payload = {
        "video": str(args.video.resolve()),
        "mode": args.mode,
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
        "query": query,
        "candidate_frames": candidate_count,
        "selected_frames": len(selected_indices),
        "selected_indices": list(selected_indices),
        "estimated_visual_tokens": estimate_visual_tokens(len(selected_indices)),
        "actual_visual_tokens": generation.visual_tokens,
        "input_tokens": generation.input_tokens,
        "decode_seconds": decode_seconds,
        "selector_seconds": selector_seconds,
        "text_encoding_seconds": text_encoding_seconds,
        "image_encoding_seconds": image_encoding_seconds,
        "scoring_selection_seconds": scoring_selection_seconds,
        "processor_seconds": generation.processor_seconds,
        "ttft_seconds": generation.ttft_seconds,
        "generation_seconds": generation.generation_seconds,
        "end_to_end_seconds": perf_counter() - total_start,
        "peak_vram_gb": generation.peak_vram_gb,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "prediction": generation.text,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
