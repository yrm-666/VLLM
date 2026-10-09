"""Reusable experiment orchestration for single examples and manifests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from math import isfinite

from .baselines import SELECTOR_MODES, selector_config_for_mode
from .cache import SQLiteEmbeddingCache, cache_namespace, video_file_identity
from .metrics import ExperimentRecord, estimate_visual_tokens, parse_choice_label
from .pipeline import QueryAwareSelectionPipeline
from .query import build_query_text
from .selector import QueryAwareFrameSelector
from .temporal import temporal_coverage, temporal_bin_ids


@dataclass(frozen=True, slots=True)
class VideoQAExample:
    example_id: str
    video_path: Path
    question: str
    options: tuple[str, ...] = ()
    gold_label: str | None = None
    task: str = "unknown"
    start_seconds: float | None = None
    end_seconds: float | None = None

    def __post_init__(self) -> None:
        bounds = [bound for bound in (self.start_seconds, self.end_seconds) if bound is not None]
        if any(not isfinite(bound) or bound < 0 for bound in bounds):
            raise ValueError("temporal bounds must be finite and non-negative")
        if self.end_seconds is not None and self.end_seconds <= (self.start_seconds or 0.0):
            raise ValueError("end_seconds must be greater than start_seconds")


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    record: ExperimentRecord
    query: str
    selected_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self.record),
            "query": self.query,
            "selected_indices": list(self.selected_indices),
        }


def _gold_label(answer: Any, options: tuple[str, ...]) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, int):
        if answer < 0 or answer >= len(options):
            raise ValueError("integer answer is outside the options")
        return chr(ord("A") + answer)
    answer_text = str(answer).strip()
    if len(answer_text) == 1 and answer_text.upper().isalpha():
        return answer_text.upper()
    for index, option in enumerate(options):
        if answer_text == option:
            return chr(ord("A") + index)
    raise ValueError(f"cannot map answer to an option label: {answer!r}")


def load_jsonl_manifest(path: str | Path) -> list[VideoQAExample]:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    examples: list[VideoQAExample] = []
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                options = tuple(str(option) for option in payload.get("options", []))
                video_path = Path(payload["video"])
                if not video_path.is_absolute():
                    video_path = manifest_path.parent / video_path
                example = VideoQAExample(
                        example_id=str(payload.get("id", line_number)),
                        video_path=video_path,
                        question=str(payload["question"]),
                        options=options,
                        gold_label=_gold_label(payload.get("answer"), options),
                        task=str(payload.get("task", "unknown")),
                        start_seconds=(float(payload["start_seconds"]) if payload.get("start_seconds") is not None else None),
                        end_seconds=(float(payload["end_seconds"]) if payload.get("end_seconds") is not None else None),
                    )
                if example.example_id in seen_ids:
                    raise ValueError(f"duplicate example ID: {example.example_id}")
                seen_ids.add(example.example_id)
                examples.append(example)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid manifest record at line {line_number}: {error}"
                ) from error
    return examples


class VideoQAExperiment:
    """Run one baseline/ablation mode while reusing loaded models."""

    def __init__(
        self,
        *,
        mode: str,
        decoder: Any,
        molmo_runner: Any,
        encoder: Any | None = None,
        embedding_cache: SQLiteEmbeddingCache | None = None,
        num_selected: int = 32,
        num_candidates: int = 128,
        official_max_frames: int = 384,
        max_new_tokens: int = 128,
        measure_ttft: bool = False,
    ) -> None:
        if num_selected <= 0 or num_candidates <= 0 or official_max_frames <= 0:
            raise ValueError("frame budgets must be positive")
        if num_selected > num_candidates:
            raise ValueError("num_selected cannot exceed num_candidates")
        if mode in SELECTOR_MODES and encoder is None:
            raise ValueError(f"mode {mode} requires a dual encoder")
        self.mode = mode
        self.decoder = decoder
        self.molmo_runner = molmo_runner
        self.encoder = encoder
        self.embedding_cache = embedding_cache
        self.num_selected = num_selected
        self.num_candidates = num_candidates
        self.official_max_frames = official_max_frames
        self.max_new_tokens = max_new_tokens
        self.measure_ttft = measure_ttft

    def run(self, example: VideoQAExample) -> ExperimentOutcome:
        total_start = perf_counter()
        query = build_query_text(example.question, example.options or None)
        selector_seconds = 0.0
        text_encoding_seconds = 0.0
        image_encoding_seconds = 0.0
        scoring_selection_seconds = 0.0
        cache_hits = 0
        cache_misses = 0

        budget = (
            self.official_max_frames if self.mode == "official_original" else
            self.num_selected if self.mode == "uniform" else self.num_candidates
        )
        bounds = {}
        if example.start_seconds is not None or example.end_seconds is not None:
            bounds = {"start_seconds": example.start_seconds, "end_seconds": example.end_seconds}
        decode_start = perf_counter()
        decoded = self.decoder.decode(example.video_path, budget, **bounds)
        decode_seconds = perf_counter() - decode_start
        if self.mode == "official_original":
            selected_frames = tuple(decoded.frames)
            selected_indices = decoded.frame_indices
            candidate_count = len(decoded.frame_indices)
        elif self.mode == "uniform":
            selected_frames = tuple(decoded.frames)
            selected_indices = decoded.frame_indices
            candidate_count = len(decoded.frame_indices)
        elif self.mode in SELECTOR_MODES:
            config = selector_config_for_mode(
                self.mode,
                num_selected=self.num_selected,
                num_candidates=self.num_candidates,
            )
            namespace = None
            if self.embedding_cache is not None:
                namespace = cache_namespace(
                    encoder_fingerprint=self.encoder.fingerprint,
                    video_identity=video_file_identity(example.video_path),
                )
            pipeline = QueryAwareSelectionPipeline(
                self.encoder,
                QueryAwareFrameSelector(config),
                embedding_cache=self.embedding_cache,
            )
            selection = pipeline.select(
                query,
                decoded.frames,
                timestamps=decoded.timestamps,
                frame_indices=decoded.frame_indices,
                cache_namespace=namespace,
                time_range=decoded.time_range,
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
        else:
            raise ValueError(f"unknown experiment mode: {self.mode}")
        time_by_index = dict(zip(decoded.frame_indices, decoded.timestamps))
        selected_times = tuple(time_by_index[index] for index in selected_indices)
        time_span, bins_covered = temporal_coverage(selected_times, decoded.time_range)
        bins_available = len(set(temporal_bin_ids(decoded.timestamps, 4, decoded.time_range)))

        generated = self.molmo_runner.generate(
            query,
            selected_frames,
            decoded.metadata_for(selected_indices),
            max_new_tokens=self.max_new_tokens,
            measure_ttft=self.measure_ttft,
        )
        labels = [chr(ord("A") + index) for index in range(len(example.options))]
        predicted_label = (
            parse_choice_label(generated.text, labels) if labels else None
        )
        correct = None
        if example.gold_label is not None:
            correct = predicted_label == example.gold_label

        record = ExperimentRecord(
            example_id=example.example_id,
            task=example.task,
            mode=self.mode,
            prediction=generated.text,
            predicted_label=predicted_label,
            gold_label=example.gold_label,
            correct=correct,
            candidate_frames=candidate_count,
            selected_frames=len(selected_indices),
            estimated_visual_tokens=estimate_visual_tokens(len(selected_indices)),
            actual_visual_tokens=generated.visual_tokens,
            input_tokens=generated.input_tokens,
            decode_seconds=decode_seconds,
            selector_seconds=selector_seconds,
            text_encoding_seconds=text_encoding_seconds,
            image_encoding_seconds=image_encoding_seconds,
            scoring_selection_seconds=scoring_selection_seconds,
            processor_seconds=generated.processor_seconds,
            ttft_seconds=generated.ttft_seconds,
            generation_seconds=generated.generation_seconds,
            end_to_end_seconds=perf_counter() - total_start,
            peak_vram_gb=generated.peak_vram_gb,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            output_tokens=generated.output_tokens,
            selected_time_span_fraction=time_span,
            temporal_bins_covered=bins_covered,
            temporal_bins_available=bins_available,
        )
        return ExperimentOutcome(
            record=record,
            query=query,
            selected_indices=tuple(selected_indices),
        )

    def run_safely(self, example: VideoQAExample) -> ExperimentOutcome:
        """Keep failed QA attempts in the accuracy denominator and error log."""
        start = perf_counter()
        try:
            return self.run(example)
        except Exception as error:
            return ExperimentOutcome(
                record=ExperimentRecord(
                    example_id=example.example_id, task=example.task, mode=self.mode,
                    prediction="", predicted_label=None, gold_label=example.gold_label,
                    correct=False if example.gold_label is not None else None,
                    candidate_frames=0, selected_frames=0, estimated_visual_tokens=0,
                    status="error", error_type=type(error).__name__, error_message=str(error),
                    end_to_end_seconds=perf_counter() - start,
                ),
                query=build_query_text(example.question, example.options or None),
                selected_indices=(),
            )
