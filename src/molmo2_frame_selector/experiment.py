"""Reusable experiment orchestration for single examples and manifests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .baselines import SELECTOR_MODES, selector_config_for_mode
from .cache import SQLiteEmbeddingCache, cache_namespace, video_file_identity
from .metrics import ExperimentRecord, estimate_visual_tokens, parse_choice_label
from .pipeline import QueryAwareSelectionPipeline
from .query import build_query_text
from .selector import QueryAwareFrameSelector


@dataclass(frozen=True, slots=True)
class VideoQAExample:
    example_id: str
    video_path: Path
    question: str
    options: tuple[str, ...] = ()
    gold_label: str | None = None
    task: str = "unknown"


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
                examples.append(
                    VideoQAExample(
                        example_id=str(payload.get("id", line_number)),
                        video_path=video_path,
                        question=str(payload["question"]),
                        options=options,
                        gold_label=_gold_label(payload.get("answer"), options),
                        task=str(payload.get("task", "unknown")),
                    )
                )
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

        decode_start = perf_counter()
        if self.mode == "official_original":
            decoded = self.decoder.decode(example.video_path, self.official_max_frames)
            selected_frames = tuple(decoded.frames)
            selected_indices = decoded.frame_indices
            candidate_count = len(decoded.frame_indices)
        elif self.mode == "uniform":
            decoded = self.decoder.decode(example.video_path, self.num_selected)
            selected_frames = tuple(decoded.frames)
            selected_indices = decoded.frame_indices
            candidate_count = len(decoded.frame_indices)
        elif self.mode in SELECTOR_MODES:
            decoded = self.decoder.decode(example.video_path, self.num_candidates)
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
        decode_and_selection_seconds = perf_counter() - decode_start
        decode_seconds = max(0.0, decode_and_selection_seconds - selector_seconds)

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
        )
        return ExperimentOutcome(
            record=record,
            query=query,
            selected_indices=tuple(selected_indices),
        )
