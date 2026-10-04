"""Metrics and JSONL logging helpers for accuracy-efficiency experiments."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


def estimate_visual_tokens(
    frame_count: int,
    *,
    image_size: int = 378,
    patch_size: int = 14,
    pooling_size: int = 3,
) -> int:
    """Estimate Molmo2 pooled visual tokens, excluding wrapper tokens."""

    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if min(image_size, patch_size, pooling_size) <= 0:
        raise ValueError("image, patch, and pooling sizes must be positive")
    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")
    patch_grid = image_size // patch_size
    if patch_grid % pooling_size != 0:
        raise ValueError("patch grid must be divisible by pooling_size")
    tokens_per_frame = (patch_grid // pooling_size) ** 2
    return frame_count * tokens_per_frame


_ANSWER_PATTERNS = (
    re.compile(r"(?i)\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\)?\b"),
    re.compile(r"^\s*\(?([A-Z])\)?(?:[.、:\s]|$)", re.IGNORECASE),
    re.compile(r"\(([A-Z])\)", re.IGNORECASE),
)


def parse_choice_label(prediction: str, labels: Sequence[str]) -> str | None:
    allowed = {str(label).upper() for label in labels}
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(prediction)
        if match and match.group(1).upper() in allowed:
            return match.group(1).upper()
    return None


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    example_id: str
    task: str
    mode: str
    prediction: str
    predicted_label: str | None
    gold_label: str | None
    correct: bool | None
    candidate_frames: int
    selected_frames: int
    estimated_visual_tokens: int
    actual_visual_tokens: int | None = None
    input_tokens: int | None = None
    decode_seconds: float | None = None
    selector_seconds: float | None = None
    text_encoding_seconds: float | None = None
    image_encoding_seconds: float | None = None
    scoring_selection_seconds: float | None = None
    processor_seconds: float | None = None
    ttft_seconds: float | None = None
    prefill_seconds: float | None = None
    generation_seconds: float | None = None
    end_to_end_seconds: float | None = None
    peak_vram_gb: float | None = None
    cache_hits: int = 0
    cache_misses: int = 0


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ExperimentRecord | Mapping[str, Any]) -> None:
        if is_dataclass(record):
            payload = asdict(record)
        else:
            payload = dict(record)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
