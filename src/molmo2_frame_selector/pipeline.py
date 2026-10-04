"""End-to-end query-aware selection over already decoded candidate frames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .cache import SQLiteEmbeddingCache
from .encoders import DualEncoder, cosine_similarities
from .selector import QueryAwareFrameSelector, SelectionResult


@dataclass(frozen=True, slots=True)
class PipelineTimings:
    text_encoding_seconds: float
    image_encoding_seconds: float
    scoring_and_selection_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class PipelineResult:
    selection: SelectionResult
    selected_frames: tuple[Any, ...]
    relevance_scores: tuple[float, ...]
    frame_embeddings: tuple[tuple[float, ...], ...]
    cache_hits: int
    cache_misses: int
    timings: PipelineTimings


class QueryAwareSelectionPipeline:
    def __init__(
        self,
        encoder: DualEncoder,
        selector: QueryAwareFrameSelector,
        *,
        embedding_cache: SQLiteEmbeddingCache | None = None,
    ) -> None:
        self.encoder = encoder
        self.selector = selector
        self.embedding_cache = embedding_cache

    def _encode_images(
        self,
        frames: Sequence[Any],
        frame_indices: Sequence[int],
        namespace: str | None,
    ) -> tuple[list[list[float]], int, int]:
        if self.embedding_cache is None or namespace is None:
            embeddings = self.encoder.encode_images(frames)
            return embeddings, 0, len(frames)

        keys = [str(index) for index in frame_indices]
        cached = self.embedding_cache.get_many(namespace, keys)
        embeddings: list[list[float] | None] = [cached.get(key) for key in keys]
        missing_positions = [
            position for position, embedding in enumerate(embeddings) if embedding is None
        ]
        if missing_positions:
            missing_frames = [frames[position] for position in missing_positions]
            computed = self.encoder.encode_images(missing_frames)
            if len(computed) != len(missing_positions):
                raise ValueError("encoder returned an unexpected number of embeddings")
            cache_items = []
            for position, embedding in zip(missing_positions, computed):
                embeddings[position] = embedding
                cache_items.append((keys[position], embedding))
            self.embedding_cache.put_many(namespace, cache_items)

        complete = [embedding for embedding in embeddings if embedding is not None]
        return complete, len(cached), len(missing_positions)

    def select(
        self,
        query: str,
        frames: Sequence[Any],
        *,
        timestamps: Sequence[float],
        frame_indices: Sequence[int],
        cache_namespace: str | None = None,
    ) -> PipelineResult:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if not (len(frames) == len(timestamps) == len(frame_indices)):
            raise ValueError("frames, timestamps, and frame_indices must align")

        total_start = perf_counter()
        text_start = perf_counter()
        text_embeddings = self.encoder.encode_texts([query])
        text_seconds = perf_counter() - text_start
        if len(text_embeddings) != 1:
            raise ValueError("encoder must return one embedding for one query")

        image_start = perf_counter()
        image_embeddings, cache_hits, cache_misses = self._encode_images(
            frames,
            frame_indices,
            cache_namespace,
        )
        image_seconds = perf_counter() - image_start
        if len(image_embeddings) != len(frames):
            raise ValueError("encoder returned an unexpected number of frame embeddings")

        selection_start = perf_counter()
        relevance = cosine_similarities(text_embeddings[0], image_embeddings)
        selection = self.selector.select(
            relevance,
            image_embeddings,
            timestamps=timestamps,
            candidate_indices=frame_indices,
        )
        selected_frames = tuple(frames[position] for position in selection.selected_positions)
        selection_seconds = perf_counter() - selection_start

        return PipelineResult(
            selection=selection,
            selected_frames=selected_frames,
            relevance_scores=tuple(relevance),
            frame_embeddings=tuple(tuple(row) for row in image_embeddings),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            timings=PipelineTimings(
                text_encoding_seconds=text_seconds,
                image_encoding_seconds=image_seconds,
                scoring_and_selection_seconds=selection_seconds,
                total_seconds=perf_counter() - total_start,
            ),
        )

