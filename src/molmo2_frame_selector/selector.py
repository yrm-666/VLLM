"""Framework-independent reference implementation of the frame selector."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Sequence

from .config import SelectorConfig


@dataclass(frozen=True, slots=True)
class SelectionStep:
    """Diagnostic record for one greedy selection step."""

    rank: int
    candidate_position: int
    frame_index: int
    timestamp: float
    total_score: float
    relevance_score: float
    diversity_score: float
    coverage_score: float


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Selected frames plus diagnostics needed for experiment logging."""

    candidate_count: int
    requested_count: int
    selection_steps: tuple[SelectionStep, ...]
    selected_positions: tuple[int, ...]
    selected_indices: tuple[int, ...]
    selected_timestamps: tuple[float, ...]


def _as_finite_floats(values: Iterable[float], name: str) -> list[float]:
    converted = [float(value) for value in values]
    if any(not isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _min_max_scale(values: Sequence[float], epsilon: float) -> list[float]:
    low = min(values)
    high = max(values)
    width = high - low
    if width <= epsilon:
        return [0.5] * len(values)
    return [(value - low) / width for value in values]


def _normalize_rows(
    rows: Sequence[Sequence[float]], epsilon: float
) -> list[list[float]]:
    if not rows:
        return []

    dimension = len(rows[0])
    if dimension == 0:
        raise ValueError("frame_embeddings cannot have zero-dimensional rows")

    normalized: list[list[float]] = []
    for row in rows:
        if len(row) != dimension:
            raise ValueError("all frame_embeddings rows must have equal length")
        vector = _as_finite_floats(row, "frame_embeddings")
        norm = sqrt(sum(value * value for value in vector))
        if norm <= epsilon:
            normalized.append([0.0] * dimension)
        else:
            normalized.append([value / norm for value in vector])
    return normalized


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    similarity = sum(a * b for a, b in zip(left, right))
    similarity = max(-1.0, min(1.0, similarity))
    return (1.0 - similarity) / 2.0


class QueryAwareFrameSelector:
    """Greedily select relevant, diverse, temporally distributed frames.

    The expensive dual-encoder step is deliberately outside this class. Pass
    its query/frame similarity values as ``relevance_scores`` and its normalized
    or unnormalized frame embeddings as ``frame_embeddings``.
    """

    def __init__(self, config: SelectorConfig | None = None) -> None:
        self.config = config or SelectorConfig()

    def select(
        self,
        relevance_scores: Sequence[float],
        frame_embeddings: Sequence[Sequence[float]],
        *,
        timestamps: Sequence[float] | None = None,
        candidate_indices: Sequence[int] | None = None,
    ) -> SelectionResult:
        relevance_raw = _as_finite_floats(relevance_scores, "relevance_scores")
        count = len(relevance_raw)
        if count == 0:
            return SelectionResult(
                candidate_count=0,
                requested_count=self.config.num_selected,
                selection_steps=(),
                selected_positions=(),
                selected_indices=(),
                selected_timestamps=(),
            )
        if len(frame_embeddings) != count:
            raise ValueError(
                "frame_embeddings and relevance_scores must have the same length"
            )

        if timestamps is None:
            time_values = [float(position) for position in range(count)]
        else:
            if len(timestamps) != count:
                raise ValueError(
                    "timestamps and relevance_scores must have the same length"
                )
            time_values = _as_finite_floats(timestamps, "timestamps")

        if candidate_indices is None:
            frame_indices = list(range(count))
        else:
            if len(candidate_indices) != count:
                raise ValueError(
                    "candidate_indices and relevance_scores must have the same length"
                )
            frame_indices = [int(index) for index in candidate_indices]
            if any(index < 0 for index in frame_indices):
                raise ValueError("candidate_indices must be non-negative")
            if len(set(frame_indices)) != count:
                raise ValueError("candidate_indices must be unique")

        embeddings = _normalize_rows(frame_embeddings, self.config.epsilon)
        relevance = _min_max_scale(relevance_raw, self.config.epsilon)
        alpha, beta, gamma = self.config.normalized_weights

        start_time = min(time_values)
        duration = max(time_values) - start_time
        selected: list[int] = []
        steps: list[SelectionStep] = []
        target_count = min(self.config.num_selected, count)

        for rank in range(target_count):
            best_position = -1
            best_key: tuple[float, float, int] | None = None
            best_components = (0.0, 0.0, 0.0)

            for position in range(count):
                if position in selected:
                    continue

                relevance_component = relevance[position]
                if selected:
                    diversity_component = min(
                        _cosine_distance(embeddings[position], embeddings[chosen])
                        for chosen in selected
                    )
                    if duration <= self.config.epsilon:
                        coverage_component = 0.0
                    else:
                        coverage_component = min(
                            abs(time_values[position] - time_values[chosen]) / duration
                            for chosen in selected
                        )
                else:
                    diversity_component = 0.0
                    coverage_component = 0.0

                total = (
                    alpha * relevance_component
                    + beta * diversity_component
                    + gamma * coverage_component
                )
                # Equal scores resolve to the earlier frame, making runs stable.
                key = (total, -time_values[position], -frame_indices[position])
                if best_key is None or key > best_key:
                    best_key = key
                    best_position = position
                    best_components = (
                        relevance_component,
                        diversity_component,
                        coverage_component,
                    )

            selected.append(best_position)
            assert best_key is not None
            steps.append(
                SelectionStep(
                    rank=rank,
                    candidate_position=best_position,
                    frame_index=frame_indices[best_position],
                    timestamp=time_values[best_position],
                    total_score=best_key[0],
                    relevance_score=best_components[0],
                    diversity_score=best_components[1],
                    coverage_score=best_components[2],
                )
            )

        chronological = sorted(
            selected,
            key=lambda position: (time_values[position], frame_indices[position]),
        )
        return SelectionResult(
            candidate_count=count,
            requested_count=self.config.num_selected,
            selection_steps=tuple(steps),
            selected_positions=tuple(chronological),
            selected_indices=tuple(frame_indices[position] for position in chronological),
            selected_timestamps=tuple(time_values[position] for position in chronological),
        )

