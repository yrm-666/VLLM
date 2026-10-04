"""Configuration for query-aware frame selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    """Hyperparameters for the training-free greedy selector.

    The weights are normalized internally, so only their relative magnitudes
    matter. ``num_candidates`` is retained here so a complete experiment can
    be represented by one serializable configuration object.
    """

    num_selected: int = 32
    num_candidates: int = 128
    relevance_weight: float = 0.60
    diversity_weight: float = 0.25
    coverage_weight: float = 0.15
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_selected <= 0:
            raise ValueError("num_selected must be positive")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.num_selected > self.num_candidates:
            raise ValueError("num_selected cannot exceed num_candidates")

        weights = (
            self.relevance_weight,
            self.diversity_weight,
            self.coverage_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("selector weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one selector weight must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @property
    def normalized_weights(self) -> tuple[float, float, float]:
        total = (
            self.relevance_weight
            + self.diversity_weight
            + self.coverage_weight
        )
        return (
            self.relevance_weight / total,
            self.diversity_weight / total,
            self.coverage_weight / total,
        )

