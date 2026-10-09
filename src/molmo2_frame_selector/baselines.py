"""Named experiment modes used by the planned ablation table."""

from __future__ import annotations

from .config import SelectorConfig


SELECTOR_MODES = (
    "relevance_only",
    "relevance_diversity",
    "query_aware",
    "query_aware_v2",
)
ALL_MODES = ("official_original", "uniform", *SELECTOR_MODES)


def selector_config_for_mode(
    mode: str,
    *,
    num_selected: int,
    num_candidates: int,
) -> SelectorConfig:
    if mode == "relevance_only":
        weights = (1.0, 0.0, 0.0)
    elif mode == "relevance_diversity":
        weights = (0.60, 0.25, 0.0)
    elif mode in {"query_aware", "query_aware_v2"}:
        weights = (0.60, 0.25, 0.15)
    else:
        raise ValueError(f"mode does not use a query-aware selector: {mode}")
    return SelectorConfig(
        num_selected=num_selected,
        num_candidates=num_candidates,
        relevance_weight=weights[0],
        diversity_weight=weights[1],
        coverage_weight=weights[2],
        temporal_bins=4 if mode == "query_aware_v2" else 0,
    )
