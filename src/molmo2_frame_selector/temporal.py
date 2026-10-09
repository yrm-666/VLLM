"""Shared time-bin definitions for selection and coverage diagnostics."""

from math import isfinite
from collections.abc import Sequence


def temporal_bin_ids(
    timestamps: Sequence[float],
    bins: int,
    time_range: tuple[float, float] | None = None,
) -> tuple[int, ...]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    times = [float(time) for time in timestamps]
    if any(not isfinite(time) for time in times):
        raise ValueError("timestamps must be finite")
    if not times:
        return ()
    start, end = time_range if time_range is not None else (min(times), max(times))
    if not isfinite(start) or not isfinite(end) or end < start:
        raise ValueError("time_range must be finite and ordered")
    if min(times) < start - 1e-7 or max(times) > end + 1e-7:
        raise ValueError("timestamps must be inside time_range")
    if end == start:
        return (0,) * len(times)
    return tuple(
        max(0, min(bins - 1, int((time - start) / (end - start) * bins)))
        for time in times
    )


def temporal_coverage(
    timestamps: Sequence[float],
    time_range: tuple[float, float],
    *,
    bins: int = 4,
) -> tuple[float, int]:
    ids = temporal_bin_ids(timestamps, bins, time_range)
    width = time_range[1] - time_range[0]
    span = (max(timestamps) - min(timestamps)) / width if len(timestamps) > 0 and width > 0 else 0.0
    return min(1.0, max(0.0, span)), len(set(ids))
