"""Candidate-frame sampling utilities."""

from __future__ import annotations


def uniform_candidate_indices(num_frames: int, max_candidates: int) -> list[int]:
    """Return up to ``max_candidates`` uniformly spaced frame indices.

    Both endpoints are retained whenever at least two candidates are requested.
    Short videos are returned without duplicated frames.
    """

    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if num_frames == 0:
        return []
    if num_frames <= max_candidates:
        return list(range(num_frames))
    if max_candidates == 1:
        return [num_frames // 2]

    last = num_frames - 1
    denominator = max_candidates - 1
    return [round(position * last / denominator) for position in range(max_candidates)]


def official_style_candidate_indices(
    num_frames: int,
    fps: float,
    max_candidates: int,
    *,
    max_fps: float = 2.0,
) -> list[int]:
    """Mirror Molmo2's ``uniform_last_frame`` candidate policy.

    For a video short enough to stay under the frame budget, sample at up to
    ``max_fps`` and include the final frame. For a longer video, distribute the
    full candidate budget uniformly over the complete timeline.
    """

    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if max_fps <= 0:
        raise ValueError("max_fps must be positive")
    if num_frames <= 2:
        return list(range(num_frames))

    duration = num_frames / fps
    max_duration = (max_candidates - 1) / max_fps
    if duration > max_duration:
        return uniform_candidate_indices(num_frames, max_candidates)

    step = fps / max_fps
    positions: list[int] = []
    cursor = 0.0
    last = num_frames - 1
    while cursor < last:
        index = round(cursor)
        if not positions or index != positions[-1]:
            positions.append(index)
        cursor += step
    if not positions or positions[-1] != last:
        positions.append(last)
    if len(positions) > max_candidates:
        return uniform_candidate_indices(num_frames, max_candidates)
    return positions

