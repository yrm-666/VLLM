"""Optional Decord-backed candidate decoding with original-frame metadata."""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from math import isfinite
from pathlib import Path
from typing import Any

from .sampling import official_style_candidate_indices


@dataclass(frozen=True, slots=True)
class DecodedCandidates:
    frames: Any
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    fps: float
    duration: float
    total_num_frames: int
    width: int
    height: int
    timeline_start: float = 0.0
    timeline_end: float | None = None

    @property
    def time_range(self) -> tuple[float, float]:
        return self.timeline_start, self.duration if self.timeline_end is None else self.timeline_end

    def metadata_for(self, selected_indices: list[int] | tuple[int, ...]) -> dict[str, Any]:
        """Build metadata consumed by Molmo2 when sampling is disabled."""

        selected = [int(index) for index in selected_indices]
        if any(index < 0 or index >= self.total_num_frames for index in selected):
            raise ValueError("selected frame index is outside the source video")
        return {
            "total_num_frames": self.total_num_frames,
            "fps": self.fps,
            "duration": self.duration,
            "video_backend": "query_aware_decord",
            "height": self.height,
            "width": self.width,
            "frames_indices": selected,
        }


class DecordCandidateDecoder:
    """Decode only candidate frames on CPU instead of loading a full video."""

    def __init__(self, *, max_fps: float = 2.0) -> None:
        if not isfinite(max_fps) or max_fps <= 0:
            raise ValueError("max_fps must be positive")
        self.max_fps = max_fps

    def decode(
        self, video_path: str | Path, max_candidates: int, *,
        start_seconds: float | None = None, end_seconds: float | None = None,
    ) -> DecodedCandidates:
        try:
            import decord
        except ImportError as error:
            raise RuntimeError(
                "Video decoding requires decord2 (import name: decord). See "
                "docs/GPU_RUNBOOK.md."
            ) from error

        path = Path(video_path).expanduser().resolve(strict=True)
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
        total_num_frames = len(reader)
        fps = float(reader.get_avg_fps())
        if total_num_frames == 0 or not isfinite(fps) or fps <= 0:
            raise ValueError(f"video has no valid frames/fps: {path}")
        try:
            frame_intervals = reader.get_frame_timestamp(list(range(total_num_frames)))
            duration = float(frame_intervals[-1][1] - frame_intervals[0][0])
            offset = float(frame_intervals[0][0])
            all_times = tuple(float(interval[0] - offset) for interval in frame_intervals)
        except (AttributeError, IndexError, TypeError):
            duration = total_num_frames / fps
            all_times = tuple(index / fps for index in range(total_num_frames))

        first, stop, time_range = bounded_frame_range(
            all_times, duration, start_seconds, end_seconds,
        )
        indices = [
            first + index for index in official_style_candidate_indices(
                stop - first, fps, max_candidates, max_fps=self.max_fps,
            )
        ]
        frames = reader.get_batch(indices).asnumpy()
        height, width = int(frames.shape[1]), int(frames.shape[2])
        timestamps = tuple(all_times[index] for index in indices)

        return DecodedCandidates(
            frames=frames,
            frame_indices=tuple(indices),
            timestamps=timestamps,
            fps=fps,
            duration=duration,
            total_num_frames=total_num_frames,
            width=width,
            height=height,
            timeline_start=time_range[0],
            timeline_end=time_range[1],
        )


def bounded_frame_range(
    frame_times: tuple[float, ...], duration: float,
    start_seconds: float | None = None, end_seconds: float | None = None,
) -> tuple[int, int, tuple[float, float]]:
    """Use frame-start timestamps inside [start, end), retaining original IDs.

    Bounds are elapsed video seconds. An end past the video duration is clipped;
    an empty/out-of-video interval is rejected instead of evaluating full video.
    """
    start = 0.0 if start_seconds is None else float(start_seconds)
    end = duration if end_seconds is None else float(end_seconds)
    if not all(isfinite(value) for value in (start, end, duration)):
        raise ValueError("video duration and temporal bounds must be finite")
    end = min(end, duration)
    if start < 0 or end <= start or not frame_times:
        raise ValueError("temporal interval must be nonempty and inside the video")
    first = bisect_left(frame_times, start)
    stop = bisect_left(frame_times, end)
    if first >= stop:
        raise ValueError("temporal interval contains no frame-start timestamps")
    return first, stop, (start, end)
