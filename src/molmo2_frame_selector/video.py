"""Optional Decord-backed candidate decoding with original-frame metadata."""

from __future__ import annotations

from dataclasses import dataclass
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
        if max_fps <= 0:
            raise ValueError("max_fps must be positive")
        self.max_fps = max_fps

    def decode(self, video_path: str | Path, max_candidates: int) -> DecodedCandidates:
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
        indices = official_style_candidate_indices(
            total_num_frames,
            fps,
            max_candidates,
            max_fps=self.max_fps,
        )
        if not indices:
            raise ValueError(f"video contains no decodable frames: {path}")
        frames = reader.get_batch(indices).asnumpy()
        height, width = int(frames.shape[1]), int(frames.shape[2])

        try:
            frame_intervals = reader.get_frame_timestamp(list(range(total_num_frames)))
            duration = float(frame_intervals[-1][1] - frame_intervals[0][0])
            offset = float(frame_intervals[0][0])
            timestamps = tuple(float(frame_intervals[index][0] - offset) for index in indices)
        except (AttributeError, IndexError, TypeError):
            duration = total_num_frames / fps
            timestamps = tuple(index / fps for index in indices)

        return DecodedCandidates(
            frames=frames,
            frame_indices=tuple(indices),
            timestamps=timestamps,
            fps=fps,
            duration=duration,
            total_num_frames=total_num_frames,
            width=width,
            height=height,
        )

