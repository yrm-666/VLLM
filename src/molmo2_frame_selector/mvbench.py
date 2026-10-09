"""Convert official MVBench annotation JSON into the project manifest format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from math import isfinite


def convert_mvbench_annotations(
    annotation_path: str | Path,
    video_root: str | Path,
    *,
    task: str,
    limit: int | None = None,
    require_videos: bool = True,
    offset: int = 0,
) -> list[dict[str, Any]]:
    annotation_path = Path(annotation_path).expanduser().resolve(strict=True)
    video_root = Path(video_root).expanduser().resolve(strict=require_videos)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("MVBench annotation JSON must contain a list")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    output = []
    stop = None if limit is None else offset + limit
    for index, item in enumerate(payload[offset:stop], start=offset):
        try:
            video_path = (video_root / item["video"]).resolve()
            if require_videos and not video_path.is_file():
                raise FileNotFoundError(video_path)
            options = [str(option) for option in item["candidates"]]
            answer = str(item["answer"])
            if answer not in options:
                raise ValueError("answer is not present in candidates")
            record = {
                    "id": f"{task}-{index:04d}",
                    "video": str(video_path),
                    "question": str(item["question"]),
                    "options": options,
                    "answer": chr(ord("A") + options.index(answer)),
                    "task": task,
                }
            if "start" in item or "end" in item:
                if item.get("start") is None or item.get("end") is None:
                    raise ValueError("interval annotations must provide both start and end")
                start, end = float(item["start"]), float(item["end"])
                if not isfinite(start) or not isfinite(end) or start < 0 or end <= start:
                    raise ValueError("invalid start/end interval")
                record.update(start_seconds=start, end_seconds=end)
            output.append(record)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            raise ValueError(f"invalid MVBench item {index}: {error}") from error
    return output
