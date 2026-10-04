"""Convert official MVBench annotation JSON into the project manifest format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def convert_mvbench_annotations(
    annotation_path: str | Path,
    video_root: str | Path,
    *,
    task: str,
    limit: int | None = None,
    require_videos: bool = True,
) -> list[dict[str, Any]]:
    annotation_path = Path(annotation_path).expanduser().resolve(strict=True)
    video_root = Path(video_root).expanduser().resolve(strict=require_videos)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("MVBench annotation JSON must contain a list")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    output = []
    for index, item in enumerate(payload[:limit] if limit is not None else payload):
        try:
            video_path = (video_root / item["video"]).resolve()
            if require_videos and not video_path.is_file():
                raise FileNotFoundError(video_path)
            options = [str(option) for option in item["candidates"]]
            answer = str(item["answer"])
            if answer not in options:
                raise ValueError("answer is not present in candidates")
            output.append(
                {
                    "id": f"{task}-{index:04d}",
                    "video": str(video_path),
                    "question": str(item["question"]),
                    "options": options,
                    "answer": chr(ord("A") + options.index(answer)),
                    "task": task,
                }
            )
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            raise ValueError(f"invalid MVBench item {index}: {error}") from error
    return output

