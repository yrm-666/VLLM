from __future__ import annotations

import argparse
import json
from pathlib import Path

from molmo2_frame_selector.mvbench import convert_mvbench_annotations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert one official MVBench task JSON into a JSONL manifest."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-missing-videos", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    records = convert_mvbench_annotations(
        args.annotations,
        args.video_root,
        task=args.task,
        limit=args.limit,
        require_videos=not args.allow_missing_videos,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

