from __future__ import annotations

import argparse
import json
from pathlib import Path

from molmo2_frame_selector.reporting import compare_jsonl_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--baseline-mode", default="uniform")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = compare_jsonl_files(args.results, baseline_mode=args.baseline_mode)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
