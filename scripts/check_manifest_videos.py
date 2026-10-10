"""Decode small candidate batches and check annotation bounds without models."""
import argparse
import json
from pathlib import Path
from molmo2_frame_selector.experiment import load_jsonl_manifest
from molmo2_frame_selector.video import DecordCandidateDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    examples = load_jsonl_manifest(args.manifest)
    if not examples:
        raise ValueError("manifest contains no examples")
    decoder = DecordCandidateDecoder(max_fps=2)
    raw_records = {str(row["id"]): row for line in Path(args.manifest).read_text(
        encoding="utf-8").splitlines() if line.strip() for row in [json.loads(line)]}
    failed = 0
    for example in examples:
        try:
            decoded = decoder.decode(example.video_path, 8, start_seconds=example.start_seconds,
                                     end_seconds=example.end_seconds)
            if example.end_seconds is not None and example.end_seconds > decoded.duration + 1e-3:
                raise ValueError("annotation end exceeds video duration; verify dataset before running")
            raw = raw_records.get(example.example_id, {})
            expected = raw.get("expected_duration_seconds")
            if raw.get("asset_layout") == "pretrimmed_segment":
                if example.start_seconds is not None or example.end_seconds is not None:
                    raise ValueError("pretrimmed clip must not be cropped again")
                if expected is None or abs(decoded.duration - float(expected)) > max(0.25, 2 / decoded.fps):
                    raise ValueError("segment duration does not match source interval")
            print(json.dumps({"id": example.example_id, "status": "ok",
                "requested_bounds": [example.start_seconds, example.end_seconds],
                "effective_bounds": decoded.time_range, "video_duration": decoded.duration,
                "original_indices": decoded.frame_indices, "timestamps": decoded.timestamps,
                "asset_layout": raw.get("asset_layout", "full_video"),
                "source_bounds": [raw.get("source_start_seconds"), raw.get("source_end_seconds")],
                "expected_duration_seconds": expected}), flush=True)
        except Exception as error:
            failed += 1
            print(json.dumps({"id": example.example_id, "status": "error", "error": str(error)}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
