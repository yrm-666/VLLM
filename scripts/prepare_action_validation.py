"""Fixed action-sequence pilot with mandatory temporal bounds and provenance."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from molmo2_frame_selector.mvbench import convert_mvbench_annotations


def choose_indices(rows, count, offset=0):
    if count <= 0 or offset < 0:
        raise ValueError("count must be positive and offset non-negative")
    selected, seen = [], set()
    for index in range(offset, len(rows)):
        row = rows[index]
        name = PurePosixPath(row["video"])
        if name.is_absolute() or ".." in name.parts or "\\" in row["video"]:
            raise ValueError("unsafe video path")
        if row.get("start") is None or row.get("end") is None:
            raise ValueError(f"action_sequence item {index} lacks start/end")
        if str(name) in seen:
            continue
        selected.append(index)
        seen.add(str(name))
        if len(selected) == count:
            return selected
    raise ValueError("not enough distinct videos for the requested pilot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.count <= 0 or args.offset < 0:
        raise ValueError("count must be positive and offset non-negative")
    source = args.output.with_suffix(".source.json")
    if args.output.exists() or source.exists():
        raise FileExistsError("choose a new manifest path; existing files are preserved")
    from huggingface_hub import HfApi, hf_hub_download
    repo = "OpenGVLab/MVBench"
    api = HfApi()
    annotation_revision = api.dataset_info(repo, revision="main").sha
    video_revision = api.dataset_info(repo, revision="video").sha
    root = args.data_root.expanduser().resolve()
    annotation = Path(hf_hub_download(repo, "json/action_sequence.json", repo_type="dataset",
        revision=annotation_revision, local_dir=str(root / "annotations")))
    rows = json.loads(annotation.read_text(encoding="utf-8"))
    selected = choose_indices(rows, args.count, args.offset)
    prefix = "star/Charades_v1_480"
    video_root = root / "videos" / prefix
    # Validate labels and bounds before downloading any videos.
    for index in selected:
        convert_mvbench_annotations(annotation, video_root, task="action_sequence",
                                   offset=index, limit=1, require_videos=False)
    records = []
    for progress, index in enumerate(selected, 1):
        hf_hub_download(repo, f"{prefix}/{rows[index]['video']}", repo_type="dataset",
                        revision=video_revision, local_dir=str(root / "videos"))
        records.extend(convert_mvbench_annotations(annotation, video_root,
                        task="action_sequence", offset=index, limit=1))
        print(json.dumps({"downloaded": f"{progress}/{len(selected)}", "annotation_index": index}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    source.write_text(json.dumps({
        "dataset": repo, "task": "action_sequence", "video_prefix": prefix,
        "annotation_revision": annotation_revision, "video_revision": video_revision,
        "annotation_indices": selected, "selection_policy": "source order; distinct videos; no label filtering",
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "temporal_policy": "frame-start timestamps in [start,end); native equivalence not asserted",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
