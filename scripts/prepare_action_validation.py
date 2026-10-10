"""Fixed action-sequence pilot with mandatory temporal bounds and provenance."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from molmo2_frame_selector.mvbench import convert_mvbench_annotations


def resolve_segment(row, paths):
    """Match video identity AND numeric bounds, never choose a nearby clip."""
    matches = []
    stem = PurePosixPath(row["video"]).stem
    for path in paths:
        if not path.startswith("star/Charades_segment/") or not path.endswith(".mp4"):
            continue
        parts = PurePosixPath(path).stem.rsplit("_", 2)
        if len(parts) != 3 or parts[0] != stem:
            continue
        try:
            start, end = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if abs(start - float(row["start"])) <= 1e-6 and abs(end - float(row["end"])) <= 1e-6:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matching segment for {row['video']} "
                         f"[{row['start']},{row['end']}], found {len(matches)}")
    return matches[0]


def segment_record(record, remote_path, video_root):
    output = dict(record)
    start, end = output.pop("start_seconds"), output.pop("end_seconds")
    output.update(video=str((video_root / remote_path).resolve()),
                  source_video=Path(record["video"]).name,
                  source_start_seconds=start, source_end_seconds=end,
                  expected_duration_seconds=end - start,
                  asset_layout="pretrimmed_segment", dataset_video_path=remote_path)
    return output


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
    prefix = "star/Charades_segment"
    video_root = root / "videos"
    paths = [entry.path for entry in api.list_repo_tree(
        repo, path_in_repo=prefix, repo_type="dataset", revision=video_revision, recursive=True)
        if entry.path.endswith(".mp4")]
    plan = {}
    # Validate labels and bounds before downloading any videos.
    for index in selected:
        record = convert_mvbench_annotations(annotation, video_root, task="action_sequence",
                                   offset=index, limit=1, require_videos=False)[0]
        remote_path = resolve_segment(rows[index], paths)
        plan[index] = segment_record(record, remote_path, video_root)
    records = []
    for progress, index in enumerate(selected, 1):
        hf_hub_download(repo, plan[index]["dataset_video_path"], repo_type="dataset",
                        revision=video_revision, local_dir=str(video_root))
        if not Path(plan[index]["video"]).is_file():
            raise FileNotFoundError(plan[index]["video"])
        records.append(plan[index])
        print(json.dumps({"downloaded": f"{progress}/{len(selected)}", "annotation_index": index}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    source.write_text(json.dumps({
        "dataset": repo, "task": "action_sequence", "video_prefix": prefix,
        "annotation_revision": annotation_revision, "video_revision": video_revision,
        "annotation_indices": selected, "selection_policy": "source order; distinct videos; no label filtering",
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "temporal_policy": "use full pretrimmed clip once; local timestamps; source bounds preserved; native equivalence not asserted",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
