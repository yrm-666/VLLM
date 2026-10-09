"""Download a fixed, video-disjoint scene pilot onto the chosen data disk."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from molmo2_frame_selector.mvbench import convert_mvbench_annotations


def choose_indices(annotations, development_records, offset, count):
    # Basenames conservatively exclude overlap even if the old manifest used
    # HF snapshots rather than local_dir. This task has one flat video folder.
    excluded = {Path(row["video"]).name for row in development_records}
    selected, seen = [], set()
    for index in range(offset, len(annotations)):
        name = Path(annotations[index]["video"]).name
        if name in excluded or name in seen:
            continue
        selected.append(index)
        seen.add(name)
        if len(selected) == count:
            return selected
    raise ValueError(f"only {len(selected)} distinct unused videos available; requested {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=20)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.offset < 0 or args.count <= 0:
        raise ValueError("offset must be non-negative and count positive")
    provenance_path = args.output.with_suffix(".source.json")
    if args.output.exists() or provenance_path.exists():
        raise FileExistsError("validation manifest/provenance already exists; choose a new output")
    development = [json.loads(line) for line in args.development_manifest.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not development:
        raise ValueError("development manifest must not be empty")
    from huggingface_hub import HfApi, hf_hub_download
    repo = "OpenGVLab/MVBench"
    api = HfApi()
    annotation_revision = api.dataset_info(repo, revision="main").sha
    video_revision = api.dataset_info(repo, revision="video").sha
    data_root = args.data_root.expanduser().resolve()
    annotation_path = Path(hf_hub_download(
        repo, "json/scene_transition.json", repo_type="dataset", revision=annotation_revision,
        local_dir=str(data_root / "annotations")))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    selected = choose_indices(annotations, development, args.offset, args.count)
    video_root = data_root / "videos" / "scene_qa" / "video"
    records = []
    for progress, index in enumerate(selected, start=1):
        name = annotations[index]["video"]
        hf_hub_download(repo, f"scene_qa/video/{name}", repo_type="dataset",
                        revision=video_revision, local_dir=str(data_root / "videos"))
        records.extend(convert_mvbench_annotations(
            annotation_path, video_root, task="scene_transition", offset=index, limit=1))
        print(json.dumps({"downloaded": f"{progress}/{len(selected)}", "annotation_index": index}),
              flush=True)
    dev_ids = {str(row["id"]) for row in development}
    if dev_ids.intersection(record["id"] for record in records):
        raise ValueError("selected IDs overlap development manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                           encoding="utf-8")
    provenance = {
        "dataset": repo, "task": "scene_transition", "annotation_revision": annotation_revision,
        "video_revision": video_revision, "annotation_indices": selected,
        "development_manifest_sha256": hashlib.sha256(args.development_manifest.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "selection_policy": "source order after offset; exclude development and repeated video basenames",
        "count": len(records),
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(f"wrote {len(records)} validation examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
