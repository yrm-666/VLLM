"""Exercise batch CLI behavior without loading PyTorch or downloading weights."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from molmo2_frame_selector.hf_runner import GenerationResult
from molmo2_frame_selector.video import DecodedCandidates, DecordCandidateDecoder


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_manifest.py"
    spec = importlib.util.spec_from_file_location("batch_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDecoder:
    def __init__(self, **kwargs):
        pass

    def decode(self, path, budget, **kwargs):
        if Path(path).name == "bad.mp4":
            raise RuntimeError("broken video")
        return DecodedCandidates([[1, 0], [0, 1]], (0, 9), (0, 0.9), 10, 1, 10, 8, 8)


class FakeRunner:
    resolved_revision = "fake-model-commit"

    def __init__(self, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        return GenerationResult("A", 12, 162, 0.01, None, 0.02, None, 1)

    def release_unused_memory(self):
        pass


class FakeEncoder:
    resolved_revision = "fake-selector-commit"
    fingerprint = "fake-selector"

    def __init__(self, **kwargs):
        pass

    def encode_texts(self, texts):
        return [[1, 0] for _ in texts]

    def encode_images(self, images):
        return list(images)


class ManifestRunnerTest(unittest.TestCase):
    def invoke(self, module, arguments):
        with patch("sys.argv", ["run_manifest.py", *map(str, arguments)]), \
             patch.object(module, "Molmo2HFRunner", FakeRunner), \
             patch.object(module, "Siglip2Encoder", FakeEncoder), \
             patch.object(module, "DecordCandidateDecoder", FakeDecoder), \
             contextlib.redirect_stdout(io.StringIO()):
            return module.main()

    def manifest(self, root, names):
        rows = []
        for index, name in enumerate(names):
            (root / name).write_bytes(b"video")
            rows.append({"id": str(index), "video": name, "question": "Q?",
                         "options": ["yes", "no"], "answer": "A"})
        path = root / "manifest.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_warm_cache_and_resume_configuration_guard(self):
        module = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest(root, ["ok.mp4"])
            output = root / "out.jsonl"
            args = ["--manifest", manifest, "--output", output, "--mode", "query_aware_v2",
                    "--num-selected", "2", "--num-candidates", "4", "--cache-state", "warm",
                    "--cache", root / "features.sqlite3"]
            self.assertEqual(self.invoke(module, args), 0)
            row = json.loads(output.read_text())
            self.assertEqual(row["cache_hits"], 2)
            self.assertEqual(row["cache_misses"], 0)
            self.assertEqual(row["resolved_model_revision"], "fake-model-commit")
            self.assertTrue(output.with_suffix(".run.json").exists())
            self.assertEqual(self.invoke(module, [*args, "--resume"]), 0)
            self.assertEqual(len(output.read_text().splitlines()), 1)
            with self.assertRaisesRegex(ValueError, "num_selected_requested"):
                self.invoke(module, [*args, "--resume", "--num-selected", "1"])

    def test_cold_cache_does_not_create_database(self):
        module = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest(root, ["ok.mp4"])
            output, cache = root / "out.jsonl", root / "cache.sqlite3"
            args = ["--manifest", manifest, "--output", output, "--mode", "query_aware_v2",
                    "--num-selected", "2", "--num-candidates", "4", "--cache-state", "cold",
                    "--warmup", "1", "--cache", cache]
            self.assertEqual(self.invoke(module, args), 0)
            row = json.loads(output.read_text())
            self.assertEqual(row["cache_hits"], 0)
            self.assertEqual(row["cache_misses"], 2)
            self.assertFalse(cache.exists())

    def test_failed_example_is_written_before_continuing(self):
        module = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest(root, ["bad.mp4", "ok.mp4"])
            output = root / "out.jsonl"
            args = ["--manifest", manifest, "--output", output, "--mode", "uniform",
                    "--continue-on-error"]
            self.assertEqual(self.invoke(module, args), 1)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["error", "ok"])
            self.assertFalse(rows[0]["correct"])
            self.assertTrue(rows[1]["correct"])
            self.assertEqual(self.invoke(module, [*args, "--resume"]), 1)

    def test_decoder_interval_keeps_full_video_metadata(self):
        import types

        class Frames:
            shape = (3, 8, 16, 3)

        class Reader:
            def __init__(self, *args, **kwargs):
                self.indices = None

            def __len__(self):
                return 10

            def get_avg_fps(self):
                return 1

            def get_frame_timestamp(self, indices):
                return [(float(index), float(index + 1)) for index in indices]

            def get_batch(self, indices):
                self.indices = indices
                return types.SimpleNamespace(asnumpy=lambda: Frames())

        reader = Reader()
        decord = types.SimpleNamespace(VideoReader=lambda *a, **k: reader, cpu=lambda x: x)
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            video.write_bytes(b"video")
            with patch.dict("sys.modules", {"decord": decord}):
                result = DecordCandidateDecoder(max_fps=2).decode(
                    video, 3, start_seconds=3, end_seconds=6)
        self.assertEqual(result.frame_indices, (3, 4, 5))
        self.assertEqual(result.timestamps, (3, 4, 5))
        self.assertEqual(result.time_range, (3, 6))
        metadata = result.metadata_for([3, 5])
        self.assertEqual(metadata["total_num_frames"], 10)
        self.assertEqual(metadata["duration"], 10)
        self.assertEqual(metadata["frames_indices"], [3, 5])


if __name__ == "__main__":
    unittest.main()
