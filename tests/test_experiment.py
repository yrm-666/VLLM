from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from molmo2_frame_selector.cache import SQLiteEmbeddingCache
from molmo2_frame_selector.experiment import (
    VideoQAExample,
    VideoQAExperiment,
    load_jsonl_manifest,
)
from molmo2_frame_selector.hf_runner import GenerationResult
from molmo2_frame_selector.video import DecodedCandidates


class FakeDecoder:
    def decode(self, video_path, max_candidates):
        frames = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
        return DecodedCandidates(
            frames=frames[:max_candidates],
            frame_indices=tuple([0, 10, 20, 30][:max_candidates]),
            timestamps=tuple([0.0, 1.0, 2.0, 3.0][:max_candidates]),
            fps=10.0,
            duration=4.0,
            total_num_frames=40,
            width=320,
            height=240,
        )


class FakeEncoder:
    fingerprint = "fake"

    def encode_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def encode_images(self, images):
        return [list(image) for image in images]


class FakeRunner:
    def generate(self, query, frames, metadata, **kwargs):
        return GenerationResult(
            text="Answer: B",
            input_tokens=200,
            visual_tokens=81 * len(frames),
            processor_seconds=0.1,
            ttft_seconds=0.2,
            generation_seconds=0.3,
            peak_vram_gb=10.0,
        )


class ExperimentTest(unittest.TestCase):
    def test_query_aware_experiment_records_correctness_and_efficiency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"placeholder")
            with SQLiteEmbeddingCache(Path(directory) / "cache.sqlite3") as cache:
                experiment = VideoQAExperiment(
                    mode="query_aware",
                    decoder=FakeDecoder(),
                    molmo_runner=FakeRunner(),
                    encoder=FakeEncoder(),
                    embedding_cache=cache,
                    num_selected=2,
                    num_candidates=4,
                )
                outcome = experiment.run(
                    VideoQAExample(
                        example_id="1",
                        video_path=video,
                        question="What is picked up?",
                        options=("Book", "Cup", "Phone", "Ball"),
                        gold_label="B",
                        task="demo",
                    )
                )
        self.assertTrue(outcome.record.correct)
        self.assertEqual(outcome.record.selected_frames, 2)
        self.assertEqual(outcome.record.estimated_visual_tokens, 162)
        self.assertEqual(outcome.record.ttft_seconds, 0.2)
        self.assertIn("(B) Cup", outcome.query)

    def test_uniform_mode_needs_no_encoder(self) -> None:
        experiment = VideoQAExperiment(
            mode="uniform",
            decoder=FakeDecoder(),
            molmo_runner=FakeRunner(),
            num_selected=2,
            num_candidates=4,
        )
        outcome = experiment.run(
            VideoQAExample(
                example_id="1",
                video_path=Path("unused.mp4"),
                question="Describe it.",
            )
        )
        self.assertEqual(outcome.selected_indices, (0, 10))

    def test_manifest_maps_answer_text_and_relative_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "x",
                        "video": "clips/x.mp4",
                        "question": "Q?",
                        "options": ["No", "Yes"],
                        "answer": "Yes",
                        "task": "demo",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            examples = load_jsonl_manifest(manifest)
        self.assertEqual(examples[0].gold_label, "B")
        self.assertEqual(examples[0].video_path, root / "clips/x.mp4")


if __name__ == "__main__":
    unittest.main()

