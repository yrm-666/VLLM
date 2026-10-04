from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from molmo2_frame_selector.mvbench import convert_mvbench_annotations


class MVBenchConversionTest(unittest.TestCase):
    def test_converts_answer_text_to_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_root = root / "videos"
            video_root.mkdir()
            (video_root / "x.mp4").write_bytes(b"placeholder")
            annotations = root / "task.json"
            annotations.write_text(
                json.dumps(
                    [
                        {
                            "video": "x.mp4",
                            "question": "What happens?",
                            "candidates": ["Run", "Jump", "Sit"],
                            "answer": "Jump",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            records = convert_mvbench_annotations(
                annotations,
                video_root,
                task="action",
            )
        self.assertEqual(records[0]["answer"], "B")
        self.assertEqual(records[0]["task"], "action")
        self.assertTrue(records[0]["video"].endswith("x.mp4"))

    def test_rejects_answer_outside_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "task.json"
            annotations.write_text(
                json.dumps(
                    [
                        {
                            "video": "x.mp4",
                            "question": "Q?",
                            "candidates": ["A", "B"],
                            "answer": "C",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                convert_mvbench_annotations(
                    annotations,
                    root,
                    task="bad",
                    require_videos=False,
                )


if __name__ == "__main__":
    unittest.main()

