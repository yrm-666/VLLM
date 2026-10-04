from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from molmo2_frame_selector.reporting import compare_jsonl_files, summarize_jsonl


class ReportingTest(unittest.TestCase):
    def test_summarizes_accuracy_and_efficiency(self) -> None:
        records = [
            {"mode": "uniform", "task": "x", "correct": True, "selected_frames": 32},
            {"mode": "uniform", "task": "x", "correct": False, "selected_frames": 32},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = summarize_jsonl(path)
        self.assertEqual(summary["overall"]["accuracy"], 0.5)
        self.assertEqual(summary["groups"]["uniform/x"]["mean_selected_frames"], 32)

    def test_compares_accuracy_to_uniform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uniform = root / "uniform.jsonl"
            query_aware = root / "qa.jsonl"
            uniform.write_text(
                json.dumps(
                    {"mode": "uniform", "task": "x", "example_id": "1", "correct": False}
                )
                + "\n",
                encoding="utf-8",
            )
            query_aware.write_text(
                json.dumps(
                    {"mode": "query_aware", "task": "x", "example_id": "1", "correct": True}
                )
                + "\n",
                encoding="utf-8",
            )
            comparison = compare_jsonl_files([uniform, query_aware])
        self.assertEqual(
            comparison["modes"]["query_aware"]["accuracy_delta_vs_baseline"],
            1.0,
        )
