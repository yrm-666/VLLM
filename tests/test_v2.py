"""CPU regressions: hard coverage, exact V1 scores, and evaluation safety."""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from molmo2_frame_selector.baselines import selector_config_for_mode
from molmo2_frame_selector.config import SelectorConfig
from molmo2_frame_selector.experiment import VideoQAExample, VideoQAExperiment, load_jsonl_manifest
from molmo2_frame_selector.mvbench import convert_mvbench_annotations
from molmo2_frame_selector.reporting import compare_jsonl_files, summarize_jsonl
from molmo2_frame_selector.selector import (
    QueryAwareFrameSelector, _cosine_distance, _min_max_scale, _normalize_rows,
)
from molmo2_frame_selector.temporal import temporal_bin_ids, temporal_coverage
from molmo2_frame_selector.video import bounded_frame_range
from test_encoders import AmbiguousArray


def naive_v1(scores, rows, times, config):
    """Independent previous exhaustive scoring, retained only as a test oracle."""
    embeddings = _normalize_rows(rows, config.epsilon)
    relevance = _min_max_scale(scores, config.epsilon)
    alpha, beta, gamma = config.normalized_weights
    duration = max(times) - min(times)
    selected, output = [], []
    for _ in range(min(config.num_selected, len(scores))):
        candidates = []
        for position in range(len(scores)):
            if position in selected:
                continue
            diversity = min((_cosine_distance(embeddings[position], embeddings[p])
                             for p in selected), default=0.0)
            coverage = min((abs(times[position] - times[p]) / duration
                            if duration > config.epsilon else 0.0
                            for p in selected), default=0.0)
            total = alpha * relevance[position] + beta * diversity + gamma * coverage
            candidates.append(((total, -times[position], -position), position,
                               (relevance[position], diversity, coverage)))
        key, position, components = max(candidates, key=lambda item: item[0])
        selected.append(position)
        output.append((position, key[0], *components))
    return output


class V2SelectorTest(unittest.TestCase):
    def test_incremental_v1_exactly_matches_previous_scores(self):
        rng = random.Random(813)
        for weights in [(0.6, 0.25, 0.15), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            for count, k in [(1, 1), (12, 8), (32, 16)]:
                scores = [rng.random() for _ in range(count)]
                rows = [[rng.uniform(-1, 1) for _ in range(7)] for _ in scores]
                times = [float(i // 2) for i in range(count)]
                config = SelectorConfig(k, max(k, count), *weights)
                result = QueryAwareFrameSelector(config).select(scores, rows, timestamps=times)
                observed = [(step.candidate_position, step.total_score, step.relevance_score,
                             step.diversity_score, step.coverage_score)
                            for step in result.selection_steps]
                self.assertEqual(observed, naive_v1(scores, rows, times, config))

    def test_v2_covers_all_quarters_despite_clustered_relevance(self):
        times = list(map(float, range(40)))
        scores = [1.0 if 15 <= i < 25 else 0.0 for i in range(40)]
        rows = [[1.0, 0.0]] * 40
        for k in (8, 16):
            config = selector_config_for_mode("query_aware_v2", num_selected=k, num_candidates=40)
            result = QueryAwareFrameSelector(config).select(scores, rows, timestamps=times,
                                                            time_range=(0.0, 40.0))
            self.assertEqual(len(result.selected_indices), k)
            self.assertEqual(temporal_coverage(result.selected_timestamps, (0, 40))[1], 4)
            self.assertEqual(result.selected_timestamps, tuple(sorted(result.selected_timestamps)))
            self.assertEqual(result, QueryAwareFrameSelector(config).select(
                scores, rows, timestamps=times, time_range=(0.0, 40.0)))

    def test_sparse_empty_bins_and_short_video_do_not_break_budget(self):
        config = SelectorConfig(num_selected=8, num_candidates=8, temporal_bins=4)
        result = QueryAwareFrameSelector(config).select([1, 0, 1], [[1], [1], [1]],
                                                        timestamps=[0, 1, 9], time_range=(0, 10))
        self.assertEqual(result.selected_indices, (0, 1, 2))
        result = QueryAwareFrameSelector(config).select([1, 1], [[0], [0]], timestamps=[2, 2])
        self.assertEqual(result.selected_indices, (0, 1))
        self.assertEqual(QueryAwareFrameSelector(config).select([], []).selected_indices, ())

    def test_small_budget_uses_budget_sized_bins(self):
        config = SelectorConfig(num_selected=2, num_candidates=8, temporal_bins=4)
        result = QueryAwareFrameSelector(config).select([1, 1, 0, 0], [[1]] * 4,
                                                        timestamps=[0, 1, 6, 9], time_range=(0, 10))
        self.assertEqual(set(temporal_bin_ids(result.selected_timestamps, 2, (0, 10))), {0, 1})

    def test_distance_work_is_linear_in_number_of_greedy_steps(self):
        count, k = 40, 16
        with patch("molmo2_frame_selector.selector._cosine_distance", wraps=_cosine_distance) as metric:
            QueryAwareFrameSelector(SelectorConfig(k, count)).select([1] * count, [[1]] * count)
        self.assertEqual(metric.call_count, sum(count - rank for rank in range(1, k)))


class TemporalSafetyTest(unittest.TestCase):
    def test_temporal_coverage_accepts_ndarray_style_times(self):
        self.assertEqual(temporal_coverage(AmbiguousArray([0.0, 3.0]), (0.0, 4.0)), (0.75, 2))

    def test_half_open_bounds_preserve_original_frame_positions(self):
        self.assertEqual(bounded_frame_range((0, 1, 2, 3, 4), 5, 1, 4), (1, 4, (1, 4)))
        self.assertEqual(bounded_frame_range((0, 1, 2), 3, 1, 10), (1, 3, (1, 3)))

    def test_invalid_bounds_and_bins_rejected(self):
        for start, end in [(-1, 2), (2, 2), (4, 5), (0, float("nan"))]:
            with self.assertRaises(ValueError):
                bounded_frame_range((0, 1, 2), 3, start, end)
        with self.assertRaises(ValueError):
            temporal_bin_ids([5], 4, (0, 4))
        self.assertEqual(temporal_bin_ids([0, 1, 2, 3, 4], 4, (0, 4)), (0, 1, 2, 3, 3))

    def test_converter_offset_and_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "items.json"
            row = {"video": "x.mp4", "question": "Q?", "candidates": ["yes", "no"],
                   "answer": "yes", "start": 2, "end": 5}
            annotation.write_text(json.dumps([row, row]), encoding="utf-8")
            result = convert_mvbench_annotations(annotation, root, task="demo", offset=1,
                                                 limit=1, require_videos=False)
            self.assertEqual(result[0]["id"], "demo-0001")
            self.assertEqual((result[0]["start_seconds"], result[0]["end_seconds"]), (2, 5))
            del row["end"]
            annotation.write_text(json.dumps([row]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both start and end"):
                convert_mvbench_annotations(annotation, root, task="demo", require_videos=False)

    def test_manifest_rejects_duplicate_ids_and_loads_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            row = {"id": "x", "video": "x.mp4", "question": "Q?",
                   "start_seconds": 1, "end_seconds": 2}
            line = json.dumps(row) + "\n"
            path.write_text(line, encoding="utf-8")
            example = load_jsonl_manifest(path)[0]
            self.assertEqual((example.start_seconds, example.end_seconds), (1, 2))
            path.write_text(line * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate example ID"):
                load_jsonl_manifest(path)

    def test_failures_count_in_accuracy_not_success_latency(self):
        class BrokenDecoder:
            def decode(self, *args, **kwargs):
                raise RuntimeError("invalid video")
        experiment = VideoQAExperiment(mode="uniform", decoder=BrokenDecoder(), molmo_runner=None)
        outcome = experiment.run_safely(VideoQAExample("bad", Path("x"), "Q?", gold_label="A"))
        self.assertEqual(outcome.record.status, "error")
        self.assertFalse(outcome.record.correct)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            good = {"correct": True, "status": "ok", "end_to_end_seconds": 2.0}
            path.write_text(json.dumps(good) + "\n" + json.dumps(outcome.as_dict()) + "\n",
                            encoding="utf-8")
            summary = summarize_jsonl(path)["overall"]
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["failed_examples"], 1)
        self.assertEqual(summary["mean_end_to_end_seconds"], 2.0)

    def test_comparison_rejects_different_sample_sets_and_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, variant = root / "a.jsonl", root / "b.jsonl"
            baseline.write_text(json.dumps({"mode": "uniform", "example_id": "1", "query": "Q"}),
                                encoding="utf-8")
            for row in [{"mode": "query_aware", "example_id": "2", "query": "Q"},
                        {"mode": "query_aware", "example_id": "1", "query": "different"}]:
                variant.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaises(ValueError):
                    compare_jsonl_files([baseline, variant])

    def test_uniform_without_selector_revision_can_be_compared_to_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for mode, revision in [("uniform", None), ("query_aware", "same"), ("query_aware_v2", "same")]:
                path = Path(directory) / f"{mode}.jsonl"
                path.write_text(json.dumps({"mode": mode, "example_id": "1", "correct": True,
                                            "resolved_selector_revision": revision}), encoding="utf-8")
                paths.append(path)
            self.assertEqual(len(compare_jsonl_files(paths)["modes"]), 3)


if __name__ == "__main__":
    unittest.main()
