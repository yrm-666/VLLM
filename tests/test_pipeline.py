from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from molmo2_frame_selector.cache import SQLiteEmbeddingCache
from molmo2_frame_selector.config import SelectorConfig
from molmo2_frame_selector.pipeline import QueryAwareSelectionPipeline
from molmo2_frame_selector.selector import QueryAwareFrameSelector


class FakeEncoder:
    fingerprint = "fake-v1"

    def __init__(self) -> None:
        self.image_calls = 0

    def encode_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def encode_images(self, images):
        self.image_calls += 1
        return [list(image) for image in images]


class SelectionPipelineTest(unittest.TestCase):
    def test_end_to_end_and_cache_reuse(self) -> None:
        selector = QueryAwareFrameSelector(
            SelectorConfig(
                num_selected=2,
                num_candidates=3,
                relevance_weight=1.0,
                diversity_weight=0.0,
                coverage_weight=0.0,
            )
        )
        encoder = FakeEncoder()
        frames = [[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]]
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteEmbeddingCache(Path(directory) / "cache.sqlite3") as cache:
                pipeline = QueryAwareSelectionPipeline(
                    encoder,
                    selector,
                    embedding_cache=cache,
                )
                first = pipeline.select(
                    "query",
                    frames,
                    timestamps=[0.0, 1.0, 2.0],
                    frame_indices=[0, 10, 20],
                    cache_namespace="video",
                )
                second = pipeline.select(
                    "query",
                    frames,
                    timestamps=[0.0, 1.0, 2.0],
                    frame_indices=[0, 10, 20],
                    cache_namespace="video",
                )

        self.assertEqual(first.selection.selected_indices, (10, 20))
        self.assertEqual(first.cache_misses, 3)
        self.assertEqual(second.cache_hits, 3)
        self.assertEqual(second.cache_misses, 0)
        self.assertEqual(encoder.image_calls, 1)

    def test_rejects_misaligned_inputs(self) -> None:
        pipeline = QueryAwareSelectionPipeline(
            FakeEncoder(),
            QueryAwareFrameSelector(
                SelectorConfig(num_selected=1, num_candidates=2)
            ),
        )
        with self.assertRaises(ValueError):
            pipeline.select(
                "query",
                [[1.0, 0.0]],
                timestamps=[0.0, 1.0],
                frame_indices=[0],
            )

