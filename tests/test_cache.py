from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from molmo2_frame_selector.cache import (
    SQLiteEmbeddingCache,
    cache_namespace,
    video_file_identity,
)


class SQLiteEmbeddingCacheTest(unittest.TestCase):
    def test_round_trip_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            with SQLiteEmbeddingCache(path) as cache:
                cache.put("model-video", "10", [0.25, -0.5])
                self.assertEqual(cache.get("model-video", "10"), [0.25, -0.5])
                cache.put("model-video", "10", [1.0, 2.0])
                self.assertEqual(cache.get("model-video", "10"), [1.0, 2.0])

    def test_video_identity_changes_with_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.bin"
            path.write_bytes(b"first")
            first = video_file_identity(path)
            path.write_bytes(b"second version")
            second = video_file_identity(path)
            self.assertNotEqual(first, second)
            self.assertNotEqual(
                cache_namespace(encoder_fingerprint="a", video_identity=first),
                cache_namespace(encoder_fingerprint="b", video_identity=first),
            )

