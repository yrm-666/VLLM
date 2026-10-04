from __future__ import annotations

import unittest

from molmo2_frame_selector import (
    official_style_candidate_indices,
    uniform_candidate_indices,
)


class UniformCandidateIndicesTest(unittest.TestCase):
    def test_keeps_short_video_without_duplicates(self) -> None:
        self.assertEqual(uniform_candidate_indices(4, 8), [0, 1, 2, 3])

    def test_keeps_endpoints_and_requested_count(self) -> None:
        indices = uniform_candidate_indices(101, 8)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 100)
        self.assertEqual(len(set(indices)), 8)

    def test_single_candidate_uses_middle_frame(self) -> None:
        self.assertEqual(uniform_candidate_indices(10, 1), [5])

    def test_empty_video(self) -> None:
        self.assertEqual(uniform_candidate_indices(0, 8), [])

    def test_official_style_uses_max_fps_for_short_video(self) -> None:
        indices = official_style_candidate_indices(31, 30.0, 8, max_fps=2.0)
        self.assertEqual(indices, [0, 15, 30])

    def test_official_style_spans_long_video_with_full_budget(self) -> None:
        indices = official_style_candidate_indices(301, 30.0, 8, max_fps=2.0)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 300)

    def test_official_style_rejects_invalid_fps(self) -> None:
        with self.assertRaises(ValueError):
            official_style_candidate_indices(10, 0.0, 8)


if __name__ == "__main__":
    unittest.main()
