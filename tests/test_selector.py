from __future__ import annotations

import unittest

from molmo2_frame_selector import QueryAwareFrameSelector, SelectorConfig


class QueryAwareFrameSelectorTest(unittest.TestCase):
    def test_relevance_only_selects_top_scores_then_returns_chronologically(self) -> None:
        selector = QueryAwareFrameSelector(
            SelectorConfig(
                num_selected=2,
                num_candidates=4,
                relevance_weight=1.0,
                diversity_weight=0.0,
                coverage_weight=0.0,
            )
        )
        result = selector.select(
            [0.1, 0.9, 0.2, 0.8],
            [[1.0, 0.0]] * 4,
            timestamps=[0.0, 1.0, 2.0, 3.0],
        )
        self.assertEqual(result.selected_positions, (1, 3))
        self.assertEqual(
            tuple(step.candidate_position for step in result.selection_steps),
            (1, 3),
        )

    def test_diversity_avoids_near_duplicate(self) -> None:
        selector = QueryAwareFrameSelector(
            SelectorConfig(
                num_selected=2,
                num_candidates=3,
                relevance_weight=0.0,
                diversity_weight=1.0,
                coverage_weight=0.0,
            )
        )
        result = selector.select(
            [1.0, 1.0, 1.0],
            [[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]],
        )
        self.assertEqual(result.selected_positions, (0, 2))

    def test_temporal_coverage_spans_video(self) -> None:
        selector = QueryAwareFrameSelector(
            SelectorConfig(
                num_selected=2,
                num_candidates=4,
                relevance_weight=0.0,
                diversity_weight=0.0,
                coverage_weight=1.0,
            )
        )
        result = selector.select(
            [0.0] * 4,
            [[1.0]] * 4,
            timestamps=[0.0, 1.0, 2.0, 3.0],
        )
        self.assertEqual(result.selected_positions, (0, 3))

    def test_preserves_original_frame_indices(self) -> None:
        selector = QueryAwareFrameSelector(
            SelectorConfig(num_selected=2, num_candidates=3)
        )
        result = selector.select(
            [0.1, 0.9, 0.2],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            timestamps=[0.0, 4.0, 8.0],
            candidate_indices=[10, 20, 30],
        )
        self.assertEqual(len(result.selected_indices), 2)
        self.assertTrue(set(result.selected_indices).issubset({10, 20, 30}))
        self.assertEqual(tuple(sorted(result.selected_timestamps)), result.selected_timestamps)

    def test_short_video_selects_every_frame_without_duplication(self) -> None:
        selector = QueryAwareFrameSelector(
            SelectorConfig(num_selected=4, num_candidates=4)
        )
        result = selector.select(
            [0.2, 0.1],
            [[1.0, 0.0], [0.0, 1.0]],
            candidate_indices=[7, 9],
        )
        self.assertEqual(result.selected_indices, (7, 9))

    def test_rejects_mismatched_lengths(self) -> None:
        selector = QueryAwareFrameSelector(
            SelectorConfig(num_selected=1, num_candidates=2)
        )
        with self.assertRaises(ValueError):
            selector.select([0.5, 0.4], [[1.0]])


if __name__ == "__main__":
    unittest.main()
