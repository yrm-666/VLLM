from __future__ import annotations

import unittest

from molmo2_frame_selector.metrics import estimate_visual_tokens, parse_choice_label


class MetricsTest(unittest.TestCase):
    def test_molmo2_visual_token_estimate(self) -> None:
        self.assertEqual(estimate_visual_tokens(1), 81)
        self.assertEqual(estimate_visual_tokens(32), 2592)
        self.assertEqual(estimate_visual_tokens(384), 31104)

    def test_choice_parser(self) -> None:
        labels = ["A", "B", "C", "D"]
        self.assertEqual(parse_choice_label("Answer: C", labels), "C")
        self.assertEqual(parse_choice_label("(B) because...", labels), "B")
        self.assertIsNone(parse_choice_label("I cannot decide", labels))

