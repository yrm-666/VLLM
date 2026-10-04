from __future__ import annotations

import unittest

from molmo2_frame_selector.encoders import cosine_similarities, l2_normalize


class EncoderMathTest(unittest.TestCase):
    def test_l2_normalize(self) -> None:
        self.assertEqual(l2_normalize([3.0, 4.0]), [0.6, 0.8])

    def test_cosine_similarities(self) -> None:
        scores = cosine_similarities(
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        )
        self.assertEqual(scores, [1.0, 0.0, -1.0])

    def test_zero_embedding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            l2_normalize([0.0, 0.0])

