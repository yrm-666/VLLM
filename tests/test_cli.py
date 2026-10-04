from __future__ import annotations

import unittest

from molmo2_frame_selector.cli import run


class CliRunTest(unittest.TestCase):
    def test_serializable_result_contains_selected_indices_and_diagnostics(self) -> None:
        output = run(
            {
                "query": "What happens?",
                "candidate_indices": [0, 10, 20],
                "timestamps": [0.0, 1.0, 2.0],
                "relevance_scores": [0.1, 0.9, 0.2],
                "frame_embeddings": [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            },
            {"num_selected": 2, "num_candidates": 3},
        )
        self.assertEqual(output["query"], "What happens?")
        self.assertEqual(output["candidate_count"], 3)
        self.assertEqual(output["actual_selected_count"], 2)
        self.assertEqual(len(output["selection_steps"]), 2)
        self.assertTrue(set(output["selected_indices"]).issubset({0, 10, 20}))


if __name__ == "__main__":
    unittest.main()

