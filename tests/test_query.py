from __future__ import annotations

import unittest

from molmo2_frame_selector.query import build_query_text


class BuildQueryTextTest(unittest.TestCase):
    def test_question_without_options_is_unchanged(self) -> None:
        self.assertEqual(build_query_text("  What happens?  "), "What happens?")

    def test_multiple_choice_includes_all_options_but_no_answer(self) -> None:
        query = build_query_text("What is picked up?", ["Book", "Cup", "Phone"])
        self.assertEqual(
            query,
            "What is picked up?\n(A) Book\n(B) Cup\n(C) Phone",
        )

    def test_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            build_query_text(" ")

