"""Question construction helpers shared by selector and Molmo2."""

from __future__ import annotations

from collections.abc import Sequence


def build_query_text(
    question: str,
    options: Sequence[str] | None = None,
    *,
    option_labels: Sequence[str] | None = None,
) -> str:
    """Build the selector query without leaking a ground-truth answer.

    Multiple-choice options are part of the task semantics, so they are included
    in the query used by both the selector and Molmo2. The caller never passes the
    answer to this function.
    """

    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")
    if not options:
        return question

    clean_options = [str(option).strip() for option in options]
    if any(not option for option in clean_options):
        raise ValueError("options cannot contain empty values")

    if option_labels is None:
        if len(clean_options) > 26:
            raise ValueError("automatic option labels support at most 26 options")
        labels = [chr(ord("A") + index) for index in range(len(clean_options))]
    else:
        labels = [str(label).strip() for label in option_labels]
        if len(labels) != len(clean_options):
            raise ValueError("option_labels and options must have the same length")
        if any(not label for label in labels):
            raise ValueError("option_labels cannot contain empty values")

    option_lines = [
        f"({label}) {option}" for label, option in zip(labels, clean_options)
    ]
    return "\n".join([question, *option_lines])

