"""Shared deterministic guards against structurally leaked benchmark answers."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from os.path import commonprefix


class PromptLeakageError(ValueError):
    """Raised when task construction makes an answer structurally identifiable."""


def assert_candidate_parity(
    *,
    target_keys: list[str],
    distractor_keys: list[str],
    target_values: list[str],
    distractor_values: list[str],
) -> None:
    """Reject simple key markers and value-distribution shortcuts."""
    if target_keys and distractor_keys:
        target_prefix = commonprefix(target_keys) if len(target_keys) > 1 else ""
        distractor_prefix = (
            commonprefix(distractor_keys) if len(distractor_keys) > 1 else ""
        )
        if len(target_prefix) >= 3 and target_prefix != distractor_prefix:
            raise PromptLeakageError("target keys have a distinguishing prefix")
        target_lengths = {len(key) for key in target_keys}
        distractor_lengths = {len(key) for key in distractor_keys}
        if target_lengths.isdisjoint(distractor_lengths):
            raise PromptLeakageError("target keys have a distinguishing length")

    numeric_distractors = [int(value) for value in distractor_values if value.isdigit()]
    numeric_targets = [int(value) for value in target_values if value.isdigit()]
    if len(numeric_distractors) >= 4 and numeric_targets:
        steps = Counter(right - left for left, right in pairwise(numeric_distractors))
        step, occurrences = steps.most_common(1)[0]
        if step and occurrences == len(numeric_distractors) - 1:
            raise PromptLeakageError("distractor values form a derivable sequence")
