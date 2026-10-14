"""Shared deterministic guards against structurally leaked benchmark answers."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise


def _common_prefix(values: list[str]) -> str:
    """Longest shared leading substring, compared character by character.

    Keys are not paths, so this is deliberately not ``os.path.commonpath``.
    """
    first, last = min(values), max(values)
    for index, (a, b) in enumerate(zip(first, last, strict=False)):
        if a != b:
            return first[:index]
    return first if len(first) <= len(last) else last


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
        target_prefix = _common_prefix(target_keys) if len(target_keys) > 1 else ""
        distractor_prefix = (
            _common_prefix(distractor_keys) if len(distractor_keys) > 1 else ""
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
