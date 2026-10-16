"""Tests for shared benchmark helpers."""

import pytest

from benchkit.benchmarks.utils import strip_think_tags


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("no reasoning here", "no reasoning here"),
        ("<think>draft</think>\nanswer", "answer"),
        ("<think>a</think>one <think>b</think>two", "one two"),
        ("answer <think>unfinished", "answer"),
        # The template opened <think> in the prompt; only the close arrives.
        ("draft reasoning</think>\nanswer", "answer"),
        ("</think>answer", "answer"),
        ("draft</think>one <think>b</think>two", "one two"),
        ("draft</think>answer <think>unfinished", "answer"),
    ],
)
def test_strip_think_tags(response: str, expected: str) -> None:
    assert strip_think_tags(response) == expected
