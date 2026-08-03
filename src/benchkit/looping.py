"""Deterministic detection of repetitive model generations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

LoopState = Literal["unavailable", "observing", "clear", "suspected", "looping"]
LoopSource = Literal["none", "thinking", "answer"]
LOOP_ANALYZER_VERSION = "1"

_WORD_RE = re.compile(r"[^\W_]+(?:['’.-][^\W_]+)*", re.UNICODE)
_OPEN_THINK = "<think>"
_CLOSE_THINK = "</think>"
_ANALYSIS_WORD_LIMIT = 4096
_ANALYSIS_CHAR_LIMIT = 131_072


@dataclass(frozen=True)
class LoopSnapshot:
    """Current repetition signals for one generation."""

    state: LoopState
    score: float
    source: LoopSource
    analyzed_words: int
    repeated_ngram_coverage: float
    max_window_similarity: float
    low_novelty_windows: int
    max_repeated_block: int

    def as_dict(self) -> dict:
        return asdict(self)


class InlineThinkingParser:
    """Split streamed ``<think>`` content even when tags cross chunks."""

    def __init__(self) -> None:
        self._state: Literal["start", "thinking", "answer"] = "start"
        self._buffer = ""
        self.saw_marker = False

    def feed(self, text: str) -> tuple[str, str]:
        """Return newly available ``(thinking, answer)`` text."""
        if not text:
            return "", ""
        self._buffer += text

        if self._state == "start":
            stripped = self._buffer.lstrip()
            if not stripped:
                return "", ""
            if _OPEN_THINK.startswith(stripped):
                return "", ""
            if stripped.startswith(_OPEN_THINK):
                self.saw_marker = True
                self._state = "thinking"
                self._buffer = stripped[len(_OPEN_THINK) :]
            else:
                self._state = "answer"

        if self._state == "answer":
            answer, self._buffer = self._buffer, ""
            return "", answer

        closing = self._buffer.find(_CLOSE_THINK)
        if closing >= 0:
            thinking = self._buffer[:closing]
            answer = self._buffer[closing + len(_CLOSE_THINK) :]
            self._buffer = ""
            self._state = "answer"
            return thinking, answer

        keep = _partial_suffix_length(self._buffer, _CLOSE_THINK)
        if keep:
            thinking = self._buffer[:-keep]
            self._buffer = self._buffer[-keep:]
        else:
            thinking, self._buffer = self._buffer, ""
        return thinking, ""

    def finish(self) -> tuple[str, str]:
        """Flush buffered text when the stream ends."""
        if self._state == "thinking":
            thinking, answer = self._buffer, ""
        else:
            thinking, answer = "", self._buffer
        self._buffer = ""
        return thinking, answer


class LoopAnalyzer:
    """Incrementally retain text and calculate bounded-cost loop signals."""

    def __init__(self) -> None:
        self._thinking_tail = ""
        self._answer_tail = ""
        self._thinking_pending: list[str] = []
        self._answer_pending: list[str] = []
        self.thinking_chars = 0
        self.answer_chars = 0

    def add(self, *, thinking: str = "", answer: str = "") -> None:
        if thinking:
            self._thinking_pending.append(thinking)
            self.thinking_chars += len(thinking)
        if answer:
            self._answer_pending.append(answer)
            self.answer_chars += len(answer)

    @property
    def thinking(self) -> str:
        self._thinking_tail = _materialize_tail(
            self._thinking_tail,
            self._thinking_pending,
        )
        return self._thinking_tail

    @property
    def answer(self) -> str:
        self._answer_tail = _materialize_tail(
            self._answer_tail,
            self._answer_pending,
        )
        return self._answer_tail

    def snapshot(self, *, final: bool = False) -> LoopSnapshot:
        if self.thinking_chars:
            thinking = _analyze_text(self.thinking, "thinking", final=final)
            if self.answer_chars:
                answer = _analyze_text(self.answer, "answer", final=final)
                severity = {
                    "unavailable": 0,
                    "observing": 1,
                    "clear": 1,
                    "suspected": 2,
                    "looping": 3,
                }
                if severity[answer.state] > severity[thinking.state]:
                    return answer
            return thinking
        if self.answer_chars:
            return _analyze_text(self.answer, "answer", final=final)
        return LoopSnapshot("unavailable", 0.0, "none", 0, 0.0, 0.0, 0, 0)


def _analyze_text(
    text: str,
    source: LoopSource,
    *,
    final: bool,
) -> LoopSnapshot:
    """Analyze one visible generation channel."""

    words = _words(text)[-_ANALYSIS_WORD_LIMIT:]
    count = len(words)
    if count < 8:
        state: LoopState = "clear" if final else "observing"
        return LoopSnapshot(state, 0.0, source, count, 0.0, 0.0, 0, 0)

    coverage = _repeated_ngram_coverage(words, size=8)
    similarity, low_novelty = _window_novelty(words)
    repeated_block = _max_repeated_block(words, size=20)

    repeat_strength = _scaled(coverage, 0.08, 0.50)
    similarity_strength = _scaled(similarity, 0.55, 0.95)
    run_strength = min(1.0, low_novelty / 3)
    block_strength = min(1.0, max(0, repeated_block - 1) / 3)
    score = round(
        0.35 * repeat_strength
        + 0.30 * similarity_strength
        + 0.20 * run_strength
        + 0.15 * block_strength,
        3,
    )

    if count < 128:
        state = "clear" if final else "observing"
    elif count >= 192 and (
        (coverage >= 0.30 and repeated_block >= 3)
        or (low_novelty >= 2 and similarity >= 0.82)
        or score >= 0.75
    ):
        state = "looping"
    elif coverage >= 0.18 or repeated_block >= 2 or low_novelty >= 1 or score >= 0.45:
        state = "suspected"
    else:
        state = "clear"

    return LoopSnapshot(
        state=state,
        score=score,
        source=source,
        analyzed_words=count,
        repeated_ngram_coverage=round(coverage, 3),
        max_window_similarity=round(similarity, 3),
        low_novelty_windows=low_novelty,
        max_repeated_block=repeated_block,
    )


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def _materialize_tail(current: str, pending: list[str]) -> str:
    if not pending:
        return current
    text = (current + "".join(pending))[-_ANALYSIS_CHAR_LIMIT:]
    pending.clear()
    return text


def _partial_suffix_length(text: str, marker: str) -> int:
    limit = min(len(text), len(marker) - 1)
    for size in range(limit, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


def _repeated_ngram_coverage(words: list[str], *, size: int) -> float:
    if len(words) < size:
        return 0.0
    ngrams = [
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    ]
    counts = Counter(ngrams)
    covered = bytearray(len(words))
    for index, ngram in enumerate(ngrams):
        if counts[ngram] > 1:
            covered[index : index + size] = b"\x01" * size
    return sum(covered) / len(words)


def _window_novelty(words: list[str]) -> tuple[float, int]:
    size = 64
    if len(words) < size * 2:
        return 0.0, 0

    windows = [
        _shingles(words[index : index + size], size=3)
        for index in range(0, len(words) - size + 1, size)
    ]
    maximum = 0.0
    longest_run = 0
    current_run = 0
    for index in range(1, len(windows)):
        similarity = max(
            (_jaccard(windows[index], previous) for previous in windows[:index]),
            default=0.0,
        )
        maximum = max(maximum, similarity)
        if similarity >= 0.72:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return maximum, longest_run


def _shingles(words: list[str], *, size: int) -> set[tuple[str, ...]]:
    return {
        tuple(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    }


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _max_repeated_block(words: list[str], *, size: int) -> int:
    if len(words) < size:
        return 0
    counts = Counter(
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    )
    return max(counts.values(), default=0)


def _scaled(value: float, floor: float, ceiling: float) -> float:
    if value <= floor:
        return 0.0
    return min(1.0, (value - floor) / (ceiling - floor))
