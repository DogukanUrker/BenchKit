"""Deterministic detection of repetitive model generations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Literal

LoopState = Literal["unavailable", "observing", "clear", "suspected", "looping"]
LoopSource = Literal["none", "thinking", "answer"]
LoopEvidence = Literal[
    "none",
    "global_repetition",
    "exact_cycle",
    "numeric_cycle",
    "structural_cycle",
]

_WORD_RE = re.compile(r"[^\W_]+(?:['’.-][^\W_]+)*", re.UNICODE)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_NUMBER_RE = re.compile(r"\d+")
_CODE_LINE_RE = re.compile(
    r"(?m)^\s*(?:def|class|if|elif|else\s*:|for|while|try\s*:|except\b|"
    r"return\b|assert\b|import\b|from\b|#include\b|function\b|"
    r"public\b|private\b|protected\b|const\b|let\b|var\b)"
)
_OPEN_THINK = "<think>"
_CLOSE_THINK = "</think>"
_ANALYSIS_WORD_LIMIT = 4096
_ANALYSIS_TOKEN_LIMIT = 4096
_ANALYSIS_CHAR_LIMIT = 131_072
_MAX_CYCLE_PERIOD = 1024
_MAX_ACTIONABLE_PERIOD = 512
_ACTIVE_TAIL_SLACK = 2
_ADVISORY_TAIL_SLACK = 8
_CONFIRM_GROWTH_CHARS = 256
_CONFIRM_GROWTH_TOKENS = 32
_CODE_KEYWORDS = {
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "def",
    "default",
    "del",
    "do",
    "elif",
    "else",
    "except",
    "false",
    "finally",
    "for",
    "from",
    "function",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "let",
    "match",
    "new",
    "none",
    "not",
    "null",
    "or",
    "pass",
    "private",
    "protected",
    "public",
    "raise",
    "return",
    "static",
    "switch",
    "throw",
    "true",
    "try",
    "var",
    "while",
    "with",
    "yield",
}
_SEVERITY: dict[LoopState, int] = {
    "unavailable": 0,
    "observing": 1,
    "clear": 1,
    "suspected": 2,
    "looping": 3,
}


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
    evidence: LoopEvidence = "none"
    active_cycle: bool = False
    confirmed_cycle: bool = False
    recovered_cycle: bool = False
    cycle_period_tokens: int = 0
    cycle_repetitions: int = 0
    repeated_suffix_tokens: int = 0
    cycle_tail_tokens: int = 0
    evidence_growth_chars: int = 0
    evidence_growth_tokens: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Cycle:
    """One suffix-anchored periodic sequence."""

    period: int
    repetitions: int
    span: int
    tail: int
    signature: tuple[str, ...]


@dataclass
class _CycleTracker:
    """Growth state for a live high-confidence suffix cycle."""

    fingerprint: tuple[LoopEvidence, int, tuple[str, ...]] | None = None
    first_chars: int = 0
    first_span: int = 0
    confirmations: int = 0

    def reset(self) -> None:
        self.fingerprint = None
        self.first_chars = 0
        self.first_span = 0
        self.confirmations = 0


@dataclass(frozen=True)
class _TextAnalysis:
    snapshot: LoopSnapshot
    fingerprint: tuple[LoopEvidence, int, tuple[str, ...]] | None = None


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
        self._trackers: dict[LoopSource, _CycleTracker] = {
            "thinking": _CycleTracker(),
            "answer": _CycleTracker(),
        }
        self._last_confirmed_cycle: LoopSnapshot | None = None

    def add(self, *, thinking: str = "", answer: str = "") -> None:
        if thinking:
            self._thinking_pending.append(thinking)
            self.thinking_chars += len(thinking)
        if answer:
            self._answer_pending.append(answer)
            self.answer_chars += len(answer)

    def reconcile(self, *, thinking: str = "", answer: str = "") -> bool:
        """Append final text omitted from callbacks if it matches the stream.

        ``False`` means the provider's final payload disagrees with streamed
        content, so callers should rebuild analysis from that authoritative
        payload instead of combining incompatible generations.
        """
        thinking_suffix = _unseen_suffix(
            retained=self.thinking,
            streamed_chars=self.thinking_chars,
            final_text=thinking,
        )
        answer_suffix = _unseen_suffix(
            retained=self.answer,
            streamed_chars=self.answer_chars,
            final_text=answer,
        )
        if thinking_suffix is None or answer_suffix is None:
            return False
        self.add(thinking=thinking_suffix, answer=answer_suffix)
        return True

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
        if not self.thinking_chars and not self.answer_chars:
            return LoopSnapshot(
                state="unavailable",
                score=0.0,
                source="none",
                analyzed_words=0,
                repeated_ngram_coverage=0.0,
                max_window_similarity=0.0,
                low_novelty_windows=0,
                max_repeated_block=0,
            )

        if not final:
            # Only the channel currently receiving visible output is actionable.
            # Historical thinking must not kill a healthy answer after the model
            # has moved on from its reasoning phase.
            source: LoopSource = "answer" if self.answer_chars else "thinking"
            text = self.answer if source == "answer" else self.thinking
            total_chars = (
                self.answer_chars if source == "answer" else self.thinking_chars
            )
            analysis = _analyze_text(text, source, final=False)
            return self._track_live_cycle(analysis, source, total_chars)

        # A final report retains genuine repetition from either channel, while
        # live action above remains scoped to the active channel.
        thinking_snapshot: LoopSnapshot | None = None
        answer_snapshot: LoopSnapshot | None = None
        if self.thinking_chars:
            thinking_snapshot = _analyze_text(
                self.thinking,
                "thinking",
                final=True,
            ).snapshot
        if self.answer_chars:
            answer_snapshot = _analyze_text(
                self.answer,
                "answer",
                final=True,
            ).snapshot

        # A visible answer cycle is never hidden by a worse thinking score.
        if answer_snapshot is not None and answer_snapshot.active_cycle:
            return answer_snapshot

        if thinking_snapshot is not None and thinking_snapshot.active_cycle:
            if answer_snapshot is None:
                return thinking_snapshot
            return replace(
                thinking_snapshot,
                state="suspected",
                active_cycle=False,
                confirmed_cycle=False,
                recovered_cycle=True,
            )

        # Preserve a cycle that was confirmed during streaming but later broke
        # before the generation ended. It is useful diagnostic evidence, but it
        # is neither actionable nor counted as a final loop.
        if self._last_confirmed_cycle is not None:
            return replace(
                self._last_confirmed_cycle,
                state="suspected",
                active_cycle=False,
                confirmed_cycle=False,
                recovered_cycle=True,
            )

        candidates = [
            snapshot
            for snapshot in (thinking_snapshot, answer_snapshot)
            if snapshot is not None
        ]
        selected = max(
            candidates,
            key=lambda snap: (_SEVERITY[snap.state], snap.score),
        )
        return selected

    def _track_live_cycle(
        self,
        analysis: _TextAnalysis,
        source: LoopSource,
        total_chars: int,
    ) -> LoopSnapshot:
        snapshot = analysis.snapshot
        tracker = self._trackers[source]
        fingerprint = analysis.fingerprint
        if not snapshot.active_cycle or fingerprint is None:
            tracker.reset()
            return snapshot

        if tracker.fingerprint != fingerprint:
            tracker.fingerprint = fingerprint
            tracker.first_chars = total_chars
            tracker.first_span = snapshot.repeated_suffix_tokens
            tracker.confirmations = 1
        else:
            tracker.confirmations += 1

        growth_chars = max(0, total_chars - tracker.first_chars)
        growth_tokens = max(
            0,
            snapshot.repeated_suffix_tokens - tracker.first_span,
        )
        confirmed = (
            tracker.confirmations >= 2
            and growth_chars >= _CONFIRM_GROWTH_CHARS
            and (
                growth_tokens >= _CONFIRM_GROWTH_TOKENS
                or snapshot.repeated_suffix_tokens >= _ANALYSIS_TOKEN_LIMIT * 0.9
            )
        )
        tracked = replace(
            snapshot,
            state="looping" if confirmed else "suspected",
            confirmed_cycle=confirmed,
            evidence_growth_chars=growth_chars,
            evidence_growth_tokens=growth_tokens,
        )
        if confirmed:
            self._last_confirmed_cycle = tracked
        return tracked


def _analyze_text(
    text: str,
    source: LoopSource,
    *,
    final: bool,
) -> _TextAnalysis:
    """Analyze one visible generation channel."""

    words = _words(text)[-_ANALYSIS_WORD_LIMIT:]
    tokens = _tokens(text)[-_ANALYSIS_TOKEN_LIMIT:]
    count = len(words)
    token_count = len(tokens)
    coverage = _repeated_ngram_coverage(words, size=8) if count >= 8 else 0.0
    similarity, low_novelty = _window_novelty(words) if count >= 128 else (0.0, 0)
    repeated_block = _max_repeated_block(words, size=20) if count >= 20 else 0

    repeat_strength = _scaled(coverage, 0.08, 0.50)
    similarity_strength = _scaled(similarity, 0.55, 0.95)
    run_strength = min(1.0, low_novelty / 3)
    block_strength = min(1.0, max(0, repeated_block - 1) / 3)
    blended = (
        0.35 * repeat_strength
        + 0.30 * similarity_strength
        + 0.20 * run_strength
        + 0.15 * block_strength
    )
    global_score = round(min(0.69, blended), 3)
    global_suspected = count >= 128 and (
        coverage >= 0.18 or repeated_block >= 2 or low_novelty >= 1 or blended >= 0.45
    )

    raw_cycles = _suffix_cycle_candidates(tokens)
    numeric_tokens = [_normalize_numbers(token) for token in tokens]
    numeric_cycles = _suffix_cycle_candidates(numeric_tokens)
    exact = _best_cycle(
        raw_cycles,
        max_period=_MAX_ACTIONABLE_PERIOD,
        max_tail=_ACTIVE_TAIL_SLACK,
        min_repetitions=4,
        min_span=64,
    )
    numeric = _best_cycle(
        numeric_cycles,
        max_period=_MAX_ACTIONABLE_PERIOD,
        max_tail=_ACTIVE_TAIL_SLACK,
        min_repetitions=6,
        min_span=96,
    )
    # A changing counter is only high-confidence when literal language is also
    # overwhelmingly repetitive. This keeps numbered tables and varied tests
    # from becoming actionable merely because their shapes match.
    if numeric is not None and coverage < 0.65:
        numeric = None

    active_cycle = exact or numeric
    evidence: LoopEvidence = "none"
    fingerprint: tuple[LoopEvidence, int, tuple[str, ...]] | None = None
    if active_cycle is not None:
        evidence = "exact_cycle" if exact is not None else "numeric_cycle"
        cycle_score = _cycle_score(active_cycle)
        snapshot = _snapshot(
            state="looping" if final else "suspected",
            score=cycle_score,
            source=source,
            count=count,
            coverage=coverage,
            similarity=similarity,
            low_novelty=low_novelty,
            repeated_block=repeated_block,
            evidence=evidence,
            cycle=active_cycle,
            active_cycle=True,
        )
        fingerprint = (evidence, active_cycle.period, active_cycle.signature)
        return _TextAnalysis(snapshot, fingerprint)

    advisory_cycles: list[tuple[LoopEvidence, _Cycle]] = []
    raw_advisory = _best_cycle(
        raw_cycles,
        max_period=_MAX_CYCLE_PERIOD,
        max_tail=_ADVISORY_TAIL_SLACK,
        min_repetitions=3,
        min_span=48,
    )
    numeric_advisory = _best_cycle(
        numeric_cycles,
        max_period=_MAX_CYCLE_PERIOD,
        max_tail=_ADVISORY_TAIL_SLACK,
        min_repetitions=4,
        min_span=64,
    )
    if raw_advisory is not None:
        advisory_cycles.append(("exact_cycle", raw_advisory))
    if numeric_advisory is not None and coverage >= 0.18:
        advisory_cycles.append(("numeric_cycle", numeric_advisory))

    if _looks_like_code(text):
        structural_cycles = _suffix_cycle_candidates(_structural_tokens(text))
        structural = _best_cycle(
            structural_cycles,
            max_period=_MAX_CYCLE_PERIOD,
            max_tail=_ADVISORY_TAIL_SLACK,
            min_repetitions=4,
            min_span=64,
        )
        if structural is not None:
            advisory_cycles.append(("structural_cycle", structural))

    advisory = _strongest_advisory(advisory_cycles)
    if advisory is not None:
        evidence, cycle = advisory
        advisory_score = _advisory_cycle_score(cycle)
        return _TextAnalysis(
            _snapshot(
                state="suspected",
                score=max(global_score, advisory_score),
                source=source,
                count=count,
                coverage=coverage,
                similarity=similarity,
                low_novelty=low_novelty,
                repeated_block=repeated_block,
                evidence=evidence,
                cycle=cycle,
            )
        )

    state: LoopState
    if global_suspected:
        state = "suspected"
        evidence = "global_repetition"
    elif final:
        state = "clear"
    else:
        state = "observing" if token_count < 64 else "clear"

    return _TextAnalysis(
        _snapshot(
            state=state,
            score=global_score,
            source=source,
            count=count,
            coverage=coverage,
            similarity=similarity,
            low_novelty=low_novelty,
            repeated_block=repeated_block,
            evidence=evidence,
        )
    )


def _snapshot(
    *,
    state: LoopState,
    score: float,
    source: LoopSource,
    count: int,
    coverage: float,
    similarity: float,
    low_novelty: int,
    repeated_block: int,
    evidence: LoopEvidence = "none",
    cycle: _Cycle | None = None,
    active_cycle: bool = False,
) -> LoopSnapshot:
    return LoopSnapshot(
        state=state,
        score=round(score, 3),
        source=source,
        analyzed_words=count,
        repeated_ngram_coverage=round(coverage, 3),
        max_window_similarity=round(similarity, 3),
        low_novelty_windows=low_novelty,
        max_repeated_block=repeated_block,
        evidence=evidence,
        active_cycle=active_cycle,
        cycle_period_tokens=cycle.period if cycle else 0,
        cycle_repetitions=cycle.repetitions if cycle else 0,
        repeated_suffix_tokens=cycle.span if cycle else 0,
        cycle_tail_tokens=cycle.tail if cycle else 0,
    )


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def _normalize_numbers(token: str) -> str:
    return "<num>" if _NUMBER_RE.fullmatch(token) else token


def _suffix_cycle_candidates(tokens: list[str]) -> list[_Cycle]:
    """Return exact periodic sequences ending at, or very near, the suffix.

    A Z-array over reversed tokens finds every prior copy of the current suffix
    in linear time. This is the same suffix-matching primitive used by DRY
    sampling, with stricter repetition thresholds applied by this detector.
    """
    count = len(tokens)
    if count < 32:
        return []

    candidates: list[_Cycle] = []
    for tail in range(min(_ADVISORY_TAIL_SLACK, count - 1) + 1):
        end = count - tail
        reversed_tokens = list(reversed(tokens[:end]))
        suffix_matches = _z_values(reversed_tokens)
        primitive_periods: list[int] = []
        max_period = min(_MAX_CYCLE_PERIOD, end - 1)
        for period in range(1, max_period + 1):
            if any(period % primitive == 0 for primitive in primitive_periods):
                continue

            matched = min(suffix_matches[period], end - period)
            span = matched + period if matched else 0
            repetitions = span // period if period else 0
            if repetitions < 2 or span < 32:
                continue

            primitive_periods.append(period)
            pattern = tokens[end - period : end]
            candidates.append(
                _Cycle(
                    period=period,
                    repetitions=repetitions,
                    span=span,
                    tail=tail,
                    signature=_cycle_signature(pattern),
                )
            )
    return candidates


def _z_values(sequence: list[str]) -> list[int]:
    """Return longest prefix matches at every sequence offset in O(n)."""
    count = len(sequence)
    if not count:
        return []
    values = [0] * count
    values[0] = count
    left = 0
    right = 0
    for index in range(1, count):
        if index <= right:
            values[index] = min(right - index + 1, values[index - left])
        while (
            index + values[index] < count
            and sequence[values[index]] == sequence[index + values[index]]
        ):
            values[index] += 1
        match_right = index + values[index] - 1
        if match_right > right:
            left = index
            right = match_right
    return values


def _cycle_signature(pattern: list[str]) -> tuple[str, ...]:
    """Build an exact, rotation-invariant signature for one cycle."""
    if not pattern:
        return ()

    # Booth's algorithm avoids both hash collisions and ambiguity between
    # distinct cycles that happen to contain the same token bigrams.
    sequence = tuple(pattern)
    doubled = sequence + sequence
    count = len(sequence)
    left = 0
    right = 1
    offset = 0
    while left < count and right < count and offset < count:
        left_token = doubled[left + offset]
        right_token = doubled[right + offset]
        if left_token == right_token:
            offset += 1
            continue
        if left_token > right_token:
            left = left + offset + 1
            if left == right:
                left += 1
        else:
            right = right + offset + 1
            if left == right:
                right += 1
        offset = 0
    start = min(left, right)
    return doubled[start : start + count]


def _best_cycle(
    candidates: list[_Cycle],
    *,
    max_period: int,
    max_tail: int,
    min_repetitions: int,
    min_span: int,
) -> _Cycle | None:
    eligible = [
        cycle
        for cycle in candidates
        if cycle.period <= max_period
        and cycle.tail <= max_tail
        and cycle.repetitions >= min_repetitions
        and cycle.span >= min_span
    ]
    if not eligible:
        return None
    maximum_span = max(cycle.span for cycle in eligible)
    near_maximum = [cycle for cycle in eligible if cycle.span >= maximum_span * 0.9]
    return max(
        near_maximum,
        key=lambda cycle: (
            cycle.repetitions,
            cycle.span,
            -cycle.period,
            -cycle.tail,
        ),
    )


def _cycle_score(cycle: _Cycle) -> float:
    repetition_strength = _scaled(cycle.repetitions, 4, 8)
    span_strength = _scaled(cycle.span, 64, 256)
    return 0.80 + 0.10 * repetition_strength + 0.10 * span_strength


def _advisory_cycle_score(cycle: _Cycle) -> float:
    repetition_strength = _scaled(cycle.repetitions, 3, 8)
    span_strength = _scaled(cycle.span, 48, 256)
    return min(0.74, 0.55 + 0.10 * repetition_strength + 0.09 * span_strength)


def _strongest_advisory(
    candidates: list[tuple[LoopEvidence, _Cycle]],
) -> tuple[LoopEvidence, _Cycle] | None:
    if not candidates:
        return None
    priority = {"exact_cycle": 3, "numeric_cycle": 2, "structural_cycle": 1}
    return max(
        candidates,
        key=lambda item: (
            _advisory_cycle_score(item[1]),
            priority.get(item[0], 0),
            item[1].span,
        ),
    )


def _looks_like_code(text: str) -> bool:
    tail = text[-16_384:]
    return "```" in tail or len(_CODE_LINE_RE.findall(tail)) >= 3


def _structural_tokens(text: str) -> list[str]:
    structured: list[str] = []
    for line in text[-_ANALYSIS_CHAR_LIMIT:].splitlines()[-512:]:
        indent = len(line) - len(line.lstrip())
        structured.append(f"<indent:{min(8, indent // 4)}>")
        for token in _tokens(line):
            if _NUMBER_RE.search(token):
                structured.append("<num>")
            elif token.replace("_", "").isalnum() and token not in _CODE_KEYWORDS:
                structured.append("<id>")
            else:
                structured.append(token)
        structured.append("<nl>")
    return structured[-_ANALYSIS_TOKEN_LIMIT:]


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def _materialize_tail(current: str, pending: list[str]) -> str:
    if not pending:
        return current
    text = (current + "".join(pending))[-_ANALYSIS_CHAR_LIMIT:]
    pending.clear()
    return text


def _unseen_suffix(
    *,
    retained: str,
    streamed_chars: int,
    final_text: str,
) -> str | None:
    """Return the part of a matching final payload not seen while streaming."""
    if streamed_chars > len(final_text):
        return None
    retained_start = max(0, streamed_chars - len(retained))
    if final_text[retained_start:streamed_chars] != retained:
        return None
    return final_text[streamed_chars:]


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
