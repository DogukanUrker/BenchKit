"""Helpers for messy multiple-choice model outputs."""

import re

DEFAULT_LETTERS = "ABCD"

# Letters that are also ordinary English words. Once the option list runs past
# D a bare "I" or "A" in prose is far more likely to be a pronoun or an article
# than an answer, so they only count when punctuation marks them as an option
# or when they sit at one end of the response.
WORD_LETTERS = frozenset("AI")

# "the answer is (C)", "Answer: C", "final option - J".
_MARKED = re.compile(
    r"(?:ANSWER|OPTION|CHOICE)S?"
    r"(?:\s+(?:IS|ARE|WOULD\s+BE|SHOULD\s+BE|WOULD\s+HAVE\s+TO\s+BE))?"
    r"\s*[:=\-–—]?\s*[(\[{*\"']*([A-Z])(?![A-Z0-9])"
)

# "(C)", "[C]" - punctuation that only ever wraps an option letter.
_BRACKETED = re.compile(r"[(\[{]\s*([A-Z])\s*[)\]}]")

# "C." / "C)" / "**C**" at the start of a line or after whitespace.
_LABELLED = re.compile(r"(?:^|[\s*\"'])([A-Z])\s*[).:,\-*]")


def _matches(pattern: re.Pattern[str], text: str, letters: str) -> list[str]:
    return [match for match in pattern.findall(text) if match in letters]


def extract_choice(
    response: str,
    choices: list[str] | None = None,
    letters: str = DEFAULT_LETTERS,
) -> str | None:
    """Extract an option letter from a free-form response.

    `letters` is the option alphabet of the benchmark: A-D by default, A-J for
    ten-option suites such as MMLU-Pro. Explicit markers ("the answer is (C)")
    win over a letter that merely stands alone in the prose, and the tail of
    the response is preferred over its opening, since models tend to reason
    first and commit last.
    """
    text = " ".join(response.strip().split())
    if not text:
        return None

    upper = text.upper()
    letters = letters.upper()

    for pattern in (_MARKED, _BRACKETED, _LABELLED):
        found = _matches(pattern, upper, letters)
        if found:
            return found[-1]

    ambiguous = WORD_LETTERS if len(letters) > len(DEFAULT_LETTERS) else frozenset()

    def pick(indices: range) -> str | None:
        for i in indices:
            ch = upper[i]
            if ch not in letters:
                continue

            prev = upper[i - 1] if i > 0 else " "
            nxt = upper[i + 1] if i + 1 < len(upper) else " "
            if prev.isalnum() or nxt.isalnum():
                continue
            # A bare "A"/"I" mid-sentence is prose, not an answer.
            if ch in ambiguous and 0 < i < len(upper) - 1:
                continue
            return ch
        return None

    end_start = max(0, len(upper) - 48)
    choice = pick(range(len(upper) - 1, end_start - 1, -1))
    if choice:
        return choice

    start_end = min(len(upper), 48)
    choice = pick(range(start_end))
    if choice:
        return choice

    choice = pick(range(len(upper)))
    if choice:
        return choice

    if choices:
        lowered = text.lower()
        for letter, choice_text in zip(letters, choices, strict=False):
            if choice_text and choice_text.lower() in lowered:
                return letter

    return None
