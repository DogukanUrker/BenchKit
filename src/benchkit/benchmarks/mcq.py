"""Helpers for messy multiple-choice model outputs."""


def extract_choice(response: str, choices: list[str] | None = None) -> str | None:
    """Extract A/B/C/D from a free-form response."""
    text = " ".join(response.strip().split())
    if not text:
        return None

    upper = text.upper()

    def pick(indices: range) -> str | None:
        for i in indices:
            ch = upper[i]
            if ch not in "ABCD":
                continue

            prev = upper[i - 1] if i > 0 else " "
            nxt = upper[i + 1] if i + 1 < len(upper) else " "
            if not prev.isalnum() and not nxt.isalnum():
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

    for i, ch in enumerate(upper):
        if ch not in "ABCD":
            continue
        prev = upper[i - 1] if i > 0 else " "
        nxt = upper[i + 1] if i + 1 < len(upper) else " "
        if not prev.isalnum() and not nxt.isalnum():
            return ch

    if choices:
        lowered = text.lower()
        for letter, choice in zip("ABCD", choices):
            if choice and choice.lower() in lowered:
                return letter

    return None
