"""Small shared helpers for benchmark evaluation."""


def strip_think_tags(text: str) -> str:
    """Remove model reasoning blocks wrapped in <think> tags."""
    cleaned = text

    while "<think>" in cleaned:
        start = cleaned.index("<think>")
        end = cleaned.find("</think>", start)
        if end == -1:
            cleaned = cleaned[:start]
            break
        cleaned = cleaned[:start] + cleaned[end + len("</think>") :]

    return cleaned.lstrip("\n").rstrip()
