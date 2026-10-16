"""Small shared helpers for benchmark evaluation."""


def strip_think_tags(text: str) -> str:
    """Remove model reasoning blocks wrapped in <think> tags.

    Some chat templates open ``<think>`` in the prompt itself, so the response
    carries only the closing tag. A ``</think>`` with no ``<think>`` before it
    marks everything up to and including it as reasoning.
    """
    cleaned = text

    close = cleaned.find("</think>")
    if close != -1 and "<think>" not in cleaned[:close]:
        cleaned = cleaned[close + len("</think>") :]

    while "<think>" in cleaned:
        start = cleaned.index("<think>")
        end = cleaned.find("</think>", start)
        if end == -1:
            cleaned = cleaned[:start]
            break
        cleaned = cleaned[:start] + cleaned[end + len("</think>") :]

    return cleaned.lstrip("\n").rstrip()
