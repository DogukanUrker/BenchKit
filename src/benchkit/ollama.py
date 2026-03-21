"""Ollama API client."""

import os

import httpx

DEFAULT_HOST = "http://localhost:11434"


def get_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


def list_models(host: str) -> list[dict]:
    r = httpx.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    models = r.json().get("models", [])
    return sorted(models, key=lambda m: m.get("size", 0))


def unload_model(host: str, model: str) -> None:
    """Evict model from Ollama VRAM by setting keep_alive to 0."""
    httpx.post(
        f"{host}/api/generate",
        json={"model": model, "keep_alive": 0},
        timeout=30,
    )


def generate(host: str, model: str, prompt: str) -> dict:
    r = httpx.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "stop": ["\ndef ", "\nclass ", "\nif __name__"],
            },
        },
        timeout=None,
    )
    r.raise_for_status()
    data = r.json()

    response = data.get("response", "")
    thinking = data.get("thinking", "")
    done_reason = data.get("done_reason", "")

    # Fallback for reasoning models that leave response empty (e.g. stuck in a
    # think loop and truncated). Strip <think> tags from the thinking field and
    # use whatever remains — downstream _extract_code will do the rest.
    if not response.strip() and thinking:
        text = thinking
        if "<think>" in text:
            start = text.index("<think>")
            if "</think>" in text:
                end = text.index("</think>") + len("</think>")
                text = text[:start] + text[end:]
            else:
                text = text[:start]
        response = text.strip()

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    total_duration_ns = data.get("total_duration", 0)
    tok_s = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
    response_time_s = total_duration_ns / 1e9

    return {
        "response": response,
        "tok_s": tok_s,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "response_time_s": response_time_s,
        "done_reason": done_reason,
    }
