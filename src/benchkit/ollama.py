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


def generate(host: str, model: str, prompt: str, timeout: float = 300.0) -> dict:
    r = httpx.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 1024,
                "stop": ["\ndef ", "\nclass ", "\nif __name__"],
            },
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    tok_s = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0

    return {
        "response": data.get("response", ""),
        "tok_s": tok_s,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
    }
