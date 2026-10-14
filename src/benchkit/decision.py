"""Native decision models served through llama.cpp's ``/v1/systemone``.

A decision model answers typed questions in one forward pass and returns a
probability for every option instead of generated text. BenchKit routes every
model whose ``/v1/models`` entry lists the ``decisions`` output modality here,
with no flag to set. Benchmarks opt in through a ``decision_instructions``
attribute. The most likely option is handed back to the engine as an ordinary
response (an option letter, a benchmark's own option key, or ``yes``/``no``),
so each benchmark's own ``evaluate()`` scores it unchanged and perturbations
and reports keep working.
"""

from __future__ import annotations

import json
import threading
import time

from benchkit.benchmarks import REGISTRY
from benchkit.benchmarks.base import Task
from benchkit.client import image_data_uri
from benchkit.perturbations import VISIBLE_CHOICE_LIMITS

QUESTION = "answer"
ECE_BINS = 10


def supports(benchmark_key: str) -> bool:
    """Return whether a decision model can answer a benchmark."""
    return bool(getattr(REGISTRY.get(benchmark_key), "decision_instructions", ""))


def supported_benchmarks() -> list[str]:
    return [key for key in REGISTRY if supports(key)]


def request_for(bench: object, task: Task) -> dict:
    """Build the ``/v1/systemone`` body for one task.

    Multiple-choice tasks become a ``choice`` question keyed by option letter,
    so the chosen key is directly the answer letter. A benchmark whose answer is
    a label rather than a letter (an intent router) supplies its own keys
    through ``decision_criteria(task)``; the chosen key is then the label.
    Tasks without either are yes/no questions (``noul``).

    Image tasks list their local files under ``images``; ``Decider`` swaps
    them for data URIs at send time, so the recorded prompt stays small.
    """
    instructions = str(bench.decision_instructions)
    choices = task.metadata.get("choices")
    criteria = getattr(bench, "decision_criteria", None)
    if callable(criteria):
        question = {
            "type": "choice",
            "instructions": instructions,
            "criteria": {str(key): str(text) for key, text in criteria(task).items()},
        }
    elif choices:
        name = str(getattr(bench, "name", ""))
        visible = choices[: VISIBLE_CHOICE_LIMITS.get(name, len(choices))]
        question = {
            "type": "choice",
            "instructions": instructions,
            "criteria": {
                chr(ord("A") + index): str(choice)
                for index, choice in enumerate(visible)
            },
        }
    else:
        question = {"type": "noul", "instructions": instructions}
    passage = task.metadata.get("passage")
    state = f"{passage}\n\nQuestion: {task.prompt}" if passage else task.prompt
    request = {"state": state, "questions": {QUESTION: question}}
    images = task.metadata.get("images")
    if images:
        request["images"] = [str(path) for path in images]
    return request


def render_request(bench: object, task: Task) -> str:
    """Return the request body as the task's recorded prompt."""
    return json.dumps(request_for(bench, task), indent=2, ensure_ascii=False)


def probabilities(answer: dict) -> dict[str, float]:
    """Normalize a ``choice`` or ``noul`` answer to option probabilities."""
    if "noul" in answer:
        yes = float(answer["noul"])
        return {"yes": yes, "no": 1.0 - yes}
    return {str(key): float(value) for key, value in answer["probabilities"].items()}


class Decider:
    """Generation-shaped adapter, so the engine runs decisions like any call.

    ``prompt`` is the JSON body from ``render_request``.
    """

    def __init__(self, client: object) -> None:
        self.client = client

    def generate(
        self,
        model: str,
        prompt: str,
        on_progress: object = None,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        started = time.perf_counter()
        request = json.loads(prompt)
        if request.get("images"):
            request["images"] = [image_data_uri(path) for path in request["images"]]
        payload = self.client.decide(model, request, cancel_event)
        elapsed = time.perf_counter() - started
        probs = probabilities(payload["answers"][QUESTION])
        usage = payload.get("usage") or {}
        return {
            "thinking": "",
            "response": max(probs, key=probs.__getitem__),
            "tok_s": 0.0,
            "eval_count": 0,
            "eval_duration_ns": 0,
            "response_time_s": elapsed,
            "done_reason": "decision",
            "trace_status": "unavailable",
            "input_tokens": int(usage.get("input_tokens") or 0),
            "decision": {"probabilities": probs, "latency_ms": elapsed * 1000},
        }


def calibration(decision: dict | None, answer: object) -> dict:
    """Attach per-task calibration to a decision's probabilities."""
    if not decision:
        return {}
    probs = decision["probabilities"]
    target = str(answer).strip()
    confidence = max(probs.values())
    return {
        "latency_ms": round(float(decision.get("latency_ms") or 0.0), 2),
        "probabilities": {key: round(value, 4) for key, value in probs.items()},
        "confidence": round(confidence, 4),
        "p_correct": round(probs.get(target, 0.0), 4),
        "correct": max(probs, key=probs.__getitem__) == target,
        # Multi-class Brier score: 0 is perfect, 2 is confidently wrong.
        "brier": round(
            sum((value - (key == target)) ** 2 for key, value in probs.items()), 4
        ),
    }


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, so the value is one that was measured."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def summary(decisions: list[dict], wall_time_s: float = 0.0) -> dict:
    """Aggregate calibration and speed for one job; empty without decisions.

    Decisions per second is the job's real rate: answered decisions over wall
    time, so it already includes any request concurrency.
    """
    rows = [row for row in decisions if row]
    if not rows:
        return {}
    total = len(rows)
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    bins: list[list[dict]] = [[] for _ in range(ECE_BINS)]
    for row in rows:
        bins[min(int(row["confidence"] * ECE_BINS), ECE_BINS - 1)].append(row)
    ece = sum(
        len(group)
        / total
        * abs(
            sum(row["correct"] for row in group) / len(group)
            - sum(row["confidence"] for row in group) / len(group)
        )
        for group in bins
        if group
    )
    return {
        "decision_brier": round(sum(row["brier"] for row in rows) / total, 4),
        "decision_ece": round(ece, 4),
        "decision_confidence": round(
            sum(row["confidence"] for row in rows) / total * 100, 1
        ),
        "decision_p_correct": round(
            sum(row["p_correct"] for row in rows) / total * 100, 1
        ),
        "decisions_per_s": round(total / wall_time_s, 1) if wall_time_s > 0 else 0.0,
        "decision_latency_p50_ms": round(_percentile(latencies, 0.50), 1),
        "decision_latency_p95_ms": round(_percentile(latencies, 0.95), 1),
    }
