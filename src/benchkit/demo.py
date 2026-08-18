"""Offline stand-in inference client used by ``benchkit --demo``.

Demo mode exists so the interface can be explored (and reviewed) without an
inference server: it answers prompts locally with a per-model skill level, so
scores, speeds and pass/fail streaks look like a real run.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import threading
import time
from collections.abc import Callable

from benchkit.benchmarks.base import Task
from benchkit.client import GenerationUpdate
from benchkit.engine import benchmark, prompt_for, tasks_for
from benchkit.perturbations import perturb_task

LETTERS = "ABCDEFGHIJ"

DEMO_MODELS = [
    {"name": "demo-nano:0.5b", "size": 0.4e9, "skill": 0.30, "speed": 240.0},
    {"name": "demo-mini:3b", "size": 2.1e9, "skill": 0.55, "speed": 130.0},
    {"name": "demo-pro:8b", "size": 5.2e9, "skill": 0.76, "speed": 68.0},
    {"name": "demo-max:14b", "size": 9.4e9, "skill": 0.91, "speed": 34.0},
]


def _dataset_path(key: str):
    module = importlib.import_module(type(benchmark(key)).__module__)
    return getattr(module, "DATASET", None)


def _canonical_solutions(key: str) -> dict[str, str]:
    """Reference solutions from a dataset file, when it ships any."""
    path = _dataset_path(key)
    if path is None:
        return {}

    solutions: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                solution = row.get("canonical_solution") or row.get("code")
                if solution:
                    solutions[row["task_id"]] = solution
    # ValueError covers both JSONDecodeError and UnicodeDecodeError.
    except (OSError, ValueError, KeyError):
        return {}
    return solutions


def _letters(task: Task) -> str:
    """Option letters a multiple-choice task actually offers."""
    choices = task.metadata.get("choices")
    if isinstance(choices, (list, tuple)) and choices:
        return LETTERS[: len(choices)]
    return LETTERS[:4]


def _wrong_letter(answer: str, letters: str) -> str:
    options = [letter for letter in letters if letter != answer]
    return random.choice(options) if options else "A"


_FILLER_SENTENCES = (
    "Here is a considered response to the request at hand.",
    "The plan below keeps every stated requirement in view.",
    "Each step is written plainly so nothing is left implicit.",
    "A short pause for structure makes the result easier to follow.",
    "The remaining detail fills out the picture without extra noise.",
    "This closing thought ties the pieces back together.",
)


def _ifeval_body(sentences: int, words: int | None) -> str:
    """Filler prose sized to the sentence and word budget of a task."""
    text: list[str] = []
    for index in range(max(sentences, 1)):
        text.append(_FILLER_SENTENCES[index % len(_FILLER_SENTENCES)])

    if words:
        while len(" ".join(text).split()) < words:
            text[-1] = text[-1].rstrip(".") + " and a little more detail follows."
    return " ".join(text)


def _ifeval_answer(task: Task) -> str:
    """Compose a response that satisfies the simpler IFEval constraints.

    Demo mode never calls a real model, so the strongest demo answer is one
    built directly from the instruction metadata. Constraints that need real
    language understanding (response language, letter frequency) are skipped,
    which keeps demo scores below a perfect run.
    """
    args = {
        instruction: dict(kwargs or {})
        for instruction, kwargs in zip(
            task.metadata["instruction_id_list"],
            task.metadata["kwargs"],
            strict=False,
        )
    }

    if "detectable_format:json_format" in args:
        return '{"answer": "demo response", "complete": true}'
    if "detectable_format:constrained_response" in args:
        return "My answer is maybe."

    sentence_spec = args.get("length_constraints:number_sentences")
    sentences = 4
    if sentence_spec:
        target = int(sentence_spec["num_sentences"])
        sentences = (
            target if sentence_spec["relation"] == "at least" else max(target - 1, 1)
        )

    word_spec = args.get("length_constraints:number_words")
    words = None
    if word_spec and word_spec["relation"] == "at least":
        words = int(word_spec["num_words"])
    elif word_spec:
        sentences = min(sentences, 2)

    body = _ifeval_body(sentences, words)

    keywords = args.get("keywords:existence", {}).get("keywords", [])
    if keywords:
        body += " Keywords in play: " + " ".join(keywords) + "."

    frequency = args.get("keywords:frequency")
    if frequency and frequency["relation"] == "at least":
        repeats = [frequency["keyword"]] * int(frequency["frequency"])
        body += " " + " ".join(repeats) + "."

    placeholders = args.get("detectable_content:number_placeholders")
    if placeholders:
        count = int(placeholders["num_placeholders"])
        body += " " + " ".join(f"[field {i + 1}]" for i in range(count))

    highlights = args.get("detectable_format:number_highlighted_sections")
    if highlights:
        count = int(highlights["num_highlights"])
        body += "\n" + "\n".join(f"*highlight {i + 1}*" for i in range(count))

    bullets = args.get("detectable_format:number_bullet_lists")
    if bullets:
        count = int(bullets["num_bullets"])
        body += "\n" + "\n".join(f"* point {i + 1}" for i in range(count))

    sections = args.get("detectable_format:multiple_sections")
    if sections:
        spliter = sections["section_spliter"]
        count = int(sections["num_sections"])
        body = "\n".join(f"{spliter} {i + 1}\n{body}" for i in range(count))

    paragraphs = args.get("length_constraints:number_paragraphs")
    if paragraphs:
        count = int(paragraphs["num_paragraphs"])
        body = "\n***\n".join([body] * count)

    if "detectable_format:title" in args:
        body = f"<<Demo Response>>\n{body}"

    if "combination:repeat_prompt" in args:
        prompt = args["combination:repeat_prompt"].get("prompt_to_repeat", task.prompt)
        body = f"{prompt}\n{body}"

    if "combination:two_responses" in args:
        body = f"{body}\n******\n{body}"

    postscript = args.get("detectable_content:postscript")
    if postscript:
        body += f"\n\n{postscript['postscript_marker']} That is the whole answer."

    end_phrase = args.get("startend:end_checker")
    if end_phrase:
        body += f" {end_phrase['end_phrase']}"

    forbidden = args.get("keywords:forbidden_words", {}).get("forbidden_words", [])
    for word in forbidden:
        body = body.replace(word, "…").replace(word.capitalize(), "…")

    if "punctuation:no_comma" in args:
        body = body.replace(",", "")

    if "change_case:english_capital" in args:
        body = body.upper()
    elif "change_case:english_lowercase" in args:
        body = body.lower()

    if "startend:quotation" in args:
        body = f'"{body}"'

    return body


def _good_answer(task: Task, solution: str | None) -> str:
    metadata = task.metadata
    if solution is not None:
        return solution

    if "instruction_id_list" in metadata:
        return _ifeval_answer(task)

    answers = metadata.get("answers")
    if isinstance(answers, (list, tuple)):
        return ", ".join(str(answer) for answer in answers)

    answer = metadata.get("answer")
    if isinstance(answer, bool):
        return "Yes" if answer else "No"
    if isinstance(answer, str) and answer.lower() in {"yes", "no"}:
        return answer.capitalize()
    if isinstance(answer, str) and answer.upper() in _letters(task):
        return f"The answer is {answer.upper()}."
    if answer is not None:
        return f"Working through it step by step.\n\n#### {answer}"
    return "A"


def _bad_answer(task: Task, has_solution: bool) -> str:
    metadata = task.metadata
    if has_solution or "test" in metadata:
        return "    raise NotImplementedError"

    if "instruction_id_list" in metadata:
        return "Sure, here is a quick answer that ignores the formatting rules."

    if isinstance(metadata.get("answers"), (list, tuple)):
        return "NOT_FOUND"

    answer = metadata.get("answer")
    if isinstance(answer, bool):
        return "No" if answer else "Yes"
    if isinstance(answer, str) and answer.lower() in {"yes", "no"}:
        return "No" if answer.lower() == "yes" else "Yes"
    if isinstance(answer, str) and answer.upper() in _letters(task):
        return f"The answer is {_wrong_letter(answer.upper(), _letters(task))}."
    if answer is not None:
        return "Let me compute that.\n\n#### 42"
    return "B"


class DemoClient:
    """Mimics :class:`benchkit.client.InferenceClient` without a server."""

    host = "demo://offline"
    label = "Demo mode"
    simulated_timing = True
    provider = "demo"
    timeout = 0.0

    def __init__(self, speed: float = 1.0) -> None:
        self.speed = max(speed, 0.01)
        self._answers: dict[str, tuple[Task, str | None]] = {}
        self._primed: set[tuple[str, str | None, int]] = set()

    def prime(
        self,
        benchmark_keys: list[str],
        perturbations: list[tuple[str, int]] | None = None,
    ) -> None:
        """Index prompts for the benchmarks about to run."""
        specs: list[tuple[str | None, int]] = [(None, 42)]
        specs.extend(perturbations or [])
        for key in benchmark_keys:
            bench = benchmark(key)
            solutions = _canonical_solutions(key)
            selected_tasks = tasks_for(key)
            variants = getattr(bench, "variants", None)
            select = getattr(bench, "tasks_for_variant", None)
            if callable(variants) and callable(select):
                selected_tasks = [
                    task
                    for variant in variants(self, DEMO_MODELS[0]["name"])
                    for task in select(selected_tasks, variant)
                ]
            for perturbation, seed in specs:
                cache_key = (key, perturbation, seed)
                if cache_key in self._primed:
                    continue
                for task in selected_tasks:
                    case = perturb_task(key, task, perturbation, seed)
                    prompt = prompt_for(
                        bench,
                        case.prompt_task,
                        self,
                        DEMO_MODELS[0]["name"],
                    )
                    solution = solutions.get(task.id) or task.metadata.get(
                        "canonical_solution"
                    )
                    self._answers[prompt] = (case.evaluation_task, solution)
                self._primed.add(cache_key)

    def list_models(self) -> list[dict]:
        return [
            {
                "name": model["name"],
                "size": model["size"],
                "status": "demo",
                "context_length": self.context_length(model["name"]),
            }
            for model in DEMO_MODELS
        ]

    def max_parallel_requests(self, model: str) -> int:
        """Keep the deterministic offline simulation single-threaded."""
        return 1

    def context_length(self, _model: str) -> int:
        """Keep the offline demo representative without building huge prompts."""
        return 8192

    def unload_model(self, model: str) -> None:
        time.sleep(0.05 / self.speed)

    def force_unload_model(self, model: str) -> None:
        self.unload_model(model)

    def _profile(self, model: str) -> dict:
        for entry in DEMO_MODELS:
            if entry["name"] == model:
                return entry
        return DEMO_MODELS[-1]

    def generate(
        self,
        model: str,
        prompt: str,
        on_progress: Callable[[GenerationUpdate], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        profile = self._profile(model)
        entry = self._answers.get(prompt)

        # Deterministic per model+prompt so repeat runs stay comparable.
        digest = hashlib.sha256(f"{model}{prompt}".encode()).hexdigest()
        roll = int(digest[:8], 16) / 0xFFFFFFFF
        random.seed(digest)

        if entry is None:
            response = "A"
        else:
            task, solution = entry
            correct = roll < profile["skill"]
            response = (
                _good_answer(task, solution)
                if correct
                else _bad_answer(task, solution is not None)
            )

        healthy_thinking = (
            "I will identify the requested output, check the relevant facts, "
            "compare the available choices, and then return the concise final "
            "answer. The evidence points consistently toward one option. "
        )
        looping = roll > profile["skill"] + 0.12
        thinking = (
            (
                "I should reconsider this carefully because the previous path "
                "may be wrong, so I will inspect the same evidence once more. "
            )
            * 14
            if looping
            else healthy_thinking
        )

        tokens = max(40 + int(roll * 260), len((thinking + response).split()))
        tok_s = profile["speed"] * (0.85 + roll * 0.3)
        elapsed = tokens / tok_s
        emitted_thinking = ""

        def cancelled_result() -> dict:
            return {
                "thinking": emitted_thinking,
                "response": "",
                "trace_status": "observed" if emitted_thinking else "unavailable",
                "tok_s": 0.0,
                "eval_count": 0,
                "eval_duration_ns": 0,
                "response_time_s": 0.0,
                "done_reason": "cancelled",
                "timed_out": False,
                "cancelled": True,
            }

        if cancel_event is not None and cancel_event.is_set():
            return cancelled_result()
        if on_progress is not None:
            chunks = [
                thinking[index : index + 80] for index in range(0, len(thinking), 80)
            ]
            for index, chunk in enumerate(chunks, start=1):
                time.sleep(min(0.012 / self.speed, 0.03))
                if cancel_event is not None and cancel_event.is_set():
                    return cancelled_result()
                emitted_thinking += chunk
                on_progress(
                    GenerationUpdate(
                        thinking=chunk,
                        elapsed_s=elapsed * index / (len(chunks) + 1),
                        reasoning_channel_seen=True,
                    )
                )
            on_progress(
                GenerationUpdate(
                    response=response,
                    elapsed_s=elapsed,
                    reasoning_channel_seen=True,
                )
            )
            on_progress(
                GenerationUpdate(
                    elapsed_s=elapsed,
                    reasoning_channel_seen=True,
                    done=True,
                )
            )
        else:
            time.sleep(min(elapsed / (40 * self.speed), 0.35))
            if cancel_event is not None and cancel_event.is_set():
                return cancelled_result()

        return {
            "thinking": thinking,
            "response": response,
            "trace_status": "observed",
            "tok_s": tok_s,
            "eval_count": tokens,
            "eval_duration_ns": int(elapsed * 1e9),
            "response_time_s": elapsed,
            "done_reason": "stop",
            "timed_out": False,
            "cancelled": False,
        }


def is_demo_client(client: object) -> bool:
    return isinstance(client, DemoClient)
