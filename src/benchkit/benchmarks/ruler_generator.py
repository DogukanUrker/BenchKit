"""Deterministic synthetic task generation for the RULER benchmark."""

from __future__ import annotations

import math
import random
import string
from collections.abc import Callable

from benchkit.benchmarks.base import Task

# Providers use both binary and decimal interpretations of "128K". The final
# target is 128,000 so a server advertising a 128,000-token served window can
# run the bucket instead of being rejected for falling 3,072 tokens short.
CONTEXT_BUCKETS = (4096, 8192, 16384, 32768, 65536, 128000)
SAMPLES_PER_KIND = 10
TASK_KINDS = ("niah_multikey", "variable_tracking")
TASKS_PER_BUCKET = SAMPLES_PER_KIND * len(TASK_KINDS)

# RULER lengths include the expected completion. Keep enough room for a short
# answer and the server's chat template, neither of which is visible to the
# benchmark-side tokenizer call.
OUTPUT_RESERVE_TOKENS = 256
FALLBACK_CHARS_PER_TOKEN = 4
FIT_TOLERANCE = 0.01

_VARIABLE_NOISE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)


def context_label(tokens: int) -> str:
    """Return the compact label used for a context bucket."""
    if tokens == 128000:
        return "128k"
    if tokens % 1024 == 0:
        return f"{tokens // 1024}k"
    if tokens % 1000 == 0:
        return f"{tokens // 1000}k"
    return f"{tokens:,}"


def _seed(bucket: int, sample: int, kind: str) -> int:
    kind_offset = 17 if kind == "niah_multikey" else 53
    return bucket * 1009 + sample * 97 + kind_offset


def _unique_variables(rng: random.Random, count: int) -> list[str]:
    values: list[str] = []
    while len(values) < count:
        value = "".join(rng.choices(string.ascii_uppercase, k=6))
        if value not in values:
            values.append(value)
    return values


def _niah_task(bucket: int, sample: int) -> Task:
    seed = _seed(bucket, sample, "niah_multikey")
    rng = random.Random(seed)
    keys = [f"target-{rng.getrandbits(40):010x}" for _ in range(4)]
    values = [str(rng.randint(1_000_000, 9_999_999)) for _ in range(4)]
    query_index = rng.randrange(len(keys))
    label = context_label(bucket)
    return Task(
        id=f"RULER/{label}/niah_multikey/{sample:02d}",
        prompt="",
        metadata={
            "variant": label,
            "context_tokens": bucket,
            "task_type": "niah_multikey",
            "seed": seed,
            "keys": keys,
            "values": values,
            "query": keys[query_index],
            "answers": [values[query_index]],
        },
    )


def _variable_task(bucket: int, sample: int) -> Task:
    seed = _seed(bucket, sample, "variable_tracking")
    rng = random.Random(seed)
    variables = _unique_variables(rng, 5)
    value = str(rng.randint(10_000, 99_999))
    label = context_label(bucket)
    return Task(
        id=f"RULER/{label}/variable_tracking/{sample:02d}",
        prompt="",
        metadata={
            "variant": label,
            "context_tokens": bucket,
            "task_type": "variable_tracking",
            "seed": seed,
            "variables": variables,
            "query": value,
            "answers": variables,
        },
    )


def generate_tasks() -> list[Task]:
    """Create lightweight task specifications for every context bucket."""
    tasks: list[Task] = []
    for bucket in CONTEXT_BUCKETS:
        for sample in range(SAMPLES_PER_KIND):
            tasks.append(_niah_task(bucket, sample))
            tasks.append(_variable_task(bucket, sample))
    return tasks


def _needle_line(key: str, value: str) -> str:
    return f"One of the special magic numbers for {key} is: {value}."


def _distractor_line(seed: int, index: int) -> str:
    key_number = (seed * 2_654_435_761 + index * 2_246_822_519) & 0xFFFFFFFF
    value = 1_000_000 + ((seed * 7919 + index * 104729) % 9_000_000)
    return _needle_line(f"archive-{key_number:08x}", str(value))


def _insert_at_positions(filler: list[str], inserts: list[str], seed: int) -> list[str]:
    """Distribute inserts through filler while preserving insert order."""
    rng = random.Random(seed)
    positions = sorted(rng.randrange(len(filler) + 1) for _ in inserts)
    by_position: dict[int, list[str]] = {}
    for position, value in zip(positions, inserts, strict=True):
        by_position.setdefault(position, []).append(value)

    output: list[str] = []
    for index in range(len(filler) + 1):
        output.extend(by_position.get(index, ()))
        if index < len(filler):
            output.append(filler[index])
    return output


def _render_niah(task: Task, filler_units: int) -> str:
    metadata = task.metadata
    seed = int(metadata["seed"])
    filler = [_distractor_line(seed, index) for index in range(filler_units)]
    needles = [
        _needle_line(key, value)
        for key, value in zip(metadata["keys"], metadata["values"], strict=True)
    ]
    context = "\n".join(_insert_at_positions(filler, needles, seed ^ 0xA51CE))
    return (
        "Some special magic numbers are hidden within the following text. "
        "Memorize the records because you will be asked for one key afterward.\n\n"
        f"{context}\n\n"
        "Question: What is the special magic number for "
        f"{metadata['query']}?\n"
        "Answer with only the matching number."
    )


def _render_variable_tracking(task: Task, filler_units: int) -> str:
    metadata = task.metadata
    variables = list(metadata["variables"])
    query = str(metadata["query"])
    chain = [f"VAR {variables[0]} = {query}"] + [
        f"VAR {variables[index]} = VAR {variables[index - 1]}"
        for index in range(1, len(variables))
    ]
    filler = [_VARIABLE_NOISE] * filler_units
    context = "\n".join(
        _insert_at_positions(filler, chain, int(metadata["seed"]) ^ 0xC4A1)
    )
    return (
        "Memorize and track the variable assignments hidden in the text. "
        "Assignments are transitive: if VAR BBBBBB = VAR AAAAAA, both variables "
        "have the same value.\n\n"
        "Example:\n"
        "VAR QAZWSX = 12345\n"
        "VAR EDCVRF = VAR QAZWSX\n"
        "For value 12345 the variables are QAZWSX, EDCVRF.\n\n"
        f"{context}\n\n"
        "Question: Find all variables assigned the value "
        f"{query} in the text above.\n"
        "Answer with only the variable names, separated by commas."
    )


def render_prompt(task: Task, filler_units: int) -> str:
    """Render a task with the requested number of synthetic filler units."""
    kind = task.metadata["task_type"]
    if kind == "niah_multikey":
        return _render_niah(task, max(0, filler_units))
    if kind == "variable_tracking":
        return _render_variable_tracking(task, max(0, filler_units))
    raise ValueError(f"Unknown RULER task type: {kind}")


def _fallback_units(task: Task, target_tokens: int) -> int:
    base = render_prompt(task, 0)
    one = render_prompt(task, 1)
    chars_per_unit = max(1, len(one) - len(base))
    target_chars = target_tokens * FALLBACK_CHARS_PER_TOKEN
    return max(0, (target_chars - len(base)) // chars_per_unit)


def fit_prompt(
    task: Task,
    token_count: Callable[[str], int | None] | None = None,
    *,
    initial_units: int | None = None,
) -> tuple[str, int, int | None]:
    """Fit a generated prompt just below its context bucket.

    When the server exposes a tokenizer, a short secant-style calibration keeps
    the prompt within one percent of the target. Generic servers fall back to a
    conservative character-to-token estimate.
    """
    bucket = int(task.metadata["context_tokens"])
    target = max(1, bucket - OUTPUT_RESERVE_TOKENS)
    units = (
        max(0, initial_units)
        if initial_units is not None
        else _fallback_units(task, target)
    )
    prompt = render_prompt(task, units)
    if token_count is None:
        return prompt, units, None

    measured = token_count(prompt)
    if not isinstance(measured, int) or measured <= 0:
        return prompt, units, None

    base_tokens: int | None = None
    for _ in range(4):
        if measured <= target and measured >= target * (1 - FIT_TOLERANCE):
            break
        if base_tokens is None:
            base_tokens = token_count(render_prompt(task, 0))
            if not isinstance(base_tokens, int) or base_tokens < 0:
                base_tokens = 0

        variable_tokens = max(1, measured - base_tokens)
        tokens_per_unit = variable_tokens / max(units, 1)
        next_units = max(0, math.floor((target - base_tokens) / tokens_per_unit))
        if next_units == units:
            next_units = max(0, units - 1) if measured > target else units + 1
        units = next_units
        prompt = render_prompt(task, units)
        next_measured = token_count(prompt)
        if not isinstance(next_measured, int) or next_measured <= 0:
            return prompt, units, None
        measured = next_measured

    # Never knowingly send a prompt above the bucket. The linear adjustment is
    # normally exact; this final correction handles unusual tokenizer merges.
    for _ in range(2):
        if measured <= target or units == 0:
            break
        excess = measured - target
        tokens_per_unit = max(1.0, measured / max(units, 1))
        units = max(0, units - math.ceil(excess / tokens_per_unit) - 1)
        prompt = render_prompt(task, units)
        next_measured = token_count(prompt)
        if not isinstance(next_measured, int) or next_measured <= 0:
            return prompt, units, None
        measured = next_measured

    return prompt, units, measured
