"""Deterministic synthetic task generation for the RULER benchmark."""

from __future__ import annotations

import json
import math
import random
import string
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.leakage import assert_candidate_parity

# Providers use both binary and decimal interpretations of "128K". The final
# target is 128,000 so a server advertising a 128,000-token served window can
# run the bucket instead of being rejected for falling 3,072 tokens short.
CONTEXT_BUCKETS = (4096, 8192, 16384, 32768, 65536, 128000)
DEFAULT_SAMPLES_PER_KIND = 500
PRACTICAL_SAMPLES_PER_KIND = 3
TASK_KINDS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "variable_tracking",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)


TASKS_PER_BUCKET = DEFAULT_SAMPLES_PER_KIND * len(TASK_KINDS)

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
    kind_offset = (
        53
        if kind == "variable_tracking"
        else sum((index + 1) * ord(char) for index, char in enumerate(kind))
    )
    return bucket * 1009 + sample * 97 + kind_offset


def _unique_variables(rng: random.Random, count: int) -> list[str]:
    values: list[str] = []
    while len(values) < count:
        value = "".join(rng.choices(string.ascii_uppercase, k=6))
        if value not in values:
            values.append(value)
    return values


def _random_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _random_word(rng: random.Random) -> str:
    return f"{''.join(rng.choices(string.ascii_lowercase, k=7))}-{''.join(rng.choices(string.ascii_lowercase, k=8))}"


def _niah_task(bucket: int, sample: int, kind: str) -> Task:
    seed = _seed(bucket, sample, kind)
    rng = random.Random(seed)
    config = {
        "niah_single_1": ("noise", "word", "number", 1, 1, 1),
        "niah_single_2": ("essay", "word", "number", 1, 1, 1),
        "niah_single_3": ("essay", "word", "uuid", 1, 1, 1),
        "niah_multikey_1": ("essay", "word", "number", 4, 1, 1),
        "niah_multikey_2": ("needle", "word", "number", 1, 1, 1),
        "niah_multikey_3": ("needle", "uuid", "uuid", 1, 1, 1),
        "niah_multivalue": ("essay", "word", "number", 1, 4, 1),
        "niah_multiquery": ("essay", "word", "number", 4, 1, 4),
    }[kind]
    haystack, key_type, value_type, key_count, value_count, query_count = config

    def generate(requested_type: str) -> str:
        if requested_type == "number":
            return str(rng.randint(1_000_000, 9_999_999))
        if requested_type == "uuid":
            return _random_uuid(rng)
        return _random_word(rng)

    keys = [generate(key_type) for _ in range(key_count)]
    values = [[generate(value_type) for _ in range(value_count)] for _ in keys]
    query_indices = rng.sample(range(key_count), query_count)
    probe_rng = random.Random(seed ^ 0xD157AC7)
    probe_pairs = [_distractor_pair(probe_rng, key_type, value_type) for _ in range(16)]
    assert_candidate_parity(
        target_keys=keys,
        distractor_keys=[key for key, _ in probe_pairs],
        target_values=[value for group in values for value in group],
        distractor_values=[value for _, value in probe_pairs],
    )
    label = context_label(bucket)
    return Task(
        id=f"RULER/{label}/{kind}/{sample:03d}",
        prompt="",
        metadata={
            "variant": label,
            "context_tokens": bucket,
            "task_type": kind,
            "seed": seed,
            "haystack": haystack,
            "key_type": key_type,
            "value_type": value_type,
            "keys": keys,
            "values": values,
            "query": [keys[index] for index in query_indices],
            "answers": [value for index in query_indices for value in values[index]],
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


def generate_tasks(*, samples_per_kind: int = DEFAULT_SAMPLES_PER_KIND) -> list[Task]:
    """Create lightweight task specifications for every context bucket."""
    if samples_per_kind < 1:
        raise ValueError("RULER samples per task must be positive")
    tasks: list[Task] = []
    for bucket in CONTEXT_BUCKETS:
        # Keep each sample's 13 task families together so small CLI slices are
        # representative instead of containing only the first NIAH family.
        for sample in range(samples_per_kind):
            for kind in TASK_KINDS:
                tasks.append(
                    _variable_task(bucket, sample)
                    if kind == "variable_tracking"
                    else _niah_task(bucket, sample, kind)
                    if kind.startswith("niah_")
                    else _aggregation_task(bucket, sample, kind)
                )
    return tasks


def _aggregation_task(bucket: int, sample: int, kind: str) -> Task:
    seed = _seed(bucket, sample, kind)
    answers: list[str] = []
    metadata = {
        "variant": context_label(bucket),
        "context_tokens": bucket,
        "task_type": kind,
        "seed": seed,
        "answers": answers,
    }
    if kind in {"cwe", "fwe"}:
        rng = random.Random(seed)
        vocabulary = [
            "".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(2000)
        ]
        metadata["vocabulary"] = list(dict.fromkeys(vocabulary))
        metadata["answers"] = metadata["vocabulary"][: 10 if kind == "cwe" else 3]
    elif kind in {"qa_1", "qa_2"}:
        metadata["qa_index"] = sample
    return Task(
        id=f"RULER/{context_label(bucket)}/{kind}/{sample:03d}",
        prompt="",
        metadata=metadata,
    )


_QA_URLS = {
    "qa_1": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "qa_2": "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
}


def _qa_path(kind: str) -> Path:
    root = Path.home() / ".cache" / "benchkit" / "ruler"
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("squad.json" if kind == "qa_1" else "hotpotqa.json")
    if not path.exists():
        try:
            urllib.request.urlretrieve(_QA_URLS[kind], path)
        except OSError as exc:
            raise RuntimeError(
                f"RULER {kind} requires {path}; download failed: {exc}"
            ) from exc
    return path


def _essay_words() -> list[str]:
    root = Path.home() / ".cache" / "benchkit" / "ruler"
    path = root / "PaulGrahamEssays.json"
    if not path.exists():
        raise RuntimeError(
            "essay-backed RULER tasks require PaulGrahamEssays.json in "
            f"{root}; generate it with NVIDIA/RULER's download_paulgraham_essay.py"
        )
    text = str(json.loads(path.read_text(encoding="utf-8"))["text"])
    return text.split()


def _qa_records(kind: str) -> tuple[list[dict], list[str]]:
    raw = json.loads(_qa_path(kind).read_text(encoding="utf-8"))
    if kind == "qa_1":
        docs = sorted({p["context"] for d in raw["data"] for p in d["paragraphs"]})
        index = {doc: i for i, doc in enumerate(docs)}
        records = [
            {
                "query": qa["question"],
                "answers": [a["text"] for a in qa["answers"]],
                "docs": [index[p["context"]]],
            }
            for d in raw["data"]
            for p in d["paragraphs"]
            for qa in p["qas"]
            if not qa.get("is_impossible")
        ]
    else:
        docs = sorted(
            {
                f"{title}\n{''.join(text)}"
                for row in raw
                for title, text in row["context"]
            }
        )
        index = {doc: i for i, doc in enumerate(docs)}
        records = [
            {
                "query": row["question"],
                "answers": [row["answer"]],
                "docs": [
                    index[f"{title}\n{''.join(text)}"] for title, text in row["context"]
                ],
            }
            for row in raw
        ]
    return records, docs


def _render_aggregation(task: Task, filler_units: int) -> str:
    metadata = task.metadata
    kind = metadata["task_type"]
    vocab = list(metadata["vocabulary"])
    rng = random.Random(int(metadata["seed"]))
    if kind == "cwe":
        common = vocab[:10]
        uncommon = vocab[10 : 10 + max(1, filler_units)]
        words = common * 30 + uncommon * 3
        rng.shuffle(words)
        context = " ".join(f"{i + 1}. {word}" for i, word in enumerate(words))
        return f"Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?"
    counts = [
        max(1, int(max(10, filler_units) * ((i + 1) ** -2.0)))
        for i in range(len(vocab))
    ]
    words = [
        word for word, count in zip(vocab, counts, strict=True) for _ in range(count)
    ]
    rng.shuffle(words)
    return (
        "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. "
        + " ".join(words)
        + "\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?"
    )


def _render_qa(task: Task, filler_units: int) -> str:
    kind = str(task.metadata["task_type"])
    records, docs = _qa_records(kind)
    record = records[int(task.metadata["qa_index"]) % len(records)]
    task.metadata["answers"] = list(record["answers"])
    selected = list(record["docs"])
    rng = random.Random(int(task.metadata["seed"]))
    available = [i for i in range(len(docs)) if i not in selected]
    selected.extend(
        rng.sample(available, min(len(available), max(0, filler_units - len(selected))))
    )
    rendered = [docs[i] for i in selected]
    rng.shuffle(rendered)
    context = "\n\n".join(f"Document {i + 1}:\n{doc}" for i, doc in enumerate(rendered))
    return f"Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {record['query']}"


def _needle_line(key: str, value: str) -> str:
    return f"One of the special magic numbers for {key} is: {value}."


def _distractor_pair(
    rng: random.Random, key_type: str, value_type: str
) -> tuple[str, str]:
    key = _random_uuid(rng) if key_type == "uuid" else _random_word(rng)
    value = (
        _random_uuid(rng)
        if value_type == "uuid"
        else str(rng.randint(1_000_000, 9_999_999))
    )
    return key, value


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
    rng = random.Random(seed ^ 0xD157AC7)
    pairs = (
        [
            _distractor_pair(rng, metadata["key_type"], metadata["value_type"])
            for _ in range(filler_units)
        ]
        if metadata["haystack"] == "needle"
        else []
    )
    if pairs:
        filler = [_needle_line(key, value) for key, value in pairs]
    elif metadata["haystack"] == "essay":
        words = _essay_words()
        filler = [words[index % len(words)] for index in range(filler_units)]
    else:
        filler = [_VARIABLE_NOISE] * filler_units
    needles = [
        _needle_line(key, value)
        for key, values in zip(metadata["keys"], metadata["values"], strict=True)
        for value in values
    ]
    assert_candidate_parity(
        target_keys=list(metadata["keys"]),
        distractor_keys=[p[0] for p in pairs],
        target_values=list(metadata["answers"]),
        distractor_values=[p[1] for p in pairs],
    )
    context = "\n".join(_insert_at_positions(filler, needles, seed ^ 0xA51CE))
    return (
        "Some special magic numbers are hidden within the following text. "
        "Memorize the records because you will be asked for one key afterward.\n\n"
        f"{context}\n\n"
        "Question: What is the special magic number for "
        f"{', and '.join(metadata['query'])}?\n"
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
    if str(kind).startswith("niah_"):
        return _render_niah(task, max(0, filler_units))
    if kind == "variable_tracking":
        return _render_variable_tracking(task, max(0, filler_units))
    if kind in {"cwe", "fwe"}:
        return _render_aggregation(task, max(0, filler_units))
    if kind in {"qa_1", "qa_2"}:
        return _render_qa(task, max(1, filler_units))
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
