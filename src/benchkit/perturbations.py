"""Deterministic, benchmark-aware input perturbations."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace

from benchkit.benchmarks.base import Task

CHOICE_ORDER = "choice-order"
PERTURBATIONS = (CHOICE_ORDER,)

# Every benchmark here stores visible options in ``metadata["choices"]`` and
# the correct option letter in ``metadata["answer"]``. TruthfulQA is the one
# exception to the all-choices-visible rule: its prompt intentionally shows at
# most A-D even though the source MC1 rows can carry more distractors.
CHOICE_ORDER_BENCHMARKS = frozenset(
    {
        "arc",
        "gpqa",
        "hellaswag",
        "mmlu",
        "mmlu-pro",
        "openbookqa",
        "piqa",
        "truthfulqa",
        "winogrande",
    }
)
_VISIBLE_CHOICE_LIMITS = {"truthfulqa": 4}


@dataclass(frozen=True)
class PerturbedCase:
    """Prompt and scoring tasks produced by one perturbation."""

    prompt_task: Task
    evaluation_task: Task
    details: dict


def perturbations_for(benchmark_key: str) -> tuple[str, ...]:
    """Return perturbations supported by one benchmark."""
    if benchmark_key in CHOICE_ORDER_BENCHMARKS:
        return (CHOICE_ORDER,)
    return ()


def supports_perturbation(benchmark_key: str, name: str) -> bool:
    """Return whether a benchmark opts into a named perturbation."""
    return name in perturbations_for(benchmark_key)


def _task_rng(benchmark_key: str, task_id: str, name: str, seed: int) -> random.Random:
    material = f"{seed}\0{benchmark_key}\0{task_id}\0{name}".encode()
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _choice_order(benchmark_key: str, task: Task, seed: int) -> PerturbedCase:
    raw_choices = task.metadata.get("choices")
    answer = str(task.metadata.get("answer", "")).upper()
    if not isinstance(raw_choices, (list, tuple)):
        raise ValueError(f"{benchmark_key}/{task.id} has no choice list")

    choices = list(raw_choices)
    visible_count = min(
        len(choices),
        _VISIBLE_CHOICE_LIMITS.get(benchmark_key, len(choices)),
    )
    if visible_count < 2:
        raise ValueError(f"{benchmark_key}/{task.id} needs at least two choices")

    answer_index = ord(answer) - ord("A") if len(answer) == 1 else -1
    if answer_index < 0 or answer_index >= visible_count:
        raise ValueError(
            f"{benchmark_key}/{task.id} answer {answer!r} is not a visible choice"
        )

    rng = _task_rng(benchmark_key, task.id, CHOICE_ORDER, seed)
    order = list(range(visible_count))
    rng.shuffle(order)

    # A choice-order run should exercise answer-position sensitivity on every
    # task. If the random permutation left the correct answer under its old
    # letter, swap it with a deterministic different position.
    new_answer_index = order.index(answer_index)
    if new_answer_index == answer_index:
        candidates = [index for index in range(visible_count) if index != answer_index]
        swap_index = rng.choice(candidates)
        order[answer_index], order[swap_index] = (
            order[swap_index],
            order[answer_index],
        )
        new_answer_index = order.index(answer_index)

    visible_choices = [choices[index] for index in order]
    perturbed_choices = visible_choices + choices[visible_count:]
    perturbed_answer = chr(ord("A") + new_answer_index)
    metadata = dict(task.metadata)
    metadata.update(choices=perturbed_choices, answer=perturbed_answer)
    perturbed_task = replace(task, metadata=metadata)
    details = {
        "name": CHOICE_ORDER,
        "seed": seed,
        # Original labels in their new A.. position. For example ["C", "A",
        # "B"] means original C is now A, original A is now B, and so on.
        "choice_order": [chr(ord("A") + index) for index in order],
        "original_answer": answer,
        "perturbed_answer": perturbed_answer,
    }
    return PerturbedCase(perturbed_task, perturbed_task, details)


def perturb_task(
    benchmark_key: str,
    task: Task,
    name: str | None,
    seed: int,
) -> PerturbedCase:
    """Copy and perturb a task without mutating the cached source task."""
    if name is None:
        return PerturbedCase(task, task, {})
    if not supports_perturbation(benchmark_key, name):
        raise ValueError(f"{benchmark_key} does not support perturbation {name!r}")
    if name == CHOICE_ORDER:
        return _choice_order(benchmark_key, task, seed)
    raise ValueError(f"Unknown perturbation: {name}")


def _result_key(result: dict) -> tuple[object, ...]:
    return (
        result.get("model"),
        result.get("benchmark"),
        result.get("variant"),
        result.get("slice"),
    )


def annotate_robustness(results: list[dict]) -> None:
    """Attach paired clean-versus-perturbed metrics to completed results."""
    baselines = {
        _result_key(result): result
        for result in results
        if not result.get("perturbation")
    }

    for result in results:
        if not result.get("perturbation"):
            continue
        baseline = baselines.get(_result_key(result))
        if baseline is None:
            continue

        baseline_tasks = {task["task_id"]: task for task in baseline.get("tasks", [])}
        pairs = [
            (baseline_tasks[task["task_id"]], task)
            for task in result.get("tasks", [])
            if task["task_id"] in baseline_tasks
        ]
        if not pairs:
            continue

        paired_total = len(pairs)
        baseline_score = sum(float(clean.get("score", 0)) for clean, _ in pairs)
        perturbed_score = sum(
            float(perturbed.get("score", 0)) for _, perturbed in pairs
        )
        regressions = sum(
            bool(clean.get("passed")) and not bool(perturbed.get("passed"))
            for clean, perturbed in pairs
        )
        recoveries = sum(
            not bool(clean.get("passed")) and bool(perturbed.get("passed"))
            for clean, perturbed in pairs
        )
        comparison = {
            "perturbation": result["perturbation"],
            "perturbation_seed": result.get("perturbation_seed"),
            "paired_total": paired_total,
            "baseline_score": round(baseline_score / paired_total, 1),
            "perturbed_score": round(perturbed_score / paired_total, 1),
            "score_delta_pp": round(
                (perturbed_score - baseline_score) / paired_total,
                1,
            ),
            "regressions": regressions,
            "regression_rate": round(regressions / paired_total * 100, 1),
            "recoveries": recoveries,
            "recovery_rate": round(recoveries / paired_total * 100, 1),
        }
        result.update(comparison)
        result["baseline_benchmark_label"] = baseline.get(
            "benchmark_label", baseline.get("benchmark")
        )
