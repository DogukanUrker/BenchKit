"""Available benchmarks."""

from benchkit.benchmarks.humaneval import HumanEval

REGISTRY: dict[str, type] = {
    "humaneval": HumanEval,
}
