"""Available benchmarks."""

from benchkit.benchmarks.arc import ARC
from benchkit.benchmarks.boolq import BoolQ
from benchkit.benchmarks.evalplus import HumanEvalPlus, MBPPPlus
from benchkit.benchmarks.gpqa import GPQA
from benchkit.benchmarks.gsm8k import GSM8K
from benchkit.benchmarks.hellaswag import HellaSwag
from benchkit.benchmarks.humaneval import HumanEval
from benchkit.benchmarks.ifeval import IFEval
from benchkit.benchmarks.mmlu import MMLU
from benchkit.benchmarks.mbpp import MBPP
from benchkit.benchmarks.openbookqa import OpenBookQA
from benchkit.benchmarks.piqa import PIQA
from benchkit.benchmarks.quickbench import QuickBench
from benchkit.benchmarks.truthfulqa import TruthfulQA
from benchkit.benchmarks.winogrande import WinoGrande

REGISTRY: dict[str, type] = {
    # Current suites - the ones worth reaching for first.
    "quickbench": QuickBench,
    "humaneval": HumanEval,
    "humaneval-plus": HumanEvalPlus,
    "mbpp": MBPP,
    "mbpp-plus": MBPPPlus,
    "gsm8k": GSM8K,
    "ifeval": IFEval,
    "gpqa": GPQA,
    "mmlu": MMLU,
    # Classic suites - kept for comparability, largely saturated.
    "arc": ARC,
    "openbookqa": OpenBookQA,
    "winogrande": WinoGrande,
    "piqa": PIQA,
    "boolq": BoolQ,
    "truthfulqa": TruthfulQA,
    "hellaswag": HellaSwag,
}

# Benchmarks modern models still separate on, versus the older suites that
# most current models saturate. Both stay runnable; the split only drives
# how they are grouped and presented.
CURRENT_BENCHMARKS: tuple[str, ...] = (
    "quickbench",
    "humaneval",
    "humaneval-plus",
    "mbpp",
    "mbpp-plus",
    "gsm8k",
    "ifeval",
    "gpqa",
    "mmlu",
)

LEGACY_BENCHMARKS: tuple[str, ...] = tuple(
    key for key in REGISTRY if key not in CURRENT_BENCHMARKS
)


def benchmark_group(key: str) -> str:
    """Return "current" or "legacy" for a registry key."""
    return "current" if key in CURRENT_BENCHMARKS else "legacy"
