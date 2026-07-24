"""Available benchmarks."""

from benchkit.benchmarks.arc import ARC
from benchkit.benchmarks.gpqa import GPQA
from benchkit.benchmarks.gsm8k import GSM8K
from benchkit.benchmarks.hellaswag import HellaSwag
from benchkit.benchmarks.humaneval import HumanEval
from benchkit.benchmarks.mbpp import MBPP
from benchkit.benchmarks.quickbench import QuickBench
from benchkit.benchmarks.truthfulqa import TruthfulQA

REGISTRY: dict[str, type] = {
    "quickbench": QuickBench,
    "humaneval": HumanEval,
    "mbpp": MBPP,
    "gsm8k": GSM8K,
    "arc": ARC,
    "gpqa": GPQA,
    "truthfulqa": TruthfulQA,
    "hellaswag": HellaSwag,
}
