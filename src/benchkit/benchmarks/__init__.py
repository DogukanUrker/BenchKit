"""Available benchmarks."""

from benchkit.benchmarks.arc import ARC
from benchkit.benchmarks.boolq import BoolQ
from benchkit.benchmarks.gpqa import GPQA
from benchkit.benchmarks.gsm8k import GSM8K
from benchkit.benchmarks.hellaswag import HellaSwag
from benchkit.benchmarks.humaneval import HumanEval
from benchkit.benchmarks.mmlu import MMLU
from benchkit.benchmarks.mbpp import MBPP
from benchkit.benchmarks.openbookqa import OpenBookQA
from benchkit.benchmarks.piqa import PIQA
from benchkit.benchmarks.quickbench import QuickBench
from benchkit.benchmarks.truthfulqa import TruthfulQA
from benchkit.benchmarks.winogrande import WinoGrande

REGISTRY: dict[str, type] = {
    "quickbench": QuickBench,
    "humaneval": HumanEval,
    "mbpp": MBPP,
    "gsm8k": GSM8K,
    "arc": ARC,
    "gpqa": GPQA,
    "mmlu": MMLU,
    "openbookqa": OpenBookQA,
    "winogrande": WinoGrande,
    "piqa": PIQA,
    "boolq": BoolQ,
    "truthfulqa": TruthfulQA,
    "hellaswag": HellaSwag,
}
