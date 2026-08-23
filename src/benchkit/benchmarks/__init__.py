"""Available benchmarks."""

from benchkit.benchmarks.aider_polyglot import AiderPolyglot
from benchkit.benchmarks.arc import ARC
from benchkit.benchmarks.boolq import BoolQ
from benchkit.benchmarks.evalplus import HumanEvalPlus, MBPPPlus
from benchkit.benchmarks.git_surgery import GitSurgery
from benchkit.benchmarks.gpqa import GPQA
from benchkit.benchmarks.gsm8k import GSM8K
from benchkit.benchmarks.hellaswag import HellaSwag
from benchkit.benchmarks.humaneval import HumanEval
from benchkit.benchmarks.ifeval import IFEval
from benchkit.benchmarks.mbpp import MBPP
from benchkit.benchmarks.mmlu import MMLU
from benchkit.benchmarks.mmlu_pro import MMLUPro
from benchkit.benchmarks.openbookqa import OpenBookQA
from benchkit.benchmarks.patcheval import PatchEval
from benchkit.benchmarks.piqa import PIQA
from benchkit.benchmarks.ruler import RULER, RULERFull
from benchkit.benchmarks.sanity import Sanity
from benchkit.benchmarks.truthfulqa import TruthfulQA
from benchkit.benchmarks.winogrande import WinoGrande
from benchkit.benchmarks.xstest import XSTest

REGISTRY: dict[str, type] = {
    # Generative suites first, then the multiple-choice ones.
    "aider-polyglot": AiderPolyglot,
    "git-surgery": GitSurgery,
    "patcheval": PatchEval,
    "sanity": Sanity,
    "humaneval": HumanEval,
    "humaneval-plus": HumanEvalPlus,
    "mbpp": MBPP,
    "mbpp-plus": MBPPPlus,
    "gsm8k": GSM8K,
    "ifeval": IFEval,
    "ruler": RULER,
    "ruler-full": RULERFull,
    "xstest": XSTest,
    "gpqa": GPQA,
    "mmlu-pro": MMLUPro,
    # Multiple-choice suites.
    "mmlu": MMLU,
    "arc": ARC,
    "openbookqa": OpenBookQA,
    "winogrande": WinoGrande,
    "piqa": PIQA,
    "boolq": BoolQ,
    "truthfulqa": TruthfulQA,
    "hellaswag": HellaSwag,
}

DESCRIPTIONS: dict[str, str] = {
    "aider-polyglot": "repository editing across six languages with the Pi agent",
    "git-surgery": "stateful Git operations in isolated repositories with Pi",
    "patcheval": "real Python bug fixes with externally isolated hidden tests",
    "sanity": "25 curated checks across code, math, instructions, science, and commonsense",
    "humaneval": "Python function completions with the original unit tests",
    "humaneval-plus": "HumanEval with 122k+ tougher EvalPlus test inputs",
    "mbpp": "short Python functions from natural-language specifications",
    "mbpp-plus": "sanitized MBPP tasks with 39k+ EvalPlus test inputs",
    "gsm8k": "multi-step grade-school math problems with exact numeric answers",
    "ifeval": "prompts with code-checkable instruction-following constraints",
    "ruler": "practical long-context retrieval across all 13 RULER task families",
    "ruler-full": "research-scale RULER with 500 samples per task and context",
    "xstest": "safe compliance and unsafe refusal using an offline string matcher",
    "arc": "challenging grade-school science questions with four choices",
    "gpqa": "expert-written graduate science questions designed to resist search",
    "mmlu-pro": "harder MMLU successor: ten options and reasoning-heavy questions",
    "mmlu": "zero-shot coverage of 57 academic and professional subjects",
    "openbookqa": "elementary science questions requiring facts plus reasoning",
    "winogrande": "commonsense pronoun resolution in ambiguous sentences",
    "piqa": "choose the most physically plausible solution to everyday tasks",
    "boolq": "answer yes/no questions using evidence from a passage",
    "truthfulqa": "avoid common misconceptions and select the truthful answer",
    "hellaswag": "choose the most plausible continuation of a real-world scenario",
}
