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
from benchkit.benchmarks.piqa import PIQA
from benchkit.benchmarks.ruler import RULER
from benchkit.benchmarks.sanity import Sanity
from benchkit.benchmarks.truthfulqa import TruthfulQA
from benchkit.benchmarks.winogrande import WinoGrande
from benchkit.benchmarks.xstest import XSTest

REGISTRY: dict[str, type] = {
    # Generative suites first, then the multiple-choice ones.
    "aider-polyglot": AiderPolyglot,
    "git-surgery": GitSurgery,
    "sanity": Sanity,
    "humaneval": HumanEval,
    "humaneval-plus": HumanEvalPlus,
    "mbpp": MBPP,
    "mbpp-plus": MBPPPlus,
    "gsm8k": GSM8K,
    "ifeval": IFEval,
    "ruler": RULER,
    "xstest": XSTest,
    "gpqa": GPQA,
    "mmlu-pro": MMLUPro,
    # Classic suites - kept for comparability, largely saturated.
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
    "sanity": "25 curated checks across code, math, instructions, science, and commonsense",
    "humaneval": "Python function completions with the original unit tests",
    "humaneval-plus": "HumanEval with 122k+ tougher EvalPlus test inputs",
    "mbpp": "short Python functions from natural-language specifications",
    "mbpp-plus": "sanitized MBPP tasks with 39k+ EvalPlus test inputs",
    "gsm8k": "multi-step grade-school math problems with exact numeric answers",
    "ifeval": "prompts with code-checkable instruction-following constraints",
    "ruler": "13 RULER tasks per context at 4k–128k; very slow",
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

# Tags describe what a benchmark measures (`code`, `math`, `knowledge`,
# `commonsense`, `instruction`, `retrieval`, `long-context`), how it is answered
# (`generative`, `mcq`) and how much signal it carries on target models.
#
# `saturated` means current models score near the ceiling, so the suite mostly
# buys comparability with published numbers. `low-signal` is the wider bucket:
# every saturated suite plus GPQA, which is the opposite case - small models sit
# near the 25% floor, so it separates them just as poorly. Filtering out
# `low-signal` leaves the benchmarks that actually spread this size band.
TAGS: dict[str, tuple[str, ...]] = {
    "aider-polyglot": ("code", "generative", "agent", "polyglot"),
    "git-surgery": ("code", "generative", "agent", "git"),
    "sanity": (
        "code",
        "math",
        "knowledge",
        "commonsense",
        "instruction",
        "generative",
        "mcq",
        "smoke",
    ),
    "humaneval": ("code", "generative"),
    "humaneval-plus": ("code", "generative"),
    "mbpp": ("code", "generative"),
    "mbpp-plus": ("code", "generative"),
    "gsm8k": ("math", "generative"),
    "ifeval": ("instruction", "generative"),
    "ruler": ("long-context", "retrieval", "generative"),
    "xstest": ("safety", "refusal", "generative"),
    "gpqa": ("knowledge", "mcq", "low-signal"),
    "mmlu-pro": ("knowledge", "mcq"),
    "mmlu": ("knowledge", "mcq", "saturated", "low-signal"),
    "arc": ("knowledge", "mcq", "saturated", "low-signal"),
    "openbookqa": ("knowledge", "mcq", "saturated", "low-signal"),
    "winogrande": ("commonsense", "mcq", "saturated", "low-signal"),
    "piqa": ("commonsense", "mcq", "saturated", "low-signal"),
    "boolq": ("knowledge", "mcq", "saturated", "low-signal"),
    "truthfulqa": ("knowledge", "mcq", "saturated", "low-signal"),
    "hellaswag": ("commonsense", "mcq", "saturated", "low-signal"),
}

# Shown as a chip in the picker; the more specific one wins.
SIGNAL_TAGS: tuple[str, ...] = ("saturated", "low-signal")


def tags_for(key: str) -> tuple[str, ...]:
    """Tags for a registry key, or an empty tuple for an untagged one."""
    return TAGS.get(key, ())


def all_tags() -> list[str]:
    """Every tag in use, sorted for display."""
    return sorted({tag for tags in TAGS.values() for tag in tags})


def signal_tag(key: str) -> str | None:
    """The strongest signal warning for a benchmark, if it carries one."""
    tags = tags_for(key)
    for tag in SIGNAL_TAGS:
        if tag in tags:
            return tag
    return None


def keys_for_tags(
    include: list[str] | None = None, exclude: list[str] | None = None
) -> list[str]:
    """Registry keys carrying every `include` tag and no `exclude` tag."""
    include = [tag.strip().lower() for tag in include or [] if tag.strip()]
    exclude = [tag.strip().lower() for tag in exclude or [] if tag.strip()]

    selected = []
    for key in REGISTRY:
        tags = set(tags_for(key))
        if include and not set(include) <= tags:
            continue
        if exclude and tags & set(exclude):
            continue
        selected.append(key)
    return selected
