"""Available benchmarks."""

from benchkit.benchmarks.arc import ARC
from benchkit.benchmarks.boolq import BoolQ
from benchkit.benchmarks.evalplus import HumanEvalPlus, MBPPPlus
from benchkit.benchmarks.gpqa import GPQA
from benchkit.benchmarks.gsm8k import GSM8K
from benchkit.benchmarks.hellaswag import HellaSwag
from benchkit.benchmarks.humaneval import HumanEval
from benchkit.benchmarks.ifeval import IFEval
from benchkit.benchmarks.mbpp import MBPP
from benchkit.benchmarks.mmlu import MMLU
from benchkit.benchmarks.openbookqa import OpenBookQA
from benchkit.benchmarks.piqa import PIQA
from benchkit.benchmarks.quickbench import QuickBench
from benchkit.benchmarks.ruler import RULER
from benchkit.benchmarks.truthfulqa import TruthfulQA
from benchkit.benchmarks.winogrande import WinoGrande

REGISTRY: dict[str, type] = {
    # Generative suites first, then the multiple-choice ones.
    "quickbench": QuickBench,
    "humaneval": HumanEval,
    "humaneval-plus": HumanEvalPlus,
    "mbpp": MBPP,
    "mbpp-plus": MBPPPlus,
    "gsm8k": GSM8K,
    "ifeval": IFEval,
    "ruler": RULER,
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
    "quickbench": ("code", "generative", "smoke"),
    "humaneval": ("code", "generative"),
    "humaneval-plus": ("code", "generative"),
    "mbpp": ("code", "generative"),
    "mbpp-plus": ("code", "generative"),
    "gsm8k": ("math", "generative"),
    "ifeval": ("instruction", "generative"),
    "ruler": ("long-context", "retrieval", "generative"),
    "gpqa": ("knowledge", "mcq", "low-signal"),
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
