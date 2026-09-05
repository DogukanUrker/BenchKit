"""Creative Minecraft arena: build prompts scored on the blocks, not the picture.

Like treejs-arena there is no ground truth here - a watchtower has no correct
answer. Unlike treejs-arena, "it rendered" is not the score: a build renders
whether it is a castle or three blocks of dirt, so small models would pass a
render check every time. What is scored instead is everything a block list can
be checked against on its own: whether the script ran, whether the block ids
exist, how much was actually built, whether it stayed in the volume, how varied
the palette is, and how much of it is floating in the air.

The model answers with one self-contained Python script. The script runs in a
network-less Docker container and prints the build as JSON; the block list it
prints is what gets measured, and then photographed from three fixed cameras
for human comparison.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, deque
from functools import lru_cache
from importlib.resources import files

from benchkit.artifacts import task_dir
from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.evaluation import EvaluationResult
from benchkit.mc_render import BUILD_SIZE, VIEWS, render_build, viewer_version
from benchkit.sandbox import SandboxError, run_python_script

# Frozen prompt set. Bump the version (and never edit a shipped prompt in
# place) so screenshots from different runs stay comparable.
PROMPT_SET_VERSION = "v1"

# The block table the renderer and the id check share, so a build cannot be
# marked valid with an id the renderer would then quietly drop.
DATASET = files("benchkit").joinpath("datasets/mc_blocks_1_20_1.jsonl")

SCRIPT_TIMEOUT_S = 60.0
MAX_ENTRIES = 200_000
MAX_FEEDBACK_CHARS = 4000

# The example matters more than it looks. Without an id spelled out in full,
# smaller models answer with "oak plank" or "Oak Planks" and every block is
# rejected for the wrong reason - bad formatting rather than a bad build.
PROMPT_PREFIX = f"""\
You are building a Minecraft structure. Answer with one self-contained Python \
script and nothing else.

The script must print, to stdout, a single JSON array of the blocks in the \
build:

[{{"x": 0, "y": 0, "z": 0, "block": "minecraft:oak_planks"}}, \
{{"x": 0, "y": 1, "z": 0, "block": "minecraft:cobblestone"}}]

Rules:
- Every block id is the full namespaced Minecraft id in lower case, exactly \
like "minecraft:oak_planks", "minecraft:stone_bricks" or "minecraft:glass". \
Ids that are not real Minecraft blocks are rejected.
- x, y and z are integers in the range 0 to {BUILD_SIZE - 1}. y is up, and \
y = 0 is the ground. Anything outside that {BUILD_SIZE}x{BUILD_SIZE}x\
{BUILD_SIZE} volume is rejected.
- Omit air; only list the blocks you place.
- The script begins with PEP 723 inline metadata and uses the standard \
library only. It runs with no network access.

Start the script with:

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

Build this:
"""

# fmt: off
PROMPTS: tuple[tuple[str, str, str], ...] = (
    ("stone-pillar", "trivial", "A single stone brick pillar 8 blocks tall, standing on a 5x5 stone slab-free base of smooth stone."),
    ("checkerboard-floor", "trivial", "A 12x12 checkerboard floor one block thick on the ground, alternating black concrete and white concrete."),
    ("stone-watchtower", "medium", "A small square stone watchtower: cobblestone walls, a doorway on one side, a window on each of the other three sides, a wooden floor partway up, and a battlemented roof."),
    ("arched-bridge", "medium", "A wooden footbridge with railings, arching over a river of water five blocks wide, with stone banks on both sides."),
    ("windmill", "hard", "A windmill: a round stone tower with a wooden balcony near the top, a sloped roof, and four sails on a horizontal axle on one side."),
)
# fmt: on


@lru_cache(maxsize=1)
def block_table() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(every block id, the air ids)`` for the pinned version."""
    known: set[str] = set()
    air: set[str] = set()
    with DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            known.add(row["block"])
            if row.get("air"):
                air.add(row["block"])
    return frozenset(known), frozenset(air)


def extract_script(response: str) -> str:
    """Pull the Python script out of a model response, fences and all."""
    text = strip_think_tags(response).strip()
    if "```" in text:
        parts = text.split("```")
        blocks = []
        for index in range(1, len(parts), 2):
            block = parts[index]
            first, newline, rest = block.partition("\n")
            label = first.strip().lower()
            if newline and (
                label in {"python", "py", "python3", ""} or " " not in label
            ):
                blocks.append(rest)
            else:
                blocks.append(block)
        candidates = [block for block in blocks if block.strip()]
        if candidates:
            return max(candidates, key=len).strip()
    # An unfenced answer is still a script if it looks like one.
    if re.search(r"^\s*(import |from |print\(|# /// script)", text, re.MULTILINE):
        return text
    return ""


def parse_blocks(stdout: str) -> tuple[list, str]:
    """Read the JSON array a build script printed, or say why it could not."""
    text = stdout.strip()
    if not text:
        return [], "the script printed nothing to stdout"
    try:
        payload = json.loads(text)
    except ValueError:
        # Scripts often print a line of chatter before the JSON.
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end <= start:
            return [], "stdout did not contain a JSON array"
        try:
            payload = json.loads(text[start : end + 1])
        except ValueError as exc:
            return [], f"stdout was not valid JSON ({exc})"
    if not isinstance(payload, list):
        return [], f"stdout was a JSON {type(payload).__name__}, not an array"
    if len(payload) > MAX_ENTRIES:
        return (
            [],
            f"the build listed {len(payload)} entries; the limit is {MAX_ENTRIES}",
        )
    return payload, ""


def _coordinate(value: object) -> int | None:
    """Accept an integer coordinate, including ``4.0``; reject anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _normalize_id(value: object) -> str:
    name = str(value).strip().lower()
    return name if ":" in name else f"minecraft:{name}"


def _entropy(counts: Counter[str]) -> float:
    """Shannon evenness of the palette: 0 for one block, 1 for a flat mix."""
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    bits = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return round(bits / math.log2(len(counts)), 4)


def _floating(positions: set[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
    """Blocks with no path of touching blocks down to the ground."""
    if not positions:
        return set()
    grounded = {position for position in positions if position[1] == 0}
    queue = deque(grounded)
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            neighbour = (x + dx, y + dy, z + dz)
            if neighbour in positions and neighbour not in grounded:
                grounded.add(neighbour)
                queue.append(neighbour)
    return positions - grounded


def measure(payload: list) -> dict:
    """Turn a printed block list into the metrics mc-arena actually scores."""
    known, air = block_table()
    placements: dict[tuple[int, int, int], str] = {}
    hallucinated: Counter[str] = Counter()
    malformed = 0
    out_of_bounds = 0
    air_entries = 0
    duplicates = 0

    for entry in payload:
        if not isinstance(entry, dict):
            malformed += 1
            continue
        x = _coordinate(entry.get("x"))
        y = _coordinate(entry.get("y"))
        z = _coordinate(entry.get("z"))
        name = entry.get("block")
        if x is None or y is None or z is None or not isinstance(name, str):
            malformed += 1
            continue
        block = _normalize_id(name)
        if block not in known:
            hallucinated[block] += 1
            continue
        if not all(0 <= value < BUILD_SIZE for value in (x, y, z)):
            out_of_bounds += 1
            continue
        if block in air:
            air_entries += 1
            continue
        if (x, y, z) in placements:
            duplicates += 1
        placements[(x, y, z)] = block

    palette = Counter(placements.values())
    floating = _floating(set(placements))
    total = len(placements)
    extent = [0, 0, 0]
    if placements:
        for axis in range(3):
            values = [position[axis] for position in placements]
            extent[axis] = max(values) - min(values) + 1

    return {
        "entries": len(payload),
        "entries_malformed": malformed,
        "blocks_total": total,
        "blocks_duplicate_positions": duplicates,
        "blocks_out_of_bounds": out_of_bounds,
        "blocks_air": air_entries,
        "hallucinated_blocks": sum(hallucinated.values()),
        "hallucinated_ids": sorted(hallucinated),
        "palette_size": len(palette),
        "palette_evenness": _entropy(palette),
        "palette_dominant_share": (
            round(max(palette.values()) / total, 4) if total else 0.0
        ),
        "palette_top": [
            {"block": block, "count": count} for block, count in palette.most_common(6)
        ],
        "floating_blocks": len(floating),
        "floating_fraction": round(len(floating) / total, 4) if total else 0.0,
        "bounding_box": extent,
        "volume_fill": round(total / BUILD_SIZE**3, 6),
        "blocks": [
            {"x": x, "y": y, "z": z, "block": block}
            for (x, y, z), block in placements.items()
        ],
    }


def _clip(text: str) -> str:
    if len(text) <= MAX_FEEDBACK_CHARS:
        return text
    return "…\n" + text[-MAX_FEEDBACK_CHARS:]


class MCArena:
    """Frozen Minecraft build prompts, scored from the block list they print."""

    name = "mc-arena"
    task_count = len(PROMPTS)
    prompt_set_version = PROMPT_SET_VERSION
    # A build-validity rate is not a capability score, so it stays out of the
    # headline average the way RULER and treejs-arena do.
    include_in_overall = False
    list_note = f"creative · build metrics · prompt set {PROMPT_SET_VERSION}"
    evaluation_activity = "running the build script in Docker and rendering it"

    def load_tasks(self) -> list[Task]:
        return [
            Task(
                id=f"MCArena/{index}",
                prompt=f"{PROMPT_PREFIX}\n{brief}",
                metadata={
                    "slug": slug,
                    "difficulty": difficulty,
                    "scene": brief,
                    "prompt_set_version": PROMPT_SET_VERSION,
                },
            )
            for index, (slug, difficulty, brief) in enumerate(PROMPTS)
        ]

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def result_metadata(self, variant: str | None = None) -> dict:
        return {
            "include_in_overall": self.include_in_overall,
            "prompt_set_version": PROMPT_SET_VERSION,
            "scoring": "build-metrics",
            "minecraft_version": viewer_version().get("minecraft_version", ""),
        }

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        slug = str(task.metadata.get("slug") or task.id)
        base = {
            "scoring": "build-metrics",
            "prompt_set_version": PROMPT_SET_VERSION,
            "prompt_slug": slug,
            "difficulty": str(task.metadata.get("difficulty") or ""),
            "scene": str(task.metadata.get("scene") or ""),
        }
        directory = task_dir(f"mc-{slug}")

        script = extract_script(response)
        if not script:
            answer = directory / "response.txt"
            answer.write_text(response, encoding="utf-8")
            return EvaluationResult(
                score=0.0,
                feedback=(
                    "No Python script was found in the answer. Reply with the "
                    "complete script and nothing else."
                ),
                details={
                    **base,
                    "build_status": "no_script",
                    "response_text": str(answer),
                },
            )

        source = directory / "build.py"
        source.write_text(script, encoding="utf-8")
        base["script_py"] = str(source)

        try:
            run = run_python_script(script, timeout_s=SCRIPT_TIMEOUT_S)
        except SandboxError as exc:
            # Docker itself is unavailable or broken: not the model's failure.
            return EvaluationResult(
                score=0.0,
                error=f"mc-arena needs Docker to run generated scripts: {exc}",
                details={**base, "build_status": "harness_error"},
            )

        base["script_exit_code"] = run.exit_code
        if run.timed_out:
            return EvaluationResult(
                score=0.0,
                feedback=(
                    f"The script was still running after {SCRIPT_TIMEOUT_S:g}s and "
                    "was killed. Generate the blocks directly instead of "
                    "searching or simulating, and reply with the complete "
                    "corrected script."
                ),
                details={**base, "build_status": "script_timeout"},
            )
        if not run.ok:
            return EvaluationResult(
                score=0.0,
                feedback=(
                    "The script exited with an error. Its output was:\n\n"
                    f"{_clip(run.stderr)}\n\n"
                    "Fix the cause and reply with the complete corrected script."
                ),
                details={**base, "build_status": "script_error"},
            )

        payload, problem = parse_blocks(run.stdout)
        if problem:
            return EvaluationResult(
                score=0.0,
                feedback=(
                    f"The script ran, but {problem}. It has to print one JSON "
                    'array of {"x", "y", "z", "block"} objects and '
                    "nothing else. Reply with the complete corrected script."
                ),
                details={**base, "build_status": "bad_output"},
            )

        metrics = measure(payload)
        blocks = metrics.pop("blocks")
        details = {**base, **metrics, "build_status": "valid"}

        listing = directory / "blocks.json"
        listing.write_text(json.dumps(blocks), encoding="utf-8")
        details["blocks_json"] = str(listing)

        # The picture is never the verdict, so a machine that cannot render
        # still scores the build. It just has nothing to show for it.
        render = render_build(blocks, directory)
        details["render_status"] = "rendered" if render.rendered else "skipped"
        if render.rendered:
            details["screenshot"] = render.contact_sheet
            for view in VIEWS:
                if path := render.views.get(view):
                    details[f"screenshot_{view}"] = path
        else:
            details["skip_reason"] = render.error or "the build did not render"
        if render.console_errors or render.page_errors:
            details["render_diagnostics"] = render.diagnostics

        if failure := self._invalid(metrics):
            return EvaluationResult(
                score=0.0,
                feedback=(
                    f"The script ran, but {failure} Reply with the complete "
                    "corrected script."
                ),
                details={**details, "build_status": "invalid"},
            )
        return EvaluationResult(score=1.0, details=details)

    def _invalid(self, metrics: dict) -> str:
        """Say what makes a printed build unusable, or nothing if it is fine.

        This is a validity check, not a taste check: a dull build passes. Only
        things that make the block list not a build at all fail here, and the
        quality metrics are carried on the task either way.
        """
        if metrics["hallucinated_blocks"]:
            names = ", ".join(metrics["hallucinated_ids"][:8])
            return (
                f"{metrics['hallucinated_blocks']} block(s) used ids that do not "
                f"exist in Minecraft ({names}). Use real namespaced ids."
            )
        if metrics["blocks_out_of_bounds"]:
            return (
                f"{metrics['blocks_out_of_bounds']} block(s) fell outside the "
                f"{BUILD_SIZE}x{BUILD_SIZE}x{BUILD_SIZE} volume."
            )
        if metrics["entries_malformed"]:
            return (
                f"{metrics['entries_malformed']} entries were not "
                '{"x", "y", "z", "block"} objects with integer coordinates.'
            )
        if not metrics["blocks_total"]:
            return "the build was empty."
        return ""

    def evaluate(self, task: Task, response: str) -> bool:
        return self.evaluate_with_feedback(task, response).passed
