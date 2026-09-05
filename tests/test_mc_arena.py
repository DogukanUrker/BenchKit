"""Prompt set, script extraction, and build metrics of the Minecraft arena."""

import json

import pytest

from benchkit import artifacts, sandbox
from benchkit.benchmarks import REGISTRY
from benchkit.benchmarks.mc_arena import (
    PROMPT_PREFIX,
    PROMPT_SET_VERSION,
    PROMPTS,
    MCArena,
    block_table,
    extract_script,
    measure,
    parse_blocks,
)
from benchkit.mc_render import BUILD_SIZE, MCRenderResult
from benchkit.report import arena_results, save
from benchkit.sandbox import SandboxError, ScriptRun, _CappedStream

SCRIPT = '# /// script\n# dependencies = []\n# ///\nimport json\nprint("[]")'


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    artifacts.cleanup()


@pytest.fixture
def offline(monkeypatch):
    """Never start a browser or Docker container from the test suite."""
    monkeypatch.setattr(
        "benchkit.benchmarks.mc_arena.render_build",
        lambda *args, **kwargs: MCRenderResult(error="rendering disabled in tests"),
    )


def run_ok(stdout: str) -> ScriptRun:
    return ScriptRun(exit_code=0, stdout=stdout, stderr="")


def house(blocks=None) -> list[dict]:
    """A small grounded build with more than one block type."""
    build = blocks if blocks is not None else []
    for x in range(4):
        for z in range(4):
            build.append({"x": x, "y": 0, "z": z, "block": "minecraft:stone"})
    for x in range(4):
        build.append({"x": x, "y": 1, "z": 0, "block": "minecraft:oak_planks"})
    return build


def test_prompt_set_is_frozen_and_prefixed() -> None:
    tasks = MCArena().load_tasks()
    assert len(tasks) == 5 == len(PROMPTS)
    assert len({slug for slug, _, _ in PROMPTS}) == 5
    assert len({brief for _, _, brief in PROMPTS}) == 5
    assert [task.id for task in tasks] == [f"MCArena/{i}" for i in range(5)]
    for task, (slug, difficulty, brief) in zip(tasks, PROMPTS, strict=True):
        assert task.prompt.startswith(PROMPT_PREFIX)
        assert task.prompt.endswith(brief)
        assert task.metadata["slug"] == slug
        assert task.metadata["difficulty"] == difficulty
        assert task.metadata["prompt_set_version"] == PROMPT_SET_VERSION
        assert MCArena().build_prompt(task) == task.prompt


def test_prompt_shows_the_block_id_format() -> None:
    # Without a full id spelled out, small models answer "oak plank" and
    # every block fails validation for the wrong reason.
    assert '"minecraft:oak_planks"' in PROMPT_PREFIX
    assert str(BUILD_SIZE - 1) in PROMPT_PREFIX
    assert "# /// script" in PROMPT_PREFIX


def test_registered_and_excluded_from_the_overall_score() -> None:
    assert REGISTRY["mc-arena"] is MCArena
    metadata = MCArena().result_metadata(None)
    assert metadata["include_in_overall"] is False
    assert metadata["scoring"] == "build-metrics"


@pytest.mark.parametrize(
    "response",
    [
        SCRIPT,
        f"Here you go:\n\n```python\n{SCRIPT}\n```\n\nEnjoy!",
        f"```\n{SCRIPT}\n```",
        f"<think>planning the build</think>\n{SCRIPT}",
    ],
)
def test_extracts_the_script(response) -> None:
    assert extract_script(response).strip() == SCRIPT


def test_extracts_nothing_from_prose() -> None:
    assert extract_script("I would build a nice tower out of stone.") == ""


def test_parses_json_around_chatter() -> None:
    payload, problem = parse_blocks(
        'building...\n[{"x": 0, "y": 0, "z": 0, "block": "minecraft:stone"}]\ndone\n'
    )
    assert not problem
    assert payload == [{"x": 0, "y": 0, "z": 0, "block": "minecraft:stone"}]


@pytest.mark.parametrize(
    "stdout",
    ["", "no blocks here", '{"x": 1}', "[1, 2"],
)
def test_reports_why_output_is_unusable(stdout) -> None:
    payload, problem = parse_blocks(stdout)
    assert payload == []
    assert problem


def test_block_table_matches_the_shipped_dataset() -> None:
    known, air = block_table()
    assert "minecraft:oak_planks" in known
    assert "minecraft:stone_bricks" in known
    assert "minecraft:not_a_real_block" not in known
    assert "minecraft:air" in air


def test_measures_a_plain_build() -> None:
    metrics = measure(house())
    assert metrics["blocks_total"] == 20
    assert metrics["palette_size"] == 2
    assert metrics["hallucinated_blocks"] == 0
    assert metrics["floating_blocks"] == 0
    assert metrics["blocks_out_of_bounds"] == 0
    assert 0 < metrics["palette_evenness"] <= 1


def test_counts_hallucinated_ids_and_out_of_bounds() -> None:
    metrics = measure(
        [
            {"x": 0, "y": 0, "z": 0, "block": "minecraft:stone"},
            {"x": 1, "y": 0, "z": 0, "block": "minecraft:oak_plank"},
            {"x": 2, "y": 0, "z": 0, "block": "shiny_diamond_wall"},
            {"x": BUILD_SIZE, "y": 0, "z": 0, "block": "minecraft:stone"},
            {"x": 0, "y": -1, "z": 0, "block": "minecraft:stone"},
            {"x": 0, "y": 0, "z": "1", "block": "minecraft:stone"},
        ]
    )
    assert metrics["hallucinated_blocks"] == 2
    assert metrics["hallucinated_ids"] == [
        "minecraft:oak_plank",
        "minecraft:shiny_diamond_wall",
    ]
    assert metrics["blocks_out_of_bounds"] == 2
    assert metrics["entries_malformed"] == 1
    assert metrics["blocks_total"] == 1


def test_counts_floating_blocks_and_duplicates() -> None:
    metrics = measure(
        [
            {"x": 0, "y": 0, "z": 0, "block": "minecraft:stone"},
            {"x": 0, "y": 1, "z": 0, "block": "minecraft:stone"},
            {"x": 0, "y": 1, "z": 0, "block": "minecraft:oak_planks"},
            {"x": 5, "y": 9, "z": 5, "block": "minecraft:stone"},
        ]
    )
    assert metrics["blocks_total"] == 3
    assert metrics["blocks_duplicate_positions"] == 1
    assert metrics["floating_blocks"] == 1
    assert metrics["floating_fraction"] == pytest.approx(1 / 3, abs=1e-3)


def test_single_block_spam_scores_no_palette_diversity() -> None:
    spam = [
        {"x": x, "y": 0, "z": 0, "block": "minecraft:dirt"} for x in range(BUILD_SIZE)
    ]
    metrics = measure(spam)
    assert metrics["palette_size"] == 1
    assert metrics["palette_evenness"] == 0.0
    assert metrics["palette_dominant_share"] == 1.0


def test_air_is_not_counted_as_a_placed_block() -> None:
    metrics = measure(
        [
            {"x": 0, "y": 0, "z": 0, "block": "minecraft:stone"},
            {"x": 1, "y": 0, "z": 0, "block": "minecraft:air"},
        ]
    )
    assert metrics["blocks_total"] == 1
    assert metrics["blocks_air"] == 1
    assert metrics["hallucinated_blocks"] == 0


def test_a_valid_build_passes_and_carries_its_metrics(staged, offline, monkeypatch):
    monkeypatch.setattr(
        "benchkit.benchmarks.mc_arena.run_python_script",
        lambda *args, **kwargs: run_ok(json.dumps(house())),
    )
    task = MCArena().load_tasks()[0]
    result = MCArena().evaluate_with_feedback(task, f"```python\n{SCRIPT}\n```")
    assert result.passed
    assert result.details["build_status"] == "valid"
    assert result.details["blocks_total"] == 20
    assert result.details["render_status"] == "skipped"
    assert result.details["skip_reason"]


@pytest.mark.parametrize(
    ("run", "status"),
    [
        (ScriptRun(exit_code=1, stdout="", stderr="ValueError: boom"), "script_error"),
        (
            ScriptRun(exit_code=137, stdout="", stderr="", timed_out=True),
            "script_timeout",
        ),
        (ScriptRun(exit_code=0, stdout="no json here", stderr=""), "bad_output"),
    ],
)
def test_failed_scripts_report_why_without_passing(
    staged, offline, monkeypatch, run, status
):
    monkeypatch.setattr(
        "benchkit.benchmarks.mc_arena.run_python_script", lambda *a, **k: run
    )
    task = MCArena().load_tasks()[0]
    result = MCArena().evaluate_with_feedback(task, f"```python\n{SCRIPT}\n```")
    assert not result.passed
    assert result.details["build_status"] == status
    # Feedback drives the repair turn, so it has to say what went wrong.
    assert result.feedback


def test_an_answer_without_a_script_keeps_the_response(staged, offline):
    task = MCArena().load_tasks()[0]
    result = MCArena().evaluate_with_feedback(task, "I would build a lovely tower.")
    assert not result.passed
    assert result.details["build_status"] == "no_script"
    assert result.details["response_text"]


def test_hallucinated_ids_fail_the_build(staged, offline, monkeypatch):
    build = house([{"x": 9, "y": 0, "z": 9, "block": "minecraft:oak_plank"}])
    monkeypatch.setattr(
        "benchkit.benchmarks.mc_arena.run_python_script",
        lambda *a, **k: run_ok(json.dumps(build)),
    )
    task = MCArena().load_tasks()[0]
    result = MCArena().evaluate_with_feedback(task, f"```python\n{SCRIPT}\n```")
    assert not result.passed
    assert result.details["build_status"] == "invalid"
    assert "minecraft:oak_plank" in result.feedback
    # The metrics survive a failed build; the run is still worth measuring.
    assert result.details["blocks_total"] == 20


def test_docker_trouble_is_a_harness_error_not_a_wrong_answer(
    staged, offline, monkeypatch
):
    def explode(*_args, **_kwargs):
        raise SandboxError("Could not run Docker")

    monkeypatch.setattr("benchkit.benchmarks.mc_arena.run_python_script", explode)
    task = MCArena().load_tasks()[0]
    result = MCArena().evaluate_with_feedback(task, f"```python\n{SCRIPT}\n```")
    assert result.error
    assert result.details["build_status"] == "harness_error"


def test_builds_reach_the_gallery_and_the_report(staged, offline, monkeypatch):
    monkeypatch.setattr(
        "benchkit.benchmarks.mc_arena.run_python_script",
        lambda *a, **k: run_ok(json.dumps(house())),
    )
    task = MCArena().load_tasks()[0]
    details = (
        MCArena().evaluate_with_feedback(task, f"```python\n{SCRIPT}\n```").details
    )
    results = [
        {
            "model": "demo",
            "benchmark": "mc-arena",
            "benchmark_label": "mc-arena",
            "scoring": "build-metrics",
            "score": 100.0,
            "passed": 1,
            "total": 1,
            "failures": 0,
            "loop_kills": 0,
            "timeouts": 0,
            "length_exceeded": 0,
            "harness_errors": 0,
            "concurrency": 1,
            "avg_response_time": 1.0,
            "total_time": 1.0,
            "tasks": [
                {
                    "task_id": task.id,
                    "passed": True,
                    "score": 100.0,
                    "prompt": task.prompt,
                    "response": SCRIPT,
                    "workspace": dict(details),
                }
            ],
        }
    ]
    assert arena_results(results) == results

    out = save(results, provider="test", host="http://localhost")
    assert (out / "arena.html").exists()
    assert "VALID BUILD" in (out / "arena.html").read_text(encoding="utf-8")
    # The generated script is collected next to the report, not left in staging.
    stored = json.loads((out / "results.json").read_text(encoding="utf-8"))
    script = stored[0]["tasks"][0]["workspace"]["script_py"]
    assert script.startswith("builds/")
    assert (out / script).is_file()


class _Pipe:
    """A readable stand-in for a subprocess pipe."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    def read(self, _size):
        return self.chunks.pop(0) if self.chunks else ""

    def close(self):
        self.closed = True


def test_capped_stream_keeps_a_bounded_prefix() -> None:
    # A script that prints forever must not be buffered whole in this process:
    # the container's memory limit does not reach the host's copy.
    stream = _CappedStream(_Pipe(["a" * 100] * 50), limit=150)
    stream.pump()
    assert stream.text == "a" * 150
    assert stream.truncated
    # Everything past the cap is still drained, so the writer never blocks.
    assert stream.stream.closed


def test_capped_stream_keeps_short_output_whole() -> None:
    stream = _CappedStream(_Pipe(['[{"x": 0}]']), limit=1024)
    stream.pump()
    assert stream.text == '[{"x": 0}]'
    assert not stream.truncated


def test_cleanup_forgets_the_image_it_deleted(monkeypatch) -> None:
    # cleanup_run_resources removes every image carrying this run's label,
    # mc-arena's included. Handing the deleted tag to a later job would turn
    # every task into a harness error.
    monkeypatch.setattr(sandbox, "_docker_binary", lambda: "docker")
    monkeypatch.setattr(sandbox, "_remove_labelled", lambda *args: None)
    monkeypatch.setattr(sandbox, "_MC_ARENA_READY", True)

    sandbox.cleanup_run_resources()

    assert sandbox._MC_ARENA_READY is False
