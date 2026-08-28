"""Prompt-set, extraction, and render-scoring behavior of the treejs arena."""

import json
from pathlib import Path

import pytest

from benchkit import artifacts
from benchkit.benchmarks import REGISTRY
from benchkit.benchmarks.treejs_arena import (
    PROMPT_PREFIX,
    PROMPT_SET_VERSION,
    PROMPTS,
    TreeJSArena,
)
from benchkit.browser import RenderResult, _host_allowed, allowed_hosts
from benchkit.report import save

PAGE = "<!DOCTYPE html><html><body><canvas></canvas></body></html>"


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    artifacts.cleanup()


def test_prompt_set_is_frozen_and_prefixed() -> None:
    tasks = TreeJSArena().load_tasks()
    assert len(tasks) == 10 == len(PROMPTS)
    assert len({slug for slug, _ in PROMPTS}) == 10
    assert [task.id for task in tasks] == [f"TreeJSArena/{i}" for i in range(10)]
    for task, (slug, scene) in zip(tasks, PROMPTS, strict=True):
        assert task.prompt.startswith(PROMPT_PREFIX)
        assert task.prompt.endswith(scene)
        assert task.metadata["slug"] == slug
        assert task.metadata["prompt_set_version"] == PROMPT_SET_VERSION
        assert TreeJSArena().build_prompt(task) == task.prompt


def test_registered_and_excluded_from_the_overall_score() -> None:
    assert REGISTRY["treejs-arena"] is TreeJSArena
    metadata = TreeJSArena().result_metadata(None)
    assert metadata["include_in_overall"] is False
    assert metadata["prompt_set_version"] == PROMPT_SET_VERSION


@pytest.mark.parametrize(
    "response",
    [
        PAGE,
        f"Here you go:\n\n```html\n{PAGE}\n```\n\nEnjoy!",
        f"```\n{PAGE}\n```",
        f"<think>planning the scene</think>\n{PAGE}",
        f"{PAGE}\nTrailing commentary that is not markup.",
    ],
)
def test_extracts_the_html_document(response) -> None:
    from benchkit.benchmarks.treejs_arena import extract_html

    extracted = extract_html(response)
    assert extracted.startswith("<!DOCTYPE html>")
    assert extracted.endswith("</html>")


def test_reports_a_missing_document_without_rendering(staged, monkeypatch) -> None:
    def fail(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("prose should never reach the browser")

    monkeypatch.setattr("benchkit.benchmarks.treejs_arena.render_page", fail)
    bench = TreeJSArena()
    result = bench.evaluate_with_feedback(bench.load_tasks()[0], "I cannot do that.")
    assert result.score == 0.0
    assert not result.error
    assert result.details["render_status"] == "no_html"
    assert "single-file HTML document" in result.feedback


def _stub(monkeypatch, outcome: RenderResult) -> None:
    def render(page, screenshot, thumbnail, **kwargs):
        screenshot.write_bytes(b"png")
        thumbnail.write_bytes(b"jpg")
        outcome.screenshot = str(screenshot)
        outcome.thumbnail = str(thumbnail)
        return outcome

    monkeypatch.setattr("benchkit.benchmarks.treejs_arena.render_page", render)


def test_scores_a_clean_render_and_keeps_the_screenshot(staged, monkeypatch) -> None:
    _stub(
        monkeypatch,
        RenderResult(
            rendered=True,
            canvases=[{"width": 1280, "height": 800}],
            contexts=["webgl2"],
            frames=240,
        ),
    )
    bench = TreeJSArena()
    result = bench.evaluate_with_feedback(bench.load_tasks()[0], PAGE)
    assert result.passed
    assert result.details["render_status"] == "rendered"
    assert Path(result.details["screenshot"]).is_file()
    assert Path(result.details["page_html"]).read_text() == PAGE


def test_failed_render_returns_console_output_as_repair_feedback(
    staged, monkeypatch
) -> None:
    _stub(
        monkeypatch,
        RenderResult(
            rendered=False,
            console_errors=["THREE.WebGLRenderer: context lost"],
            page_errors=["ReferenceError: THREE is not defined"],
            blocked_requests=["https://example.invalid/assets/three.js"],
        ),
    )
    bench = TreeJSArena()
    result = bench.evaluate_with_feedback(bench.load_tasks()[1], PAGE)
    assert result.score == 0.0
    assert not result.error
    assert result.details["render_status"] == "failed"
    assert "ReferenceError: THREE is not defined" in result.feedback
    assert "THREE.WebGLRenderer: context lost" in result.feedback
    assert "https://example.invalid/assets/three.js" in result.feedback


def test_a_browser_that_cannot_start_is_a_harness_error(staged, monkeypatch) -> None:
    _stub(monkeypatch, RenderResult(error="playwright is not installed"))
    bench = TreeJSArena()
    result = bench.evaluate_with_feedback(bench.load_tasks()[2], PAGE)
    assert result.score == 0.0
    assert result.error == "playwright is not installed"
    assert result.details["render_status"] == "harness_error"


def test_only_allowlisted_hosts_are_reachable(monkeypatch) -> None:
    hosts = allowed_hosts()
    assert _host_allowed("https://unpkg.com/three@0.160.0/build/three.module.js", hosts)
    assert _host_allowed("https://fonts.cdn.jsdelivr.net/x.css", hosts)
    assert not _host_allowed("https://example.invalid/beacon", hosts)
    assert not _host_allowed("file:///etc/passwd", hosts)

    monkeypatch.setenv("BENCHKIT_RENDER_OFFLINE", "1")
    assert allowed_hosts() == ()
    monkeypatch.setenv("BENCHKIT_RENDER_OFFLINE", "0")
    monkeypatch.setenv("BENCHKIT_RENDER_ALLOWED_HOSTS", "example.test")
    assert _host_allowed("https://example.test/three.js", allowed_hosts())
    assert not _host_allowed("https://unpkg.com/three.js", allowed_hosts())


def test_report_collects_screenshots_and_embeds_them(staged) -> None:
    directory = artifacts.task_dir("treejs-black-hole")
    (directory / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (directory / "screenshot.jpg").write_bytes(b"\xff\xd8\xfffake")
    (directory / "page.html").write_text(PAGE)
    results = [
        {
            "model": "demo:8b",
            "benchmark": "treejs-arena",
            "benchmark_label": "treejs-arena",
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
            "prompt_set_version": PROMPT_SET_VERSION,
            "include_in_overall": False,
            "tasks": [
                {
                    "task_id": "TreeJSArena/0",
                    "passed": True,
                    "score": 100.0,
                    "prompt": "prompt",
                    "response": PAGE,
                    "workspace": {
                        "render_status": "rendered",
                        "scene": "A black hole",
                        "screenshot": str(directory / "screenshot.png"),
                        "screenshot_thumbnail": str(directory / "screenshot.jpg"),
                        "page_html": str(directory / "page.html"),
                    },
                }
            ],
        }
    ]

    out = save(results)
    workspace = json.loads((out / "results.json").read_text())[0]["tasks"][0][
        "workspace"
    ]
    assert workspace["screenshot"] == "screenshots/demo-8b__TreeJSArena-0.png"
    assert workspace["page_html"] == "pages/demo-8b__TreeJSArena-0.html"
    assert (out / workspace["screenshot"]).is_file()
    assert (out / workspace["page_html"]).read_text() == PAGE

    html = (out / "results.html").read_text()
    assert "data:image/jpeg;base64," in html
    assert "Arena screenshots" in html
    assert (out / "results.md").read_text().count("**Render:** rendered") == 1
