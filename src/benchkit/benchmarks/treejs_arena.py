"""Creative three.js arena: one-shot scenes judged by whether they render.

Unlike every other suite in BenchKit there is no ground truth here. Each task
asks for one self-contained HTML file, the file is opened in headless Chromium,
and the only automatic score is binary: did the page come up without errors and
put a live canvas on screen. The screenshots are the interesting output and are
carried into the HTML report so runs can be compared side by side.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from benchkit.artifacts import task_dir
from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.browser import (
    probe_environment,
    render_page,
    render_settle_s,
    render_timeout_s,
)
from benchkit.evaluation import EvaluationResult

# Frozen prompt set. Bump the version (and never edit a shipped prompt in
# place) so screenshots from different runs stay comparable.
PROMPT_SET_VERSION = "v1"

PROMPT_PREFIX = (
    "Write a single self-contained HTML file that runs by opening it in a "
    "browser. No build step, no local asset files. The canvas must fill the "
    "window. Output only the HTML file."
)

# fmt: off
PROMPTS: tuple[tuple[str, str], ...] = (
    ("black-hole", "A black hole with a gravitationally lensed accretion disk, slowly orbiting camera."),
    ("japanese-garden", "A voxel Japanese garden at dusk with a pagoda, a bridge over a pond, and glowing stone lanterns."),
    ("alien-planet", "A procedurally generated alien planet surface with drifting fog."),
    ("spur-gears", "Three meshing spur gears with 12, 24 and 36 teeth, rotating at correct relative speeds and directions."),
    ("inner-solar-system", "The inner solar system with orbital periods in correct relative proportion, Earth completing one orbit every 10 seconds."),
    ("newtons-cradle", "A Newton's cradle with five balls where momentum transfers correctly."),
    ("walkable-maze", "A first-person walkable maze on a 10x10 grid with WASD and mouse look, with wall collision."),
    ("bar-chart", 'A 3D bar chart of {"A":42,"B":17,"C":88,"D":63,"E":5}, bars labeled and proportional.'),
    ("particle-field", "50,000 particles orbiting a central attractor with a live FPS counter."),
    ("mobius-strip", "A parametric Mobius strip with a sphere traveling along the surface."),
)
# fmt: on

# Diagnostics are pasted verbatim into the repair turn, so cap them well below
# anything that would crowd out the model's own previous answer.
MAX_FEEDBACK_CHARS = 4000


def extract_html(response: str) -> str:
    """Pull the HTML document out of a model response, fences and all."""
    text = strip_think_tags(response).strip()
    if "```" in text:
        blocks = []
        parts = text.split("```")
        for index in range(1, len(parts), 2):
            block = parts[index]
            first, _, rest = block.partition("\n")
            if first.strip().lower() in {"html", "xml", ""}:
                blocks.append(rest if "\n" in block else block)
            else:
                blocks.append(block)
        candidates = [block for block in blocks if "<" in block]
        if candidates:
            text = max(candidates, key=len).strip()

    lowered = text.lower()
    for opener in ("<!doctype html", "<html"):
        start = lowered.find(opener)
        if start != -1:
            end = lowered.rfind("</html>")
            return (
                text[start : end + len("</html>")] if end != -1 else text[start:]
            ).strip()
    # A bare fragment is still worth rendering, as long as it is markup.
    if "<canvas" in lowered or "<script" in lowered or "<body" in lowered:
        return text
    return ""


def _environment_fault(outcome) -> str:
    """Name the machine-level reason a page could not render, if there is one.

    A model is only responsible for what it wrote. When a host BenchKit itself
    allowed could not be reached, or the browser has no working WebGL stack at
    all, the fault is the environment: report it as a harness error so the task
    is excluded from the score instead of being counted as a model failure.
    """
    unreachable = [
        url
        for url in outcome.failed_requests
        if url.startswith(("http://", "https://"))
    ]
    if unreachable:
        hosts = sorted({urlsplit(url).hostname or url for url in unreachable})
        return (
            "the page could not reach "
            + ", ".join(hosts)
            + " — BenchKit allows these hosts, so this machine has no route to "
            "the module CDN (proxy, firewall, or DNS). Fix outbound HTTPS, or "
            "set BENCHKIT_RENDER_ALLOWED_HOSTS to a mirror you can reach."
        )

    if outcome.contexts:
        return ""
    ok, detail = probe_environment()
    if not ok:
        return (
            f"headless rendering is broken on this machine: {detail}. "
            "Install the browser's system dependencies "
            "('playwright install --with-deps chromium' on Debian/Ubuntu); "
            "BenchKit already asks Chromium for the SwiftShader software "
            "renderer, so no GPU is required."
        )
    return ""


def _clip(text: str) -> str:
    if len(text) <= MAX_FEEDBACK_CHARS:
        return text
    return text[:MAX_FEEDBACK_CHARS] + "\n… diagnostics truncated"


class TreeJSArena:
    """Ten frozen creative prompts scored only on whether the page renders."""

    name = "treejs-arena"
    task_count = len(PROMPTS)
    prompt_set_version = PROMPT_SET_VERSION
    # A render rate is not a capability score, so it stays out of the headline
    # average the way RULER's context curve does.
    include_in_overall = False
    list_note = f"creative · render-only scoring · prompt set {PROMPT_SET_VERSION}"
    evaluation_activity = "rendering the page in headless Chromium"

    def load_tasks(self) -> list[Task]:
        return [
            Task(
                id=f"TreeJSArena/{index}",
                prompt=f"{PROMPT_PREFIX}\n\n{prompt}",
                metadata={
                    "slug": slug,
                    "scene": prompt,
                    "prompt_set_version": PROMPT_SET_VERSION,
                },
            )
            for index, (slug, prompt) in enumerate(PROMPTS)
        ]

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def result_metadata(self, variant: str | None = None) -> dict:
        return {
            "include_in_overall": self.include_in_overall,
            "prompt_set_version": PROMPT_SET_VERSION,
            "scoring": "render-only",
        }

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        slug = str(task.metadata.get("slug") or task.id)
        base = {
            "scoring": "render-only",
            "prompt_set_version": PROMPT_SET_VERSION,
            "prompt_slug": slug,
            "scene": str(task.metadata.get("scene") or ""),
        }

        directory = task_dir(f"treejs-{slug}")
        html = extract_html(response)
        if not html:
            # Nothing to render, but the answer is still the artifact of record:
            # keep it on disk so the gallery has something to open.
            answer = directory / "response.txt"
            answer.write_text(response, encoding="utf-8")
            return EvaluationResult(
                score=0.0,
                feedback=(
                    "No HTML document was found in the answer. Reply with the "
                    "complete single-file HTML document and nothing else."
                ),
                details={
                    **base,
                    "render_status": "no_html",
                    "response_text": str(answer),
                },
            )

        page = directory / "page.html"
        page.write_text(html, encoding="utf-8")

        outcome = render_page(
            page,
            directory / "screenshot.png",
            directory / "screenshot.jpg",
            settle_s=render_settle_s(),
            timeout_s=render_timeout_s(),
        )

        details = {
            **base,
            "page_html": str(page),
            "html_bytes": len(html.encode("utf-8")),
            "console_errors": outcome.console_errors,
            "page_errors": outcome.page_errors,
            "blocked_requests": outcome.blocked_requests,
            "failed_requests": outcome.failed_requests,
            "canvases": outcome.canvases,
            "contexts": outcome.contexts,
            "animation_frames": outcome.frames,
        }
        if outcome.screenshot:
            details["screenshot"] = outcome.screenshot
        if outcome.thumbnail:
            details["screenshot_thumbnail"] = outcome.thumbnail

        if outcome.error:
            # BenchKit could not render at all: report it as a harness error so
            # the task is excluded from the score instead of blamed on the model.
            return EvaluationResult(
                score=0.0,
                error=outcome.error,
                details={**details, "render_status": "harness_error"},
            )

        if outcome.rendered:
            return EvaluationResult(
                score=1.0,
                details={**details, "render_status": "rendered"},
            )

        if fault := _environment_fault(outcome):
            return EvaluationResult(
                score=0.0,
                error=fault,
                details={**details, "render_status": "harness_error"},
            )

        return EvaluationResult(
            score=0.0,
            feedback=(
                "The page was opened in headless Chromium and did not render "
                "cleanly. Browser diagnostics:\n\n"
                f"{_clip(outcome.diagnostics)}\n\n"
                "Fix the cause and reply with the complete corrected "
                "single-file HTML document and nothing else. Network access is "
                "restricted to public module CDNs, so anything a blocked "
                "request refers to must be replaced or inlined."
            ),
            details={**details, "render_status": "failed"},
        )

    def evaluate(self, task: Task, response: str) -> bool:
        return self.evaluate_with_feedback(task, response).passed
