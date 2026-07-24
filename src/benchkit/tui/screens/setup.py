"""Setup screen: choose models, benchmarks and task slices."""

from __future__ import annotations

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, SelectionList, Static
from textual.widgets.selection_list import Selection

from benchkit.benchmarks import REGISTRY
from benchkit.demo import DemoClient
from benchkit.engine import JobSpec, SliceError, parse_slice, slice_label, task_count
from benchkit.tui.formatting import fmt_count, fmt_size
from benchkit.tui.screens.modals import LimitScreen
from benchkit.tui.theme import score_palette
from benchkit.tui.widgets import SectionTitle

DESCRIPTIONS = {
    "quickbench": "fast code-generation smoke test",
    "humaneval": "code generation, pass@1",
    "mbpp": "Python programming tasks",
    "gsm8k": "grade-school math reasoning",
    "arc": "science multiple choice",
    "gpqa": "graduate-level science",
    "mmlu": "broad academic knowledge",
    "openbookqa": "elementary science reasoning",
    "winogrande": "pronoun resolution",
    "piqa": "physical commonsense",
    "boolq": "yes/no reading comprehension",
    "truthfulqa": "truthfulness multiple choice",
    "hellaswag": "sentence completion",
}


class SetupScreen(Screen[None]):
    """Pick what to run. Everything here is keyboard driven."""

    BINDINGS = [
        Binding("s", "start", "Start run"),
        Binding("f5", "start", "Start run", show=False),
        Binding("a", "select_all", "Select all"),
        Binding("n", "select_none", "Clear"),
        Binding("i", "invert", "Invert"),
        Binding("l", "set_limit", "Task limit"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.model_order: list[str] = []
        self.model_meta: dict[str, dict] = {}
        self.selected_models: set[str] = set()
        self.bench_order: list[str] = list(REGISTRY.keys())
        self.selected_benchmarks: set[str] = set()
        self.counts: dict[str, int] = {}
        self.limits: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="setup-body"):
            with Vertical(classes="pane", id="model-pane"):
                yield SectionTitle("Models", id="model-title")
                yield Input(
                    placeholder="Filter models…", id="model-filter", classes="filter"
                )
                yield SelectionList(id="model-list")
            with Vertical(classes="pane", id="bench-pane"):
                yield SectionTitle("Benchmarks", id="bench-title")
                yield Input(
                    placeholder="Filter benchmarks…",
                    id="bench-filter",
                    classes="filter",
                )
                yield SelectionList(id="bench-list")
        with Horizontal(id="setup-options"):
            yield Label("Tasks per benchmark", classes="field-label wide")
            yield Input(placeholder="all", id="global-limit", classes="limit-input")
            yield Static(
                "N first N · -N last N · A-B range · press [b]l[/b] for a per-benchmark limit",
                classes="hint",
            )
        with Horizontal(id="setup-footer"):
            yield Static("", id="plan-summary")
            yield Button("Start run ▸", id="start", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self.app.client.label} · {self.app.client.host}"
        self.model_order = [model["name"] for model in self.app.models]
        self.model_meta = {model["name"]: model for model in self.app.models}
        if len(self.model_order) == 1:
            self.selected_models = set(self.model_order)
        self.selected_benchmarks = {"quickbench"} if "quickbench" in REGISTRY else set()

        self._rebuild_models()
        self._rebuild_benchmarks()
        self._refresh_summary()
        self.query_one("#model-list", SelectionList).focus()
        self.watch(self.app, "theme", self._theme_changed, init=False)
        self._count_tasks()

    def _theme_changed(self, _theme: str) -> None:
        self._rebuild_benchmarks()

    # Task counting ----------------------------------------------------

    @work(thread=True, exclusive=True, group="counts")
    def _count_tasks(self) -> None:
        for key in list(REGISTRY):
            try:
                count = task_count(key)
            except Exception:
                count = -1
            self.app.call_from_thread(self._count_ready, key, count)

    def _count_ready(self, key: str, count: int) -> None:
        self.counts[key] = count
        self._rebuild_benchmarks()
        self._refresh_summary()

    # List construction ------------------------------------------------

    def _model_prompt(self, name: str) -> Text:
        meta = self.model_meta.get(name, {})
        detail = fmt_size(meta.get("size")) or str(
            meta.get("status") or meta.get("owned_by") or ""
        )
        text = Text()
        text.append(_clip(name, 34).ljust(34))
        text.append(detail.rjust(9), style="dim")
        return text

    def _bench_prompt(self, key: str) -> Text:
        count = self.counts.get(key)
        if count is None:
            count_text = "counting…"
        elif count < 0:
            count_text = "unavailable"
        else:
            count_text = f"{fmt_count(count)} tasks"

        text = Text()
        text.append(_clip(key, 12).ljust(13))
        text.append(count_text.rjust(12), style="dim")
        limit = self.limits.get(key)
        if limit:
            accent = score_palette(self.app.current_theme.dark)["mid"]
            text.append(f"  {slice_label(limit)}", style=f"bold {accent}")
        else:
            text.append(f"  {DESCRIPTIONS.get(key, '')}", style="dim")
        return text

    def _rebuild_models(self) -> None:
        widget = self.query_one("#model-list", SelectionList)
        needle = self.query_one("#model-filter", Input).value.strip().lower()
        visible = [name for name in self.model_order if needle in name.lower()]
        highlighted = widget.highlighted
        widget.clear_options()
        widget.add_options(
            [
                Selection(self._model_prompt(name), name, name in self.selected_models)
                for name in visible
            ]
        )
        if visible:
            widget.highlighted = min(highlighted or 0, len(visible) - 1)
        self.query_one("#model-title", SectionTitle).set_detail(
            f"{len(self.selected_models)}/{len(self.model_order)} selected"
        )

    def _rebuild_benchmarks(self) -> None:
        widget = self.query_one("#bench-list", SelectionList)
        needle = self.query_one("#bench-filter", Input).value.strip().lower()
        visible = [
            key
            for key in self.bench_order
            if needle in key.lower() or needle in DESCRIPTIONS.get(key, "").lower()
        ]
        highlighted = widget.highlighted
        widget.clear_options()
        widget.add_options(
            [
                Selection(self._bench_prompt(key), key, key in self.selected_benchmarks)
                for key in visible
            ]
        )
        if visible:
            widget.highlighted = min(highlighted or 0, len(visible) - 1)
        self.query_one("#bench-title", SectionTitle).set_detail(
            f"{len(self.selected_benchmarks)}/{len(self.bench_order)} selected"
        )

    # Events -----------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-filter":
            self._rebuild_models()
        elif event.input.id == "bench-filter":
            self._rebuild_benchmarks()
        elif event.input.id == "global-limit":
            self._refresh_summary()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in {"model-filter", "bench-filter"}:
            target = (
                "#model-list" if event.input.id == "model-filter" else "#bench-list"
            )
            self.query_one(target, SelectionList).focus()
        elif event.input.id == "global-limit":
            self.action_start()

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        widget = event.selection_list
        visible = {
            widget.get_option_at_index(index).value
            for index in range(widget.option_count)
        }
        chosen = set(widget.selected)
        if widget.id == "model-list":
            self.selected_models = (self.selected_models - visible) | chosen
            self.query_one("#model-title", SectionTitle).set_detail(
                f"{len(self.selected_models)}/{len(self.model_order)} selected"
            )
        else:
            self.selected_benchmarks = (self.selected_benchmarks - visible) | chosen
            self.query_one("#bench-title", SectionTitle).set_detail(
                f"{len(self.selected_benchmarks)}/{len(self.bench_order)} selected"
            )
        self._refresh_summary()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()

    # Actions ----------------------------------------------------------

    def _focused_list(self) -> SelectionList | None:
        node = self.focused
        while node is not None:
            if isinstance(node, SelectionList):
                return node
            node = node.parent
        return None

    def action_select_all(self) -> None:
        widget = self._focused_list()
        if widget is not None:
            widget.select_all()

    def action_select_none(self) -> None:
        widget = self._focused_list()
        if widget is not None:
            widget.deselect_all()

    def action_invert(self) -> None:
        widget = self._focused_list()
        if widget is not None:
            widget.toggle_all()

    def action_focus_filter(self) -> None:
        widget = self._focused_list()
        target = (
            "#bench-filter"
            if widget is not None and widget.id == "bench-list"
            else "#model-filter"
        )
        self.query_one(target, Input).focus()

    def action_set_limit(self) -> None:
        widget = self.query_one("#bench-list", SelectionList)
        index = widget.highlighted
        if index is None:
            return
        key = widget.get_option_at_index(index).value
        total = self.counts.get(key, 0)

        def apply(spec: str | None) -> None:
            if spec:
                self.limits[key] = spec
            else:
                self.limits.pop(key, None)
            self._rebuild_benchmarks()
            self._refresh_summary()

        self.app.push_screen(LimitScreen(key, self.limits.get(key, ""), total), apply)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_start(self) -> None:
        jobs = self._build_jobs()
        if jobs is None:
            return
        if self.app.demo and isinstance(self.app.client, DemoClient):
            self.app.client.prime(sorted({job.benchmark for job in jobs}))
        self.app.start_run(jobs)

    # Planning ---------------------------------------------------------

    def _global_limit(self) -> str:
        return self.query_one("#global-limit", Input).value.strip()

    def _limit_for(self, key: str) -> str | None:
        return self.limits.get(key) or self._global_limit() or None

    def _build_jobs(self) -> list[JobSpec] | None:
        models = [name for name in self.model_order if name in self.selected_models]
        benches = [key for key in self.bench_order if key in self.selected_benchmarks]

        if not models:
            self.notify("Select at least one model", severity="error", timeout=4)
            self.query_one("#model-list", SelectionList).focus()
            return None
        if not benches:
            self.notify("Select at least one benchmark", severity="error", timeout=4)
            self.query_one("#bench-list", SelectionList).focus()
            return None

        for key in benches:
            spec = self._limit_for(key)
            try:
                parse_slice(spec, self.counts.get(key, 0) or 1)
            except SliceError:
                self.notify(
                    f"{key}: {spec!r} is not a valid slice. Use N, -N or A-B.",
                    severity="error",
                    timeout=6,
                )
                return None

        return [
            JobSpec(model, key, self._limit_for(key))
            for model in models
            for key in benches
        ]

    def _planned_tasks(self, key: str) -> int:
        total = self.counts.get(key, 0)
        if total <= 0:
            return 0
        try:
            start, end = parse_slice(self._limit_for(key), total)
        except SliceError:
            return 0
        return max(0, end - start)

    def _refresh_summary(self) -> None:
        models = len(self.selected_models)
        benches = [key for key in self.bench_order if key in self.selected_benchmarks]
        runs = models * len(benches)
        tasks = sum(self._planned_tasks(key) for key in benches) * models

        if runs == 0:
            summary = Content.from_markup(
                "[dim]Nothing selected yet — pick models and benchmarks.[/dim]"
            )
        else:
            summary = Content.from_markup(
                "[b]$models[/b][dim] model$ms  ×  [/dim][b]$benches[/b][dim] benchmark$bs"
                "  =  [/dim][b]$runs[/b][dim] run$rs  ·  [/dim][b]$tasks[/b][dim] tasks total[/dim]",
                models=models,
                ms="" if models == 1 else "s",
                benches=len(benches),
                bs="" if len(benches) == 1 else "s",
                runs=runs,
                rs="" if runs == 1 else "s",
                tasks=fmt_count(tasks),
            )
        self.query_one("#plan-summary", Static).update(summary)


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
