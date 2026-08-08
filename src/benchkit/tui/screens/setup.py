"""Setup screen: choose models, benchmarks and task slices."""

from __future__ import annotations

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection

from benchkit.benchmarks import REGISTRY, all_tags, signal_tag, tags_for
from benchkit.demo import DemoClient
from benchkit.engine import (
    JobSpec,
    SliceError,
    expand_jobs,
    parse_slice,
    slice_label,
    slice_task_count,
    task_count,
)
from benchkit.perturbations import CHOICE_ORDER, perturbations_for
from benchkit.template_check import TemplateReport, check_template
from benchkit.tui.formatting import fmt_count, fmt_size
from benchkit.tui.screens.modals import LimitScreen, TemplateCheckScreen
from benchkit.tui.theme import score_palette
from benchkit.tui.widgets import SectionTitle

DESCRIPTIONS = {
    "quickbench": "tiny Python tasks for a fast end-to-end sanity check",
    "humaneval": "Python function completions with the original unit tests",
    "humaneval-plus": "HumanEval with 122k+ tougher EvalPlus test inputs",
    "mbpp": "short Python functions from natural-language specifications",
    "mbpp-plus": "sanitized MBPP tasks with 39k+ EvalPlus test inputs",
    "gsm8k": "multi-step grade-school math problems with exact numeric answers",
    "ifeval": "prompts with code-checkable instruction-following constraints",
    "ruler": "20 synthetic tasks per context at 4k–128k; very slow",
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

MODEL_NAME_WIDTH = 40
BENCHMARK_NAME_WIDTH = 15
BENCHMARK_COUNT_WIDTH = 13


def parse_filter(needle: str) -> tuple[list[str], list[str]]:
    """Split a filter box query into required and excluded terms.

    Terms are whitespace separated and a leading `-` excludes, so
    `mcq -saturated` reads as "multiple choice, minus the saturated ones".
    """
    include: list[str] = []
    exclude: list[str] = []
    for term in needle.lower().split():
        if term.startswith("-") and len(term) > 1:
            exclude.append(term[1:])
        elif not term.startswith("-"):
            include.append(term)
    return include, exclude


def benchmark_matches(key: str, include: list[str], exclude: list[str]) -> bool:
    """Match a benchmark against filter terms by key, tag or description.

    A term that is itself a tag matches tags only, so typing `code` selects the
    same benchmarks as `--tag code` rather than also catching IFEval for the
    "code-checkable" in its description. Anything else is a free-text search
    over the key, the description and tag prefixes.
    """
    tags = set(tags_for(key))
    description = DESCRIPTIONS.get(key, "").lower()
    known_tags = set(all_tags())

    def hit(term: str) -> bool:
        if term in known_tags:
            return term in tags
        return (
            term in key.lower()
            or term in description
            or any(tag.startswith(term) for tag in tags)
        )

    if any(hit(term) for term in exclude):
        return False
    return all(hit(term) for term in include)


class SetupScreen(Screen[None]):
    """Pick what to run. Everything here is keyboard driven."""

    BINDINGS = [
        Binding("s", "start", "Start run"),
        Binding("f5", "start", "Start run", show=False),
        Binding("a", "select_all", "Select all"),
        Binding("n", "select_none", "Clear"),
        Binding("i", "invert", "Invert"),
        Binding("l", "set_limit", "Task limit"),
        Binding("p", "toggle_perturbation", "Choice order"),
        Binding("v", "check_templates", "Check templates"),
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
        self.template_status: dict[str, str] = {}
        self.checking_templates = False

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
                    placeholder="Filter benchmarks or tags…",
                    id="bench-filter",
                    classes="filter",
                )
                yield Static(
                    "tags: " + " · ".join(all_tags()) + "   [dim](-tag excludes)[/dim]",
                    id="bench-tags",
                    classes="hint",
                )
                yield SelectionList(id="bench-list")
        with Horizontal(id="setup-options"):
            yield Label("Tasks per benchmark", classes="field-label wide")
            yield Input(placeholder="all", id="global-limit", classes="limit-input")
            yield Static(
                "N first N · -N last N · A-B range · press [b]l[/b] for a per-benchmark limit",
                classes="hint",
            )
        with Horizontal(id="setup-perturbation"):
            yield Label("Perturbation", classes="field-label wide")
            yield Checkbox(
                "Run clean + choice-order",
                id="choice-order",
                compact=True,
            )
            yield Label("Seed", id="perturbation-seed-label")
            yield Input(
                value="42",
                id="perturbation-seed",
                classes="seed-input",
                disabled=True,
            )
            yield Static(
                "paired shuffled options · supported multiple-choice benchmarks only",
                classes="hint",
            )
        with Horizontal(id="setup-footer"):
            yield Static("", id="plan-summary")
            yield Button("Check templates", id="check-templates")
            yield Button("Start run ▸", id="start", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self.app.client.label} · {self.app.client.host}"
        self.model_order = [model["name"] for model in self.app.models]
        self.model_meta = {model["name"]: model for model in self.app.models}
        self.selected_models = {
            name for name, model in self.model_meta.items() if _model_is_loaded(model)
        }
        if not self.selected_models and len(self.model_order) == 1:
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
        status = str(meta.get("status") or "")
        if isinstance(meta.get("loaded"), bool):
            detail = status or ("loaded" if meta["loaded"] else "unloaded")
        else:
            detail = (
                fmt_size(meta.get("size")) or status or str(meta.get("owned_by") or "")
            )
        text = Text()
        text.append(_clip(name, MODEL_NAME_WIDTH).ljust(MODEL_NAME_WIDTH))
        status = self.template_status.get(name)
        if status == "checking":
            text.append("… checking".rjust(11), style="dim")
        elif status == "pass":
            text.append("✓ template".rjust(11), style="green")
        elif status == "warn":
            text.append("! template".rjust(11), style="yellow")
        elif status == "fail":
            text.append("✗ template".rjust(11), style="red")
        elif status == "unavailable":
            text.append("— n/a".rjust(11), style="dim")
        else:
            text.append(detail.rjust(9), style="dim")
        return text

    def _bench_prompt(self, key: str) -> Text:
        count = self.counts.get(key)
        custom_count = getattr(REGISTRY[key], "list_task_count", "")
        if count is None:
            count_text = "counting…"
        elif count < 0:
            count_text = "unavailable"
        else:
            count_text = f"{custom_count or fmt_count(count)} tasks"

        text = Text()
        text.append(_clip(key, BENCHMARK_NAME_WIDTH).ljust(BENCHMARK_NAME_WIDTH))
        text.append(count_text.rjust(BENCHMARK_COUNT_WIDTH), style="dim")
        signal = signal_tag(key)
        if signal:
            text.append(f"  {signal}", style="dim italic")

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
        include, exclude = parse_filter(
            self.query_one("#bench-filter", Input).value.strip()
        )
        visible = [
            key for key in self.bench_order if benchmark_matches(key, include, exclude)
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
        self._refresh_bench_title(len(visible))

    def _refresh_bench_title(self, shown: int | None = None) -> None:
        detail = f"{len(self.selected_benchmarks)}/{len(self.bench_order)} selected"
        if shown is None:
            shown = self.query_one("#bench-list", SelectionList).option_count
        if shown != len(self.bench_order):
            detail = f"{detail} · {shown} shown"
        self.query_one("#bench-title", SectionTitle).set_detail(detail)

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
        elif event.input.id in {"global-limit", "perturbation-seed"}:
            self.action_start()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id != "choice-order":
            return
        self.query_one("#perturbation-seed", Input).disabled = not event.value
        self._refresh_summary()

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
            self._refresh_bench_title()
        self._refresh_summary()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()
        elif event.button.id == "check-templates":
            self.action_check_templates()

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
        total = slice_task_count(key) if self.counts.get(key, 0) >= 0 else -1

        def apply(spec: str | None) -> None:
            if spec:
                self.limits[key] = spec
            else:
                self.limits.pop(key, None)
            self._rebuild_benchmarks()
            self._refresh_summary()

        self.app.push_screen(LimitScreen(key, self.limits.get(key, ""), total), apply)

    def action_toggle_perturbation(self) -> None:
        checkbox = self.query_one("#choice-order", Checkbox)
        checkbox.value = not checkbox.value

    def action_check_templates(self) -> None:
        targets = [
            name
            for name in self.model_order
            if not self.selected_models or name in self.selected_models
        ]
        if not targets:
            return
        if self.app.demo:
            self.notify("Template checks require a llama.cpp or llama-swap server.")
            return
        for name in targets:
            self.template_status[name] = "checking"
        self.checking_templates = True
        self._rebuild_models()
        self.query_one("#check-templates", Button).disabled = True
        self.query_one("#start", Button).disabled = True
        self._run_template_checks(targets)

    @work(thread=True, exclusive=True, group="template-checks")
    def _run_template_checks(self, targets: list[str]) -> None:
        reports: list[TemplateReport] = []
        for name in targets:
            report = check_template(self.app.client, name)
            reports.append(report)
            self.app.call_from_thread(self._template_check_ready, report)
        self.app.call_from_thread(self._template_checks_finished, reports)

    def _template_check_ready(self, report: TemplateReport) -> None:
        self.template_status[report.model] = report.status
        self._rebuild_models()

    def _template_checks_finished(self, reports: list[TemplateReport]) -> None:
        self.checking_templates = False
        self.query_one("#check-templates", Button).disabled = False
        self.query_one("#start", Button).disabled = False
        self.app.push_screen(TemplateCheckScreen(reports))

    def action_back(self) -> None:
        if self.checking_templates:
            self.notify("Wait for the template check to finish.", timeout=3)
            return
        self.app.pop_screen()

    def action_start(self) -> None:
        if self.checking_templates:
            self.notify("Wait for the template check to finish.", timeout=3)
            return
        jobs = self._build_jobs()
        if jobs is None:
            return
        if self.app.demo and isinstance(self.app.client, DemoClient):
            perturbations = sorted(
                {
                    (job.perturbation, job.perturbation_seed)
                    for job in jobs
                    if job.perturbation
                }
            )
            self.app.client.prime(
                sorted({job.benchmark for job in jobs}),
                perturbations=perturbations,
            )
        self.app.start_run(jobs)

    # Planning ---------------------------------------------------------

    def _global_limit(self) -> str:
        return self.query_one("#global-limit", Input).value.strip()

    def _limit_for(self, key: str) -> str | None:
        return self.limits.get(key) or self._global_limit() or None

    def _choice_order_enabled(self) -> bool:
        return self.query_one("#choice-order", Checkbox).value

    def _perturbation_seed(self) -> int | None:
        value = self.query_one("#perturbation-seed", Input).value.strip()
        try:
            return int(value)
        except ValueError:
            return None

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

        choice_order = self._choice_order_enabled()
        seed = self._perturbation_seed()
        if choice_order and seed is None:
            self.notify(
                "Choice-order seed must be an integer.",
                severity="error",
                timeout=4,
            )
            self.query_one("#perturbation-seed", Input).focus()
            return None
        if choice_order:
            unsupported = [
                key for key in benches if CHOICE_ORDER not in perturbations_for(key)
            ]
            if unsupported:
                self.notify(
                    "Choice-order is unsupported for: " + ", ".join(unsupported),
                    severity="error",
                    timeout=6,
                )
                self.query_one("#bench-list", SelectionList).focus()
                return None

        for key in benches:
            total = slice_task_count(key) if self.counts.get(key, 0) >= 0 else -1
            if total < 0:
                self.notify(
                    f"{key}: dataset could not be loaded.",
                    severity="error",
                    timeout=6,
                )
                return None
            spec = self._limit_for(key)
            try:
                # A count of 0 means the tally is still running, which
                # parse_slice reads as "syntax only".
                parse_slice(spec, max(total, 0))
            except SliceError:
                self.notify(
                    f"{key}: {spec!r} is not a valid slice. Use N, -N or A-B.",
                    severity="error",
                    timeout=6,
                )
                return None

        jobs: list[JobSpec] = []
        for model in models:
            for key in benches:
                jobs.append(JobSpec(model, key, self._limit_for(key)))
                if choice_order:
                    assert seed is not None
                    jobs.append(
                        JobSpec(
                            model,
                            key,
                            self._limit_for(key),
                            perturbation=CHOICE_ORDER,
                            perturbation_seed=seed,
                        )
                    )
        expanded = expand_jobs(jobs, self.app.client)
        if not expanded:
            self.notify(
                "No supported RULER context bucket fits the selected model(s).",
                severity="error",
                timeout=6,
            )
            return None
        return expanded

    def _planned_tasks(self, key: str) -> int:
        loaded_total = self.counts.get(key, 0)
        if loaded_total <= 0:
            return 0
        total = slice_task_count(key)
        try:
            start, end = parse_slice(self._limit_for(key), total)
        except SliceError:
            return 0
        return max(0, end - start)

    def _variant_count(self, key: str, model: str) -> int:
        buckets = getattr(REGISTRY[key], "context_buckets", ())
        if not buckets:
            return 1
        context = self.model_meta.get(model, {}).get("context_length")
        if isinstance(context, int) and not isinstance(context, bool) and context > 0:
            return sum(bucket <= context for bucket in buckets)
        return len(buckets)

    def _refresh_summary(self) -> None:
        models = len(self.selected_models)
        benches = [key for key in self.bench_order if key in self.selected_benchmarks]
        choice_order = self._choice_order_enabled()
        unsupported = (
            [key for key in benches if CHOICE_ORDER not in perturbations_for(key)]
            if choice_order
            else []
        )
        multiplier = 2 if choice_order else 1
        runs = (
            sum(
                self._variant_count(key, model)
                for model in self.selected_models
                for key in benches
            )
            * multiplier
        )
        tasks = (
            sum(
                self._planned_tasks(key) * self._variant_count(key, model)
                for model in self.selected_models
                for key in benches
            )
            * multiplier
        )

        if runs == 0:
            summary = Content.from_markup(
                "[dim]Nothing selected yet — pick models and benchmarks.[/dim]"
            )
        elif unsupported:
            summary = Content.from_markup(
                "[red]Choice-order unsupported for: $unsupported[/red]"
                "[dim] — change the benchmark selection or turn it off.[/dim]",
                unsupported=", ".join(unsupported),
            )
        else:
            summary = Content.from_markup(
                "[b]$models[/b][dim] model$ms  ×  [/dim][b]$benches[/b][dim] benchmark$bs"
                "  =  [/dim][b]$runs[/b][dim] run$rs  ·  [/dim][b]$tasks[/b][dim] tasks total"
                "$paired[/dim]",
                models=models,
                ms="" if models == 1 else "s",
                benches=len(benches),
                bs="" if len(benches) == 1 else "s",
                runs=runs,
                rs="" if runs == 1 else "s",
                tasks=fmt_count(tasks),
                paired=" · clean + choice-order" if choice_order else "",
            )
        self.query_one("#plan-summary", Static).update(summary)


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _model_is_loaded(model: dict) -> bool:
    """Return whether discovery positively identified a resident model."""
    loaded = model.get("loaded")
    if isinstance(loaded, bool):
        return loaded
    status = str(model.get("status") or "").strip().lower()
    return status in {"loaded", "running"}
