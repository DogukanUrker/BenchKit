# Repository Guidelines

## Project Structure & Module Organization
BenchKit is a Python 3.11 package using a `src/` layout. `src/benchkit/cli.py` parses flags and either launches the TUI or runs headless. `engine.py` owns benchmark execution (jobs, slices, pause/skip/stop controls, events) and is shared by both front-ends; `client.py` talks to inference servers, `executor.py` sandboxes generated code, `report.py` writes artifacts, and `runner.py` renders headless progress. The terminal UI lives in `src/benchkit/tui/` (`app.py`, `app.tcss`, `theme.py`, `widgets.py`, `screens/`), and `demo.py` provides the offline client behind `--demo`. Benchmark implementations belong in `src/benchkit/benchmarks/`, and bundled benchmark data lives in `src/benchkit/datasets/` as JSONL files. Runtime outputs are written to `results/<timestamp>/` and should not be committed.

## Build, Test, and Development Commands
- `uv sync`: install and lock project dependencies from `pyproject.toml` and `uv.lock`.
- `uv run benchkit`: launch the terminal app against the configured host.
- `uv run benchkit --demo`: drive the whole UI offline with fake models - the fastest way to check TUI changes.
- `OLLAMA_HOST=http://localhost:11434 uv run benchkit`: run against an explicit local or remote Ollama endpoint.
- `uv run benchkit --headless --models MODEL --benchmarks sanity:5 --verbose`: scripted run that prints per-task prompts and responses.
- `uv run benchkit --list`: print the benchmark registry with task counts.
- `uv run pre-commit install`: install the hooks (ruff on commit, pytest on push).
- `uv run ruff check .` / `uv run ruff format .`: lint and format.
- `uv run pre-commit run -a`: run every hook against the whole tree.

The package metadata and CLI entrypoint are defined in `pyproject.toml`; `uv build` produces the sdist and wheel. Two workflows run on every pull request: `.github/workflows/lint.yml` (ruff check and format) and `.github/workflows/tests.yml` (the test suite on Python 3.11-3.13, an offline `--demo` smoke run and the package build).

## Coding Style & Naming Conventions
Ruff owns formatting and linting; its configuration lives in `pyproject.toml` (88 columns, `py311` target). Run `uv run ruff format .` rather than hand-wrapping, and fence hand-aligned data tables with `# fmt: off` / `# fmt: on` when the layout matters. Use 4-space indentation, type hints, and short module docstrings consistent with the existing codebase. Prefer `snake_case` for modules and functions, `PascalCase` for classes, and lowercase benchmark registry keys such as `sanity` or `humaneval`. Keep new benchmark code narrow in scope: implement `load_tasks()`, `build_prompt()`, and `evaluate()`, store dataset payloads as JSONL, and register new classes in `src/benchkit/benchmarks/__init__.py`.

## Testing Guidelines
Streaming, parser, and loop-detection behavior is covered by `tests/`; run it with `uv run pytest`. Everything else depends on manual validation. Before opening a PR, run `uv run benchkit`, confirm model discovery and benchmark execution, and inspect generated `results.json`, `results.csv`, and `results.md` for correctness. For scoring or parser changes, validate with a small benchmark slice before running larger suites. TUI changes can be driven without a terminal via Textual's pilot (`async with BenchKitApp(demo=True).run_test() as pilot`), which also captures SVG screenshots with `app.save_screenshot()`.

## Commit & Pull Request Guidelines
See `CONTRIBUTING.md` for the tooling walkthrough. Recent commits use short, imperative subjects such as `Add per-task details and prompt/response to report` and `Format code and improve CLI selection`. Follow that pattern and keep each commit focused on one change. PRs should describe user-visible behavior, list touched benchmark or dataset files, include the command used for validation, and attach a terminal or report snippet when CLI output changes.

## Configuration & Data
Copy `.env.example` to `.env` and set `OLLAMA_HOST` before local runs. Do not commit `.env`, local caches, or generated `results/` artifacts. Treat bundled dataset files as source data: update benchmark logic and dataset contents together when schema or evaluation behavior changes.

## Product & Feature Map
Keep the public README short and human-oriented. Put implementation guidance,
edge cases, and agent-facing operational detail here or next to the relevant
code. Treat the CLI's `--help`, `.env.example`, and the benchmark registry as
the canonical, version-matched references rather than duplicating large option
tables in the README.

### User Flows
- The TUI moves through Connect, Setup, Run, and Results screens. It supports
  model and benchmark filters, per-benchmark slices, live task inspection,
  pause, skip, stop, sortable results, and report drill-down.
- `--demo` exercises the TUI offline, including healthy and deliberately
  looping traces. Demo mode does not support the Pi harness.
- `--headless` uses the same engine and report pipeline as the TUI. `--verbose`
  prints prompts, available reasoning traces, and responses.
- `benchkit history` serves completed benchmark and performance reports from
  one or more results directories on localhost.
- `benchkit perf MODEL` profiles prompt processing, generation speed, time to
  first token, wall time, and client overhead across configurable contexts.

### Generation, Loops, and Retries
- Generation streams have no BenchKit token cap. `BENCHKIT_TIMEOUT` is the hard
  per-task deadline; partial traces are retained on timeout, loop kill, stop,
  and other recoverable terminal states.
- Reasoning comes from Ollama `thinking`, OpenAI-compatible
  `reasoning_content`, or inline `<think>` blocks. Providers that hide it must
  be reported as `NO TRACE`, not as producing no reasoning.
- Loop killing is controlled by `BENCHKIT_LOOP_KILL`,
  `BENCHKIT_LOOP_KILL_PERCENT`, and `BENCHKIT_LOOP_KILL_SECONDS`. Detection
  remains visible when killing is disabled. Only a confirmed, continuously
  growing suffix cycle is actionable; global repetition and code similarity
  are advisory.
- Transient gateway, rate-limit, DNS, connection, and dropped-stream failures
  use configurable exponential backoff. Never replay a generation after it has
  emitted tokens. Client errors are not retried. See `.env.example` and
  `client.py` for the current knobs and exact policy.

### Harnesses and Perturbations
- `--harness direct`, `pi`, and `both` compare raw generation with the stock Pi
  coding agent. Paired scoring uses only items valid on both sides.
- Pi requires Docker. Each task gets a fresh persistent `/workspace` in an
  isolated container plus a restricted inference proxy. Do not add host
  mounts, the Docker socket, direct network egress, hidden answers, or hidden
  tests to the agent environment.
- `--repair-attempts 1` gives an incorrect answer one sanitized verifier
  message and one full replacement attempt. Feedback must never expose the
  expected answer or hidden test bodies.
- `aider-polyglot` is Pi-only and uses pinned task content and toolchains. It
  measures the stock Pi protocol on Aider tasks, not Aider's edit formats.
- `--perturbation choice-order` runs supported MCQ tasks clean and with a
  deterministic permutation whose correct option moves. Perturbed jobs are
  paired with the baseline and excluded from the overall model score.

### Concurrency and Metrics
- Request concurrency is detected from server slot endpoints or explicit
  metadata and is bounded by task count. Model jobs remain sequential; tasks
  within the active job may run concurrently.
- Pause waits for active requests to drain before withholding new work. Skip,
  stop, and interrupt cancel in-flight requests.
- Reports distinguish aggregate throughput, per-stream throughput, effective
  concurrency, and throughput coverage. Preserve raw timing/token denominators
  and the legacy `tok_s` compatibility alias when changing report schemas.
- Harness errors are excluded from scores and paired comparisons. Ordinary
  incorrect answers, timeouts, length limits, and loop kills remain scored
  failures when applicable.

### Reports and Benchmarks
- Every run writes `results.json`, `results.csv`, `results.md`, and a standalone
  `results.html`. Performance runs write matching `perf.*` artifacts. Keep old
  reports readable when adding fields; missing historical fields mean “not
  captured,” not zero.
- Benchmark metadata, counts, tags, perturbation support, and descriptions are
  canonical in `src/benchkit/benchmarks/__init__.py`; inspect it or run
  `uv run benchkit --list` instead of maintaining a second table.
- RULER is generated deterministically at runtime and reported by context
  bucket. It is excluded from overall-score averaging so its degradation curve
  stays visible.
- EvalPlus datasets may be downloaded and cached on first use. Generated code
  is untrusted; preserve evaluator guards and process isolation.
