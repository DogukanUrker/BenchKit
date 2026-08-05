# Repository Guidelines

## Project Structure & Module Organization
BenchKit is a Python 3.11 package using a `src/` layout. `src/benchkit/cli.py` parses flags and either launches the TUI or runs headless. `engine.py` owns benchmark execution (jobs, slices, pause/skip/stop controls, events) and is shared by both front-ends; `client.py` talks to inference servers, `executor.py` sandboxes generated code, `report.py` writes artifacts, and `runner.py` renders headless progress. The terminal UI lives in `src/benchkit/tui/` (`app.py`, `app.tcss`, `theme.py`, `widgets.py`, `screens/`), and `demo.py` provides the offline client behind `--demo`. Benchmark implementations belong in `src/benchkit/benchmarks/`, and bundled benchmark data lives in `src/benchkit/datasets/` as JSONL files. Runtime outputs are written to `results/<timestamp>/` and should not be committed.

## Build, Test, and Development Commands
- `uv sync`: install and lock project dependencies from `pyproject.toml` and `uv.lock`.
- `uv run benchkit`: launch the terminal app against the configured host.
- `uv run benchkit --demo`: drive the whole UI offline with fake models - the fastest way to check TUI changes.
- `OLLAMA_HOST=http://localhost:11434 uv run benchkit`: run against an explicit local or remote Ollama endpoint.
- `uv run benchkit --headless --models MODEL --benchmarks quickbench:5 --verbose`: scripted run that prints per-task prompts and responses.
- `uv run benchkit --list`: print the benchmark registry with task counts.
- `uv run pre-commit install`: install the hooks (ruff on commit, pytest on push).
- `uv run ruff check .` / `uv run ruff format .`: lint and format.
- `uv run pre-commit run -a`: run every hook against the whole tree.

The package metadata and CLI entrypoint are defined in `pyproject.toml`; `uv build` produces the sdist and wheel. Two workflows run on every pull request: `.github/workflows/lint.yml` (ruff check and format) and `.github/workflows/tests.yml` (the test suite on Python 3.11-3.13, an offline `--demo` smoke run and the package build).

## Coding Style & Naming Conventions
Ruff owns formatting and linting; its configuration lives in `pyproject.toml` (88 columns, `py311` target). Run `uv run ruff format .` rather than hand-wrapping, and fence hand-aligned data tables with `# fmt: off` / `# fmt: on` when the layout matters. Use 4-space indentation, type hints, and short module docstrings consistent with the existing codebase. Prefer `snake_case` for modules and functions, `PascalCase` for classes, and lowercase benchmark registry keys such as `quickbench` or `humaneval`. Keep new benchmark code narrow in scope: implement `load_tasks()`, `build_prompt()`, and `evaluate()`, store dataset payloads as JSONL, and register new classes in `src/benchkit/benchmarks/__init__.py`.

## Testing Guidelines
Streaming, parser, and loop-detection behavior is covered by `tests/`; run it with `uv run pytest`. Everything else depends on manual validation. Before opening a PR, run `uv run benchkit`, confirm model discovery and benchmark execution, and inspect generated `results.json`, `results.csv`, and `results.md` for correctness. For scoring or parser changes, validate with a small benchmark slice before running larger suites. TUI changes can be driven without a terminal via Textual's pilot (`async with BenchKitApp(demo=True).run_test() as pilot`), which also captures SVG screenshots with `app.save_screenshot()`.

## Commit & Pull Request Guidelines
See `CONTRIBUTING.md` for the tooling walkthrough. Recent commits use short, imperative subjects such as `Add per-task details and prompt/response to report` and `Format code and improve CLI selection`. Follow that pattern and keep each commit focused on one change. PRs should describe user-visible behavior, list touched benchmark or dataset files, include the command used for validation, and attach a terminal or report snippet when CLI output changes.

## Configuration & Data
Copy `.env.example` to `.env` and set `OLLAMA_HOST` before local runs. Do not commit `.env`, local caches, or generated `results/` artifacts. Treat bundled dataset files as source data: update benchmark logic and dataset contents together when schema or evaluation behavior changes.
