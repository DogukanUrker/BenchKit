# Repository Guidelines

## Project Structure & Module Organization
BenchKit is a Python 3.11 package using a `src/` layout. Core entrypoints and orchestration live in `src/benchkit/cli.py`, `runner.py`, `ollama.py`, `executor.py`, and `report.py`. Benchmark implementations belong in `src/benchkit/benchmarks/`, and bundled benchmark data lives in `src/benchkit/datasets/` as JSONL files. Runtime outputs are written to `results/<timestamp>/` and should not be committed.

## Build, Test, and Development Commands
- `uv sync`: install and lock project dependencies from `pyproject.toml` and `uv.lock`.
- `uv run benchkit`: launch the interactive benchmark runner against the default Ollama host.
- `OLLAMA_HOST=http://localhost:11434 uv run benchkit`: run against an explicit local or remote Ollama endpoint.
- `uv run benchkit --verbose`: print per-task prompts and responses and include full detail in saved reports.

There is no separate build pipeline documented here; the package metadata and CLI entrypoint are defined in `pyproject.toml`.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and short module docstrings consistent with the existing codebase. Prefer `snake_case` for modules and functions, `PascalCase` for classes, and lowercase benchmark registry keys such as `quickbench` or `humaneval`. Keep new benchmark code narrow in scope: implement `load_tasks()`, `build_prompt()`, and `evaluate()`, store dataset payloads as JSONL, and register new classes in `src/benchkit/benchmarks/__init__.py`.

## Testing Guidelines
This repository currently depends on manual validation rather than a maintained automated test suite. Before opening a PR, run `uv run benchkit`, confirm model discovery and benchmark execution, and inspect generated `results.json`, `results.csv`, and `results.md` for correctness. For scoring or parser changes, validate with a small benchmark slice before running larger suites.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Add per-task details and prompt/response to report` and `Format code and improve CLI selection`. Follow that pattern and keep each commit focused on one change. PRs should describe user-visible behavior, list touched benchmark or dataset files, include the command used for validation, and attach a terminal or report snippet when CLI output changes.

## Configuration & Data
Copy `.env.example` to `.env` and set `OLLAMA_HOST` before local runs. Do not commit `.env`, local caches, or generated `results/` artifacts. Treat bundled dataset files as source data: update benchmark logic and dataset contents together when schema or evaluation behavior changes.
