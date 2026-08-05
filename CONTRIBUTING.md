# Contributing

Thanks for helping out. This page covers the tooling; `AGENTS.md` covers layout
and conventions.

## Setup

```bash
uv sync                     # runtime + dev dependencies
uv run pre-commit install   # installs the pre-commit and pre-push hooks
cp .env.example .env        # then set BENCHKIT_HOST / OLLAMA_HOST
```

`pre-commit install` wires up two stages:

- **pre-commit** — `ruff check --fix`, `ruff format`, plus whitespace, YAML,
  TOML, JSON, merge-conflict and large-file checks.
- **pre-push** — the test suite.

To run everything by hand:

```bash
uv run ruff check .          # lint (add --fix to autofix)
uv run ruff format .         # format
uv run pytest                # tests
uv run pre-commit run -a     # every hook against the whole tree
```

Ruff is configured in `pyproject.toml` (88 columns, `py311` target,
pycodestyle/pyflakes/isort/pyupgrade/bugbear/comprehensions/simplify/ruff
rules). Data tables that are more readable hand-wrapped can be fenced with
`# fmt: off` / `# fmt: on`, as in `benchmarks/ifeval_instructions.py`.

## CI

Two workflows run on every pull request and on pushes to `main`:

`.github/workflows/lint.yml`

| Job | What it does |
| --- | --- |
| `ruff` | `ruff check` and `ruff format --check` |

`.github/workflows/tests.yml`

| Job | What it does |
| --- | --- |
| `test` | `pytest` on Python 3.11, 3.12 and 3.13 |
| `smoke` | `benchkit --list` plus a `--demo --headless` run, asserting `results.{json,csv,md,html}` are written; uploads them as an artifact |
| `build` | `uv build` and an import check against the built wheel |

The smoke job is the cheap end-to-end guard: it exercises the engine, the
offline demo client and the report writers without needing an inference server.

## Before opening a PR

- Run `uv run pytest` and `uv run pre-commit run -a`.
- For scoring or parser changes, validate against a small benchmark slice first
  (`uv run benchkit --headless --models MODEL --benchmarks quickbench:5 -v`).
- For TUI changes, `uv run benchkit --demo` is the fastest check.
- Keep commits focused, with short imperative subjects.
- Don't commit `.env`, local caches, or generated `results/` artifacts.
