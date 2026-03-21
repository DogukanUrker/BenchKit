# BenchKit

Benchmark your local LLMs with real evaluation suites. Not vibes — actual scores.

## Install

```bash
uv sync
```

## Usage

```bash
uv run benchkit
```

That's it. BenchKit will:

1. Connect to your Ollama instance
2. Let you pick which models to test
3. Let you pick which benchmarks to run
4. Run everything and show results
5. Save JSON, CSV, and Markdown reports to `results/`

### Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

```env
OLLAMA_HOST=http://localhost:11434
```

Or point to a remote Ollama instance:

```bash
OLLAMA_HOST=http://my-server:11434 uv run benchkit
```

## Benchmarks

| Benchmark | Tasks | What it tests            |
| --------- | ----- | ------------------------ |
| HumanEval | 164   | Code generation (pass@1) |

More coming soon.

## Output

Each run creates a timestamped folder in `results/`:

```
results/2026-03-21_14-30-00/
├── results.json   # Full results with per-task details
├── results.csv    # Summary table
└── results.md     # Markdown table (paste into GitHub)
```

## Adding a benchmark

Create a file in `src/benchkit/benchmarks/` that implements three methods:

```python
class MyBenchmark:
    name = "mybench"

    def load_tasks(self) -> list[Task]: ...
    def build_prompt(self, task: Task) -> str: ...
    def evaluate(self, task: Task, response: str) -> bool: ...
```

Then add it to `REGISTRY` in `benchmarks/__init__.py`. Done.

## License

Apache License 2.0
