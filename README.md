# BenchKit

Benchmark your local LLMs with real evaluation suites. Not vibes - actual scores.

## Install

```bash
uv sync
```

## Usage

```bash
uv run benchkit
```

To see raw prompts and model responses for every task (useful for debugging failures):

```bash
uv run benchkit --verbose
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

| Benchmark  | Tasks | What it tests                    |
| ---------- | ----- | -------------------------------- |
| QuickBench | 20    | Fast code-generation smoke test  |
| HumanEval  | 164   | Code generation (pass@1)         |
| MBPP       | 500   | Python programming tasks         |
| GSM8K      | 1319  | Math reasoning with answer parse |
| ARC        | 1172  | Science multiple choice QA       |
| TruthfulQA | 817   | Truthfulness multiple choice QA  |
| HellaSwag  | 1000  | Commonsense sentence completion  |

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

## Leaderboard

| Model           | Benchmark | Score | Passed | Total | tok/s |
| --------------- | --------- | ----- | ------ | ----- | ----- |
| gemma3:270m     | humaneval | 11.6% | 19     | 164   | 404.0 |
| gemma3:1b       | humaneval | 29.9% | 49     | 164   | 217.0 |
| qwen3.5:0.8b    | humaneval | 0.0%  | 0      | 164   | 176.3 |
| llama3.2:1b     | humaneval | 26.2% | 43     | 164   | 223.3 |
| qwen3:1.7b      | humaneval | 0.0%  | 0      | 164   | 188.3 |
| llama3.2:3b     | humaneval | 42.7% | 70     | 164   | 135.1 |
| phi4-mini:3.8b  | humaneval | 49.4% | 81     | 164   | 113.9 |
| qwen3.5:2b      | humaneval | 0.0%  | 0      | 164   | 109.6 |
| ministral-3:3b  | humaneval | 50.0% | 82     | 164   | 124.2 |
| gemma3:4b       | humaneval | 28.0% | 46     | 164   | 98.2  |
| qwen3.5:4b      | humaneval | 0.0%  | 0      | 164   | 68.3  |
| qwen2.5:7b      | humaneval | 20.7% | 34     | 164   | 73.4  |
| qwen3:8b        | humaneval | 5.5%  | 9      | 164   | 59.7  |
| ministral-3:8b  | humaneval | 68.9% | 113    | 164   | 60.6  |
| qwen3.5:9b      | humaneval | 1.8%  | 3      | 164   | 47.3  |
| gemma3:12b      | humaneval | 81.1% | 133    | 164   | 39.6  |
| ministral-3:14b | humaneval | 78.7% | 129    | 164   | 38.7  |
| gpt-oss:20b     | humaneval | 53.0% | 87     | 164   | 27.7  |

> [!NOTE]
> Qwen3/3.5 models frequently fail to exit thinking mode on complex tasks, producing correct reasoning internally but no parseable output. Scores reflect real-world usability, not theoretical capability.

## License

Apache License 2.0
