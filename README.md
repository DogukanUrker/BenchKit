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

| Model                | Benchmark | Score | Passed | Total | tok/s | Avg Resp | Total Time |
| -------------------- | --------- | ----- | ------ | ----- | ----- | -------- | ---------- |
| gemma3:270m          | humaneval | 11.6% | 19     | 164   | 403.5 | 0.7s     | 2m 30s     |
| gemma3:1b            | humaneval | 29.9% | 49     | 164   | 217.5 | 0.7s     | 2m 0s      |
| llama3.2:1b          | humaneval | 26.2% | 43     | 164   | 223.6 | 0.4s     | 1m 22s     |
| smollm2:1.7b         | humaneval | 26.8% | 44     | 164   | 157.0 | 0.5s     | 1m 32s     |
| llama3.2:3b          | humaneval | 42.7% | 70     | 164   | 135.3 | 0.7s     | 1m 52s     |
| phi4-mini:3.8b       | humaneval | 51.2% | 84     | 164   | 114.0 | 0.9s     | 2m 37s     |
| ministral-3:3b       | humaneval | 54.3% | 89     | 164   | 124.1 | 0.8s     | 2m 27s     |
| gemma3:4b            | humaneval | 59.8% | 98     | 164   | 97.4  | 2.5s     | 6m 52s     |
| mistral:7b           | humaneval | 31.7% | 52     | 164   | 70.6  | 1.3s     | 3m 47s     |
| llama3.1:8b          | humaneval | 57.9% | 95     | 164   | 66.3  | 1.1s     | 3m 9s      |
| ministral-3:8b       | humaneval | 67.7% | 111    | 164   | 60.6  | 1.5s     | 4m 5s      |
| mistral-nemo:12b     | humaneval | 47.6% | 78     | 164   | 47.5  | 1.3s     | 3m 38s     |
| gemma3:12b           | humaneval | 81.7% | 134    | 164   | 39.7  | 5.7s     | 15m 46s    |
| phi4:14b             | humaneval | 75.0% | 123    | 164   | 36.4  | 2.0s     | 5m 38s     |
| ministral-3:14b      | humaneval | 79.9% | 131    | 164   | 38.7  | 2.2s     | 6m 1s      |
| devstral-small-2:24b | humaneval | 76.8% | 126    | 164   | 8.8   | 8.5s     | 23m 20s    |
| mistral-small3.2:24b | humaneval | 78.7% | 129    | 164   | 8.9   | 8.9s     | 24m 34s    |
| gemma3:27b           | humaneval | 85.4% | 140    | 164   | 6.2   | 35.0s    | 95m 43s    |

> [!NOTE]
> Qwen3/3.5 models frequently fail to exit thinking mode on complex tasks, producing correct reasoning internally but no parseable output. Scores reflect real-world usability, not theoretical capability.

## License

Apache License 2.0
