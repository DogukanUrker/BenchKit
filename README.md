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

## Leaderboard

All results from a single RTX 3060 12GB system. Models run via Ollama with default quantization (Q4_K_M) and a 5-minute timeout per task.

If a model solves a problem inside `<think>` tags but fails to produce parseable output, it scores 0. This reflects real-world pipeline behavior - downstream tools receive model output as-is.

### HumanEval (164 tasks)

|   # | Model                | Params |     Score |  Passed | tok/s |     Time |  Offload |
| --: | -------------------- | -----: | --------: | ------: | ----: | -------: | -------: |
|   1 | gemma3:27b           |    27B | **85.4%** | 140/164 |   6.2 |  95m 43s | ~60% GPU |
|   2 | gemma3:12b           |    12B | **81.7%** | 134/164 |  39.7 |  15m 46s | 100% GPU |
|   3 | ministral-3:14b      |    14B | **79.9%** | 131/164 |  38.7 |    6m 1s | 100% GPU |
|   4 | mistral-small3.2:24b |    24B | **78.7%** | 129/164 |   8.9 |  24m 34s | ~70% GPU |
|   5 | devstral-small-2:24b |    24B | **76.8%** | 126/164 |   8.8 |  23m 20s | ~70% GPU |
|   6 | phi4:14b             |    14B | **75.0%** | 123/164 |  36.4 |   5m 38s | 100% GPU |
|   7 | ministral-3:8b       |     8B | **67.7%** | 111/164 |  60.6 |    4m 5s | 100% GPU |
|   8 | gemma3:4b            |     4B | **59.8%** |  98/164 |  97.4 |   6m 52s | 100% GPU |
|   9 | lfm2.5-thinking:1.2b |   1.2B | **58.5%** |  96/164 | 302.1 |  23m 48s | 100% GPU |
|  10 | llama3.1:8b          |     8B | **57.9%** |  95/164 |  66.3 |    3m 9s | 100% GPU |
|  11 | ministral-3:3b       |     3B | **54.3%** |  89/164 | 124.1 |   2m 27s | 100% GPU |
|  12 | phi4-mini:3.8b       |   3.8B | **51.2%** |  84/164 | 114.0 |   2m 37s | 100% GPU |
|  13 | mistral-nemo:12b     |    12B | **47.6%** |  78/164 |  47.5 |   3m 38s | 100% GPU |
|  14 | deepseek-r1:1.5b     |   1.5B | **42.7%** |  70/164 | 180.7 | 213m 38s | 100% GPU |
|  15 | llama3.2:3b          |     3B | **42.7%** |  70/164 | 135.3 |   1m 52s | 100% GPU |
|  16 | mistral:7b           |     7B | **31.7%** |  52/164 |  70.6 |   3m 47s | 100% GPU |
|  17 | gemma3:1b            |     1B | **29.9%** |  49/164 | 217.5 |    2m 0s | 100% GPU |
|  18 | smollm2:1.7b         |   1.7B | **26.8%** |  44/164 | 157.0 |   1m 32s | 100% GPU |
|  19 | llama3.2:1b          |     1B | **26.2%** |  43/164 | 223.6 |   1m 22s | 100% GPU |
|  20 | gemma3:270m          |   270M | **11.6%** |  19/164 | 403.5 |   2m 30s | 100% GPU |

Thinking models pending: deepseek-r1 (7b, 14b), glm4:9b, gpt-oss:20b, qwen3 (8b, 14b), qwen3.5 (0.8b, 2b, 4b, 9b)

### On thinking models

Models with reasoning capabilities (Qwen3, DeepSeek-R1, gpt-oss) wrap chain-of-thought in `<think>` tags before producing a final answer.

**Tag closure failure** - Qwen models frequently never close the `</think>` tag on complex tasks. The model reasons correctly but produces no usable output. Every Qwen model tested in earlier runs scored near 0%.

**Token overhead** - DeepSeek-R1 closes its tags but generates thousands of thinking tokens per task. deepseek-r1:1.5b and llama3.2:3b both score 42.7% on HumanEval - one takes 213 minutes, the other takes 1 minute 52 seconds.

Both behaviors break automated pipelines where downstream tools expect clean, fast responses.

### Hardware

| Component | Spec                      |
| --------- | ------------------------- |
| CPU       | AMD Ryzen 5 5600 (6C/12T) |
| GPU       | NVIDIA RTX 3060 12GB      |
| RAM       | 16GB DDR4                 |
| Swap      | 16GB (SATA SSD)           |
| OS        | Debian 13                 |
| Runtime   | Ollama                    |

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
