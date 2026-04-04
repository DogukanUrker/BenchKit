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
|   1 | gpt-oss:20b          |    20B | **95.1%** | 156/164 |  27.5 |  59m 29s | ~84% GPU |
|   2 | gemma3:27b           |    27B | **85.4%** | 140/164 |   6.2 |  95m 43s | ~60% GPU |
|   3 | deepseek-r1:14b      |    14B | **83.5%** | 137/164 |  33.1 | 207m 58s | 100% GPU |
|   4 | gemma3:12b           |    12B | **81.7%** | 134/164 |  39.7 |  15m 46s | 100% GPU |
|   5 | ministral-3:14b      |    14B | **79.9%** | 131/164 |  38.7 |    6m 1s | 100% GPU |
|   6 | gemma4:e2b           |   2.3B | **79.3%** | 130/164 | 103.2 |  41m 26s | 100% GPU |
|   7 | mistral-small3.2:24b |    24B | **78.7%** | 129/164 |   8.9 |  24m 34s | ~70% GPU |
|   8 | devstral-small-2:24b |    24B | **76.8%** | 126/164 |   8.8 |  23m 20s | ~70% GPU |
|   9 | phi4:14b             |    14B | **75.0%** | 123/164 |  36.4 |   5m 38s | 100% GPU |
|  10 | gemma4:31b           |    31B | **74.4%** | 122/164 |   3.9 | 901m 19s | ~50% GPU |
|  11 | qwen3:8b             |     8B | **73.8%** | 121/164 |  55.8 |  166m 9s | 100% GPU |
|  12 | gemma4:26b           |    26B | **73.2%** | 120/164 |  21.7 | 536m 59s | ~60% GPU |
|  13 | gemma4:e4b           |   4.5B | **68.9%** | 113/164 |  69.7 |  50m 30s | 100% GPU |
|  14 | qwen3:14b            |    14B | **68.3%** | 112/164 |  32.3 | 201m 52s | 100% GPU |
|  15 | ministral-3:8b       |     8B | **67.7%** | 111/164 |  60.6 |    4m 5s | 100% GPU |
|  16 | glm4:9b              |     9B | **64.0%** | 105/164 |  61.2 |   4m 25s | 100% GPU |
|  17 | deepseek-r1:7b       |     7B | **61.6%** | 101/164 |  63.9 | 202m 33s | 100% GPU |
|  18 | gemma3:4b            |     4B | **59.8%** |  98/164 |  97.4 |   6m 52s | 100% GPU |
|  19 | lfm2.5-thinking:1.2b |   1.2B | **58.5%** |  96/164 | 302.1 |  23m 48s | 100% GPU |
|  20 | llama3.1:8b          |     8B | **57.9%** |  95/164 |  66.3 |    3m 9s | 100% GPU |
|  21 | ministral-3:3b       |     3B | **54.3%** |  89/164 | 124.1 |   2m 27s | 100% GPU |
|  22 | phi4-mini:3.8b       |   3.8B | **51.2%** |  84/164 | 114.0 |   2m 37s | 100% GPU |
|  23 | mistral-nemo:12b     |    12B | **47.6%** |  78/164 |  47.5 |   3m 38s | 100% GPU |
|  24 | deepseek-r1:1.5b     |   1.5B | **42.7%** |  70/164 | 180.7 | 213m 38s | 100% GPU |
|  25 | llama3.2:3b          |     3B | **42.7%** |  70/164 | 135.3 |   1m 52s | 100% GPU |
|  26 | mistral:7b           |     7B | **31.7%** |  52/164 |  70.6 |   3m 47s | 100% GPU |
|  27 | gemma3:1b            |     1B | **29.9%** |  49/164 | 217.5 |    2m 0s | 100% GPU |
|  28 | smollm2:1.7b         |   1.7B | **26.8%** |  44/164 | 157.0 |   1m 32s | 100% GPU |
|  29 | llama3.2:1b          |     1B | **26.2%** |  43/164 | 223.6 |   1m 22s | 100% GPU |
|  30 | qwen3.5:2b           |     2B | **13.4%** |  22/164 | 107.4 |  236m 6s | 100% GPU |
|  31 | gemma3:270m          |   270M | **11.6%** |  19/164 | 403.5 |   2m 30s | 100% GPU |
|  32 | qwen3.5:9b           |     9B |  **7.9%** |  13/164 |  45.9 | 371m 48s | 100% GPU |
|  33 | qwen3.5:0.8b         |   0.8B |  **6.7%** |  11/164 | 171.7 | 252m 36s | 100% GPU |
|  34 | qwen3.5:4b           |     4B |  **1.2%** |   2/164 |  65.9 | 272m 50s | 100% GPU |

### On thinking models

Models with reasoning capabilities (Qwen3, DeepSeek-R1) wrap chain-of-thought in `<think>` tags before producing a final answer. gpt-oss:20b also uses thinking tags but handles them correctly, scoring 95.1%.

**Tag closure failure** - Qwen3.5 models consistently fail to close the `</think>` tag, producing correct reasoning but no usable output. Scores: 0.8b (6.7%), 2b (13.4%), 4b (1.2%), 9b (7.9%) - all taking 4-6 hours each. Larger models think harder and fail more. Qwen3 (non-.5) handles tags better - qwen3:8b scored 73.8% - but still takes 40x longer than non-thinking models at similar accuracy.

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
