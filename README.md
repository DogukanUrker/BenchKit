<div align="center">

<img src="assets/benchkit-connect.png" alt="BenchKit" width="880">

# BenchKit

**Benchmark local LLMs with real evaluation suites — from a full terminal UI.**

Not vibes. Actual scores.

<img src="https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white&labelColor=0b0b0b" alt="Python 3.11+">
<img src="https://img.shields.io/badge/TUI-Textual-60A5FA?style=flat-square&labelColor=0b0b0b" alt="Built with Textual">
<img src="https://img.shields.io/badge/Suites-15-34D399?style=flat-square&labelColor=0b0b0b" alt="15 benchmark suites">
<img src="https://img.shields.io/badge/License-Apache%202.0-6B7280?style=flat-square&labelColor=0b0b0b" alt="Apache 2.0">

</div>

---

## Quick start

```bash
uv sync
uv run benchkit
```

No inference server around? Explore the whole interface offline:

```bash
uv run benchkit --demo
```

BenchKit talks to any OpenAI-compatible server — llama.cpp, llama-swap, vLLM,
LM Studio — as well as a native Ollama host.

## The app

<div align="center">
  <img src="assets/benchkit-run.png" alt="BenchKit running a benchmark" width="940">
</div>

| Screen      | What happens there                                                                                                                                                                                         |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Connect** | Host, provider, API key and timeout are editable in place. Auto-connects from your `.env`, shows the real error when it fails, retries on `Ctrl+R`.                                                        |
| **Setup**   | Multi-select panes for models and benchmarks with live filters, task counts, and a task limit set globally or per benchmark. A summary line keeps score: `3 models × 2 benchmarks = 6 runs · 1,240 tasks`. |
| **Run**     | Per-job and overall progress, live accuracy / tok-s / latency / elapsed / ETA, the run queue with per-job scores, and a task table that streams in. Pause, skip a job or stop at any point.                |
| **Results** | Sortable summary, drill-down into every task with a pass/fail/search filter, and the path to the saved reports.                                                                                            |

Press `Enter` on any task — during the run or afterwards — to read the exact
prompt and the model's raw response.

## Keys

| Key             | Action                                       |
| --------------- | -------------------------------------------- |
| `?` / `F1`      | Keyboard reference                           |
| `Space`         | Toggle the highlighted model or benchmark    |
| `a` / `n` / `i` | Select all / clear / invert the focused list |
| `l`             | Task limit for the highlighted benchmark     |
| `/`             | Jump to the filter box                       |
| `s` / `F5`      | Start the run                                |
| `p` / `k` / `x` | Pause / skip job / stop during a run         |
| `f`             | Failures only                                |
| `Enter`         | Inspect the highlighted row                  |
| `F2`            | Dogi light / dark                            |
| `Ctrl+P`        | Command palette                              |
| `Ctrl+Q`        | Quit                                         |

Task limits accept `20` (first 20), `-20` (last 20) and `40-80` (a range).

## Headless

For CI and scripts, skip the TUI entirely:

```bash
uv run benchkit --headless --models qwen3:8b,gemma3:12b --benchmarks humaneval:20,gsm8k
uv run benchkit --headless --models all --benchmarks quickbench --verbose
uv run benchkit --list
```

`--verbose` prints every prompt and response. Reports are saved exactly as they
are from the TUI, and a run that fails part-way still keeps what finished.

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

```env
BENCHKIT_PROVIDER=openai
BENCHKIT_HOST=http://hub:11434/v1
BENCHKIT_HARDWARE=RTX 3060 12GB
```

The `/v1` suffix is optional. BenchKit uses `GET /v1/models` for discovery and
`POST /v1/chat/completions` for benchmark requests. API keys are supported with
`BENCHKIT_API_KEY`. Set `BENCHKIT_PROVIDER=auto` to try OpenAI compatibility
first and fall back to Ollama's native API.

Existing Ollama configuration remains supported:

```env
BENCHKIT_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
```

The per-request timeout defaults to 300 seconds and can be changed with
`BENCHKIT_TIMEOUT`.

## Benchmarks

| Benchmark  | Key              |  Tasks | What it tests                                                       |
| ---------- | ---------------- | -----: | ------------------------------------------------------------------- |
| QuickBench | `quickbench`     |     20 | Tiny Python tasks for a fast end-to-end sanity check                |
| HumanEval  | `humaneval`      |    164 | Python function completion with the original unit tests             |
| HumanEval+ | `humaneval-plus` |    164 | HumanEval with more than 122,000 tougher EvalPlus test inputs       |
| MBPP       | `mbpp`           |    500 | Short Python functions from natural-language specifications         |
| MBPP+      | `mbpp-plus`      |    378 | Sanitized MBPP with more than 39,000 EvalPlus test inputs           |
| GSM8K      | `gsm8k`          |  1,319 | Multi-step grade-school math with exact numeric answers             |
| ARC        | `arc`            |  1,172 | Challenging grade-school science multiple choice                    |
| GPQA       | `gpqa`           |    198 | Expert-written graduate science questions designed to resist search |
| MMLU       | `mmlu`           | 14,042 | Zero-shot coverage of 57 academic and professional subjects         |
| OpenBookQA | `openbookqa`     |    500 | Elementary science requiring factual knowledge and reasoning        |
| WinoGrande | `winogrande`     |  1,267 | Commonsense pronoun resolution in ambiguous sentences               |
| PIQA       | `piqa`           |  1,838 | Physical plausibility of solutions to everyday tasks                |
| BoolQ      | `boolq`          |  3,270 | Yes/no questions answered from evidence in a passage                |
| TruthfulQA | `truthfulqa`     |    817 | Resistance to common misconceptions and false beliefs               |
| HellaSwag  | `hellaswag`      |  1,000 | Plausible continuations of real-world scenarios                     |

More coming soon.

HumanEval+ and MBPP+ use the complete official EvalPlus datasets and both the
base and expanded test inputs. The upstream datasets are downloaded and cached
on first use. Generated code is untrusted; EvalPlus recommends running its
evaluator in Docker for strong isolation, while BenchKit's integrated runner
uses EvalPlus's guarded local subprocess evaluator for interactive per-task
progress.

## Output

Each run creates a timestamped folder in `results/`:

```
results/2026-03-21_14-30-00/
├── results.json   # Full results with per-task details
├── results.csv    # Summary table
├── results.md     # Markdown table (paste into GitHub)
└── results.html   # Interactive, self-contained visual report
```

Open `results.html` directly in a browser. It carries the same Dogi theme as the
TUI, opens light, and has a toggle in the header. Inside is a gallery of
screenshot-ready charts — overall and per-benchmark accuracy, a full suite
overview, accuracy versus speed, raw generation speed, pass/fail composition,
total runtime and a score matrix — followed by every task's prompt and response.
No server or external web assets are required.

## Leaderboard

All results from a single RTX 3060 12GB system. Models run via Ollama with
default quantization (Q4_K_M) and a 5-minute timeout per task.

If a model solves a problem inside `<think>` tags but fails to produce parseable
output, it scores 0. This reflects real-world pipeline behavior — downstream
tools receive model output as-is.

### HumanEval (164 tasks)

|   # | Model                | Params |     Score |  Passed | tok/s |     Time |  Offload |
| --: | -------------------- | -----: | --------: | ------: | ----: | -------: | -------: |
|   1 | gpt-oss:20b          |    20B | **95.1%** | 156/164 |  27.5 |  59m 29s | ~84% GPU |
|   2 | deepseek-r1:14b      |    14B | **83.5%** | 137/164 |  33.1 | 207m 58s | 100% GPU |
|   3 | gemma3:12b           |    12B | **81.7%** | 134/164 |  39.7 |  15m 46s | 100% GPU |
|   4 | ministral-3:14b      |    14B | **79.9%** | 131/164 |  38.7 |    6m 1s | 100% GPU |
|   5 | gemma4:e2b           |   2.3B | **79.3%** | 130/164 | 103.2 |  41m 26s | 100% GPU |
|   6 | phi4:14b             |    14B | **75.0%** | 123/164 |  36.4 |   5m 38s | 100% GPU |
|   7 | qwen3:8b             |     8B | **73.8%** | 121/164 |  55.8 |  166m 9s | 100% GPU |
|   8 | gemma4:e4b           |   4.5B | **68.9%** | 113/164 |  69.7 |  50m 30s | 100% GPU |
|   9 | qwen3:14b            |    14B | **68.3%** | 112/164 |  32.3 | 201m 52s | 100% GPU |
|  10 | ministral-3:8b       |     8B | **67.7%** | 111/164 |  60.6 |    4m 5s | 100% GPU |
|  11 | granite4:3b          |     3B | **65.9%** | 108/164 | 116.8 |    2m 2s | 100% GPU |
|  12 | glm4:9b              |     9B | **64.0%** | 105/164 |  61.2 |   4m 25s | 100% GPU |
|  13 | deepseek-r1:7b       |     7B | **61.6%** | 101/164 |  63.9 | 202m 33s | 100% GPU |
|  14 | gemma3:4b            |     4B | **59.8%** |  98/164 |  97.4 |   6m 52s | 100% GPU |
|  15 | gemma4:12b           |    12B | **58.5%** |  96/164 |  35.9 | 199m 34s | 100% GPU |
|  16 | lfm2.5-thinking:1.2b |   1.2B | **58.5%** |  96/164 | 302.1 |  23m 48s | 100% GPU |
|  17 | llama3.1:8b          |     8B | **57.9%** |  95/164 |  66.3 |    3m 9s | 100% GPU |
|  18 | ministral-3:3b       |     3B | **54.3%** |  89/164 | 124.1 |   2m 27s | 100% GPU |
|  19 | phi4-mini:3.8b       |   3.8B | **51.2%** |  84/164 | 114.0 |   2m 37s | 100% GPU |
|  20 | mistral-nemo:12b     |    12B | **47.6%** |  78/164 |  47.5 |   3m 38s | 100% GPU |
|  21 | deepseek-r1:1.5b     |   1.5B | **42.7%** |  70/164 | 180.7 | 213m 38s | 100% GPU |
|  22 | llama3.2:3b          |     3B | **42.7%** |  70/164 | 135.3 |   1m 52s | 100% GPU |
|  23 | mistral:7b           |     7B | **31.7%** |  52/164 |  70.6 |   3m 47s | 100% GPU |
|  24 | gemma3:1b            |     1B | **29.9%** |  49/164 | 217.5 |    2m 0s | 100% GPU |
|  25 | smollm2:1.7b         |   1.7B | **26.8%** |  44/164 | 157.0 |   1m 32s | 100% GPU |
|  26 | llama3.2:1b          |     1B | **26.2%** |  43/164 | 223.6 |   1m 22s | 100% GPU |
|  27 | qwen3.5:2b           |     2B | **13.4%** |  22/164 | 107.4 |  236m 6s | 100% GPU |
|  28 | gemma3:270m          |   270M | **11.6%** |  19/164 | 403.5 |   2m 30s | 100% GPU |
|  29 | qwen3.5:9b           |     9B |  **7.9%** |  13/164 |  45.9 | 371m 48s | 100% GPU |
|  30 | qwen3.5:0.8b         |   0.8B |  **6.7%** |  11/164 | 171.7 | 252m 36s | 100% GPU |
|  31 | qwen3.5:4b           |     4B |  **1.2%** |   2/164 |  65.9 | 272m 50s | 100% GPU |

<details>
<summary><b>On thinking models</b></summary>

<br>

Models with reasoning capabilities (Qwen3, DeepSeek-R1) wrap chain-of-thought in
`<think>` tags before producing a final answer. gpt-oss:20b also uses thinking
tags but handles them correctly, scoring 95.1%.

**Tag closure failure** — Qwen3.5 models consistently fail to close the
`</think>` tag, producing correct reasoning but no usable output. Scores: 0.8b
(6.7%), 2b (13.4%), 4b (1.2%), 9b (7.9%) — all taking 4-6 hours each. Larger
models think harder and fail more. Qwen3 (non-.5) handles tags better —
qwen3:8b scored 73.8% — but still takes 40x longer than non-thinking models at
similar accuracy.

**Token overhead** — DeepSeek-R1 closes its tags but generates thousands of
thinking tokens per task. deepseek-r1:1.5b and llama3.2:3b both score 42.7% on
HumanEval — one takes 213 minutes, the other takes 1 minute 52 seconds.

Both behaviors break automated pipelines where downstream tools expect clean,
fast responses.

</details>

<details>
<summary><b>Hardware</b></summary>

<br>

| Component | Spec                      |
| --------- | ------------------------- |
| CPU       | AMD Ryzen 5 5600 (6C/12T) |
| GPU       | NVIDIA RTX 3060 12GB      |
| RAM       | 16GB DDR4                 |
| Swap      | 16GB (SATA SSD)           |
| OS        | Debian 13                 |
| Runtime   | Ollama                    |

</details>

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

<div align="center">
<br>
<sub>Themed with <a href="https://github.com/DogukanUrker/DogiZed">Dogi</a> · built on <a href="https://github.com/Textualize/textual">Textual</a></sub>
</div>
