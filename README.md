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

## Performance profiles

Profile a model served by llama.cpp or llama-swap through its OpenAI-compatible
endpoint:

```bash
uv run benchkit perf ornith-35b
```

The default sweep covers minimal, 4k, 16k, 32k and 64k contexts with one
discarded warmup and five measured 128-token generations at each depth. It
reports prompt processing (PP), token generation (TG), time to first token
(TTFT), wall time, proxy/client overhead and the first-request latency that may
include llama-swap model loading. When llama-swap does not expose llama.cpp's
native tokenizer route, BenchKit calibrates deterministic prompt text against
the input-token count reported by the server instead of relying on a fixed
characters-per-token estimate.

Defaults can be narrowed or expanded when needed:

```bash
uv run benchkit perf ornith-35b --depths minimal,4k,16k --gen 256 --reps 10
```

Each run writes `perf.json`, `perf.csv`, `perf.md` and a standalone
`perf.html` dashboard under `results/<timestamp>/`. Add an optional reproducible
configuration line with `BENCHKIT_PERF_CONFIG` in `.env` or `--config-note`;
`BENCHKIT_HARDWARE` is used as a fallback. The first context is measured again
at the end of the sweep to flag GPU warmup or performance drift.

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

```env
BENCHKIT_PROVIDER=openai
BENCHKIT_HOST=http://hub:11434/v1
BENCHKIT_HARDWARE=RTX 3060 12GB
# Optional details shown in performance reports:
BENCHKIT_PERF_CONFIG=RTX 3060 12GB | q8_0 KV | -np 1
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
