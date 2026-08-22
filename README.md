<div align="center">

<img src="assets/benchkit-connect.png" alt="BenchKit" width="880">

# BenchKit

**Benchmark local LLMs with real evaluation suites from a full terminal UI.**

Not vibes. Actual scores.

<img src="https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white&labelColor=0b0b0b" alt="Python 3.11+">
<img src="https://img.shields.io/badge/TUI-Textual-60A5FA?style=flat-square&labelColor=0b0b0b" alt="Built with Textual">
<img src="https://img.shields.io/badge/Suites-21-34D399?style=flat-square&labelColor=0b0b0b" alt="21 benchmark suites">
<img src="https://img.shields.io/badge/License-Apache%202.0-6B7280?style=flat-square&labelColor=0b0b0b" alt="Apache 2.0">

</div>

## What it does

BenchKit runs established coding, reasoning, knowledge, instruction-following,
and long-context benchmarks against local models. It supports OpenAI-compatible
servers such as llama.cpp, llama-swap, vLLM, and LM Studio, plus native Ollama.

- Guided terminal UI for connecting, selecting models, running suites, and
  exploring results
- Headless mode for scripts and CI
- Live streamed responses, reasoning traces, progress, speed, and loop detection
- Direct model evaluation and optional Pi coding-agent evaluation
- Benchmark slicing, tag filters, verifier repair, and choice-order robustness
- JSON, CSV, Markdown, and standalone interactive HTML reports
- Performance profiling and a local dashboard for historical runs

<div align="center">
  <img src="assets/benchkit-run.png" alt="BenchKit running a benchmark" width="940">
</div>

## Quick start

BenchKit requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
uv run benchkit
```

Set your server in `.env`, or enter it on the Connect screen:

```env
BENCHKIT_PROVIDER=openai
BENCHKIT_HOST=http://localhost:8080/v1
BENCHKIT_API_KEY=
```

For Ollama:

```env
BENCHKIT_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
```

No inference server available? Explore the complete UI offline:

```bash
uv run benchkit --demo
```

## Run benchmarks

The TUI lets you select models, benchmarks, task limits, and optional run modes.
For automation, use headless mode:

```bash
# Run the first 20 HumanEval tasks
uv run benchkit --headless --models qwen3:8b --benchmarks humaneval:20

# Run multiple models and benchmarks
uv run benchkit --headless \
  --models qwen3:8b,gemma3:12b \
  --benchmarks sanity,gsm8k:50

# Print prompts, reasoning traces, and responses
uv run benchkit --headless --models qwen3:8b \
  --benchmarks sanity --verbose
```

Task slices accept `20` for the first 20 tasks, `-20` for the last 20, and
`40-80` for a range.

Useful commands:

```bash
uv run benchkit --list
uv run benchkit --help
```

## Benchmarks

| Benchmark | Key | Tasks | What it tests |
| --- | --- | ---: | --- |
| Aider Polyglot | `aider-polyglot` | 225 | Repository editing across six languages with Pi |
| Git Surgery | `git-surgery` | 5 | Stateful Git operations with Pi |
| Sanity | `sanity` | 25 | Fast checks across five core capabilities |
| HumanEval | `humaneval` | 164 | Python function completion |
| HumanEval+ | `humaneval-plus` | 164 | HumanEval with expanded EvalPlus tests |
| MBPP | `mbpp` | 500 | Short Python programming tasks |
| MBPP+ | `mbpp-plus` | 378 | MBPP with expanded EvalPlus tests |
| GSM8K | `gsm8k` | 1,319 | Multi-step grade-school math |
| IFEval | `ifeval` | 541 | Verifiable instruction following |
| RULER | `ruler` | 20 × 6 | Long-context retrieval from 4K to 128K |
| XSTest | `xstest` | 450 | Safe compliance and unsafe refusal with an offline checker |
| GPQA | `gpqa` | 198 | Graduate-level science questions |
| MMLU-Pro | `mmlu-pro` | 12,032 | Reasoning across 14 knowledge categories |
| MMLU | `mmlu` | 14,042 | Academic and professional knowledge |
| ARC | `arc` | 1,172 | Grade-school science reasoning |
| OpenBookQA | `openbookqa` | 500 | Elementary science knowledge |
| WinoGrande | `winogrande` | 1,267 | Commonsense pronoun resolution |
| PIQA | `piqa` | 1,838 | Physical commonsense reasoning |
| BoolQ | `boolq` | 3,270 | Passage-based yes/no questions |
| TruthfulQA | `truthfulqa` | 817 | Resistance to common misconceptions |
| HellaSwag | `hellaswag` | 1,000 | Plausible real-world continuations |

The CLI registry is the canonical source for current counts, descriptions,
and supported perturbations:

```bash
uv run benchkit --list
```

## Advanced runs

Compare raw generation with the stock Pi coding agent (Docker required):

```bash
uv run benchkit --headless --models qwen3:8b \
  --benchmarks gsm8k:20 --harness both
```

Give incorrect answers one verifier-guided replacement attempt:

```bash
uv run benchkit --headless --models qwen3:8b \
  --benchmarks gsm8k:20 --repair-attempts 1
```

Repairs default to off and can be set from 0–10. One is a practical starting
point, but the right depth depends on what you want to measure.

Test whether multiple-choice accuracy survives reordered answer choices:

```bash
uv run benchkit --headless --models qwen3:8b \
  --benchmarks mmlu-pro:100 --perturbation choice-order
```

Profile inference performance:

```bash
uv run benchkit perf qwen3:8b
uv run benchkit perf qwen3:8b --depths minimal,4k,16k --gen 256 --reps 10
```

## Results

Every run creates a timestamped directory containing:

```text
results/<timestamp>/
├── results.json
├── results.csv
├── results.md
└── results.html
```

Open `results.html` for an interactive report with charts and task-level
prompts, responses, reasoning traces, timing, and diagnostics.

Browse completed benchmark and performance runs locally:

```bash
uv run benchkit history
```

## Development

```bash
uv sync
uv run pre-commit install
uv run ruff check .
uv run ruff format .
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor setup and workflow.

## License

Apache License 2.0

<div align="center">
<br>
<sub>Themed with <a href="https://github.com/DogukanUrker/DogiZed">Dogi</a> · built on <a href="https://github.com/Textualize/textual">Textual</a></sub>
</div>
