# Dataset attribution

## GPQA Diamond

`gpqa_diamond.jsonl` is a transformed copy of the 198-question GPQA Diamond
split by David Rein et al. The source answer choices were shuffled with the
official evaluator's default seed (`0`) and converted from CSV to BenchKit's
JSONL schema.

- Source: <https://github.com/idavidrein/gpqa>
- Paper: <https://arxiv.org/abs/2311.12022>
- Dataset license: [Creative Commons Attribution 4.0 International][cc-by-4.0]

[cc-by-4.0]: https://creativecommons.org/licenses/by/4.0/

## XSTest

`xstest.csv` is the complete 450-prompt XSTest suite: 250 safe prompts across
ten prompt types and 200 unsafe contrast prompts. BenchKit uses the authors'
offline string-matching evaluator, which is less reliable than manual or
LLM-based classification and cannot identify partial refusals.

- Source: <https://github.com/paul-rottger/xstest>
- Paper: <https://aclanthology.org/2024.naacl-long.301/>
- Dataset license: [Creative Commons Attribution 4.0 International][cc-by-4.0]

## WinoGrande

`winogrande.jsonl` is a transformed copy of the 1,267-example labeled
development split from WinoGrande 1.1.

- Source: <https://github.com/allenai/winogrande>
- Dataset license: [Creative Commons Attribution 2.0][cc-by-2.0]

## PIQA

`piqa.jsonl` is a transformed copy of the 1,838-example labeled validation
split from Physical Interaction: Question Answering.

- Source: <https://yonatanbisk.com/piqa/data/>
- Dataset license: [Academic Free License 3.0][afl-3.0]

## MMLU

`mmlu.jsonl` is a transformed copy of the 14,042-question test split covering
all 57 MMLU subjects. BenchKit evaluates it zero-shot.

- Source: <https://github.com/hendrycks/test>
- Dataset license: MIT; see `MMLU_LICENSE.txt`

## MMLU-Pro

`mmlu_pro.jsonl` is a transformed copy of the MMLU-Pro test split: ten-option
questions across 14 categories, filtered by the authors for ones that require
reasoning. Only the schema changed - `question_id` became a namespaced
`task_id` and `options` became `choices`, with the `N/A` padding left by the
authors' pruning pass dropped, which is why some questions carry fewer than ten
choices. Regenerate it with `uv run --with datasets python
scripts/build_mmlu_pro.py`. BenchKit evaluates it zero-shot.

- Source: <https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro>
- Code: <https://github.com/TIGER-AI-Lab/MMLU-Pro>
- Paper: <https://arxiv.org/abs/2406.01574>
- Dataset license: MIT

## OpenBookQA

`openbookqa.jsonl` is a transformed copy of the 500-question OpenBookQA test
split.

- Source: <https://github.com/allenai/OpenBookQA>
- Dataset license: Apache 2.0

## BoolQ

`boolq.jsonl` is a transformed copy of the 3,270-example labeled validation
split distributed with SuperGLUE.

- Source: <https://github.com/google-research-datasets/boolean-questions>
- Dataset license: [Creative Commons Attribution-ShareAlike 3.0][cc-by-sa-3.0]

## IFEval

`ifeval.jsonl` is a transformed copy of the 541 prompts distributed with the
IFEval paper. Only the schema changed: `key` became `task_id`, and the
`instruction_id_list` / `kwargs` pairs are carried over unmodified.

- Source: <https://github.com/google-research/google-research/tree/master/instruction_following_eval>
- Paper: <https://arxiv.org/abs/2311.07911>
- Dataset license: Apache 2.0

## RULER

RULER deterministically generates its NIAH, variable-tracking, CWE, and FWE
tasks. Essay-backed NIAH uses upstream `PaulGrahamEssays.json`; QA-1 and QA-2
use SQuAD 2.0 development and HotpotQA distractor validation data cached under
`~/.cache/benchkit/ruler`. BenchKit downloads the QA files on first use. The
essay corpus is generated with NVIDIA/RULER's
`download_paulgraham_essay.py`.
retrieval and variable-tracking prompts at runtime for each context bucket.

- Source design: <https://github.com/NVIDIA/RULER>
- Paper: <https://arxiv.org/abs/2404.06654>
- Reference implementation license: Apache 2.0

[afl-3.0]: https://opensource.org/license/afl-3-0-php
[cc-by-2.0]: https://creativecommons.org/licenses/by/2.0/
[cc-by-sa-3.0]: https://creativecommons.org/licenses/by-sa/3.0/
