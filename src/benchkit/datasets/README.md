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

[afl-3.0]: https://opensource.org/license/afl-3-0-php
[cc-by-2.0]: https://creativecommons.org/licenses/by/2.0/
[cc-by-sa-3.0]: https://creativecommons.org/licenses/by-sa/3.0/
