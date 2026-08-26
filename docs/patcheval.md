# PatchEval runner contract

PatchEval is split deliberately. The one-shot miner and frozen corpus live in a
separate Hugging Face dataset repository. BenchKit contains only the runner,
prompt, sandbox boundary, and deterministic grader.

## Threat model

The Pi agent receives a parent-commit source archive in a fresh `/workspace`.
It has no host mount, Docker socket, GitHub network access, real Git history,
gold patch, Hugging Face cache, or grader assets. Its only network route is the
restricted inference proxy already used by BenchKit's Pi harness.

After each attempt, BenchKit copies the workspace to the trusted host and diffs
it against a fresh extraction of the checksummed source archive. Agent Git
metadata is ignored. Changes matching `protected_globs` or `ignored_globs` are
left out of the submitted patch. The runner always protects conventional Python
test paths (`tests/`, `test/`, `test_*.py`, `*_test.py`, and `conftest.py`);
dataset `protected_globs` must cover any repository-specific test locations.
An agent may therefore write tests for itself without making those tests part
of scoring.

BenchKit then starts two new containers with `--network none`:

1. The fail-to-pass grader applies the submitted patch and the hidden test
   patch, then runs `fail_to_pass_command`.
2. The regression grader applies only the submitted patch, then runs
   `regression_command` against the parent commit's previously passing tests.

The task passes only when both exit codes are zero. A grader setup failure is a
harness error, not an incorrect answer. Raw output remains in the report for
benchmark maintainers but is never sent to the model. Repair turns receive only
the generic instruction to re-examine the issue and continue.

## Frozen dataset layout

```text
dataset.json
tasks.jsonl
SHA256SUMS
sources/
  <task-id>.tar
hidden-tests/
  <task-id>.patch
attestations/
  <task-id>.json
```

`dataset.json` uses schema version 1 and contains `release` and `task_count`.
Each line of `tasks.jsonl` contains:

- `id`, `repository`, `issue_title`, and reviewed `issue_body`
- `runtime_recipe`, containing schema version 1, an explicitly version-tagged
  Debian-compatible `base_image`, required `sync_command`, and optional
  `bootstrap_command` and `environment`
- relative `source_archive` and `hidden_test_patch` paths plus SHA-256 hashes
- argv arrays for `setup_command`, `fail_to_pass_command`, and
  `regression_command`
- `protected_globs`, `ignored_globs`, `timeout_s`, and `validated: true`

Source archives must contain repository contents at their root, exclude `.git`,
and dereference symlinks. BenchKit builds each runtime locally at benchmark
start. The generated Dockerfile has three stages: shared Pi assets on top of
the recipe's version-tagged base, the runtime that installs task dependencies
from the verified parent source and then removes that build copy, and a final
stage that adds only the task environment and the agent user. Build commands run
with network access and version tags may drift; this is an explicitly accepted
tradeoff and does not cryptographically bind the runtime to the miner's
validation image.

One `benchkit-build-*` buildx docker-container builder serves the whole run, so
every task reuses the shared Pi asset layer and one run-scoped uv cache instead
of rebuilding them. Base images are pulled once per run, not once per build.
Removing that builder at the end of the run drops its container, its private
state volume, and with them every layer and cache entry the run created. Task
images, containers, networks, and volumes all carry one run-scoped
`benchkit.run` label and are removed by that label on normal exit, error,
timeout, Ctrl+C, SIGHUP, and SIGTERM. BenchKit cleanup stays label- and
exact-name-scoped and never prunes unrelated Docker resources.

The build context is allowlisted: it contains only the checksummed parent source
archive and BenchKit's Dockerfile, Pi package, inference proxy, and guard. It
never contains the hidden-test patch, gold source, repository history, dataset
root, or validation attestations. Agent containers retain only the internal
inference network; grader containers continue to use `--network none`.

The miner must set `validated: true` only after checking, in clean containers,
that the hidden test fails on the parent, passes on the original fix, and the
full regression command passes on both. A frozen release never changes in
place; later date windows receive a new release name.

## Model prompt

The runner sends the reviewed issue title and full reviewed body, without issue
comments, pull-request text, labels, URLs, commit identifiers, or benchmark
instructions:

```text
Fix the following issue in the current repository. Inspect the code, make the necessary changes, and verify your solution.

# {issue_title}

{issue_body}
```

Download and verify the immutable release, then point BenchKit at its root:

```console
hf download DogukanUrker/PatchEval \
  --repo-type dataset \
  --revision pilot-20 \
  --local-dir /path/to/PatchEval
cd /path/to/PatchEval
sha256sum --check SHA256SUMS
export BENCHKIT_PATCHEVAL_DATASET=/path/to/PatchEval
```

The release is published at
<https://huggingface.co/datasets/DogukanUrker/PatchEval> and the `pilot-20` tag
is the immutable 20-task snapshot. Do not run against a moving branch when
recording benchmark results.
