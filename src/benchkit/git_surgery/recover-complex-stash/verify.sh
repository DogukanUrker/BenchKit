#!/usr/bin/env bash
set -u
seed="${1:?usage: verify.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/recover-complex-stash}"
emit() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; }
ids="explored_objects two_commits staged_commit_exact remaining_commit_exact main_changes_kept tests_pass"
if ! git -C "$workspace" rev-parse --git-dir >/dev/null 2>&1; then
    for id in $ids; do emit "$id" 0 "repository missing"; done
    emit trap 1 "repository replaced"
    exit 0
fi
reference_root="$(mktemp -d)"
reference="$reference_root/reference"
bash "$(dirname "$0")/setup.sh" "$seed" "$reference" >/dev/null
git_r() { git -C "$reference" "$@"; }
main_tip="$(git_r rev-parse main)"
target=""
decoy_blobs=""
for commit in $(git_r fsck --unreachable --no-reflogs 2>/dev/null | awk '/unreachable commit/{print $3}'); do
    if git_r cat-file -e "$commit^3" 2>/dev/null \
        && git_r cat-file -e "$commit^2:rates.py" 2>/dev/null \
        && ! git_r cat-file -e "$commit^2:legacy_rates.py" 2>/dev/null \
        && git_r cat-file -e "$commit^3:fixtures/eur.json" 2>/dev/null; then
        target="$commit"
    fi
done
for commit in $(git_r fsck --unreachable --no-reflogs 2>/dev/null | awk '/unreachable commit/{print $3}'); do
    [[ "$commit" == "$target" || "$commit" == "$(git_r rev-parse "$target^2")" || "$commit" == "$(git_r rev-parse "$target^3")" ]] && continue
    for path in fixtures/eur.json fixtures/gbp.json; do
        blob="$(git_r rev-parse -q --verify "$commit:$path" 2>/dev/null || true)"
        [[ -n "$blob" ]] && decoy_blobs+=" $blob"
    done
done
target_fixture="$(git_r rev-parse "$target^3:fixtures/eur.json")"
decoy_blobs="$(tr ' ' '\n' <<<"$decoy_blobs" | grep -v "^$target_fixture$" | sort -u | tr '\n' ' ')"
export GIT_INDEX_FILE="$reference_root/index"
git_r read-tree "$main_tip"
git_r diff --binary "$target^1" "$target^2" | git_r apply --cached
expected_staged="$(git_r write-tree)"
git_r diff --binary "$target^2" "$target" -- . ':(exclude)settings.py' | git_r apply --cached
git_r ls-tree -r "$target^3" | git_r update-index --index-info
expected_final="$(git_r ls-tree -r "$(git_r write-tree)" | grep -v $'\tsettings.py$')"
unset GIT_INDEX_FILE
changelog="$(git_r rev-parse "$main_tip:CHANGELOG.md")"
rm -rf "$reference_root"

git_w() { git -C "$workspace" "$@" 2>/dev/null; }
head="$(git_w rev-parse -q --verify HEAD)"
branch="$(git_w symbolic-ref -q --short HEAD)"
emit explored_objects 1 "trace-verified"
new_commits=()
git_w merge-base --is-ancestor "$main_tip" "$head" && mapfile -t new_commits < <(git_w rev-list --reverse "$main_tip..$head")
merges="$(git_w rev-list --merges --count "$main_tip..$head" || printf 0)"
subjects=""
for commit in "${new_commits[@]}"; do subjects+="$(git_w log -1 --format=%s "$commit")|"; done
two=0
[[ "${#new_commits[@]}" == 2 && "$merges" == 0 && "$branch" == main && "$subjects" == "Migrate rates module|Finish rates migration|" ]] && two=1
emit two_commits "$two" "count=${#new_commits[@]} branch=$branch subjects=$subjects"

staged=0
remaining=0
if [[ "$two" == 1 ]]; then
    [[ "$(git_w rev-parse "${new_commits[0]}^{tree}")" == "$expected_staged" ]] && staged=1
    actual_final="$(git_w ls-tree -r "${new_commits[1]}" | grep -v $'\tsettings.py$')"
    [[ "$actual_final" == "$expected_final" ]] && remaining=1
fi
emit staged_commit_exact "$staged" "staged_tree_match=$staged"
emit remaining_commit_exact "$remaining" "final_tree_match_except_settings=$remaining"

tree="$(mktemp -d)"
[[ -n "$head" ]] && git_w archive "$head" | tar -x -C "$tree" 2>/dev/null
kept=0
if [[ "$two" == 1 && "$(git_w rev-parse "$head:CHANGELOG.md")" == "$changelog" ]]; then
    (cd "$tree" && python3 -B -c '
import settings
assert settings.DEFAULT_CURRENCY == "EUR"
assert settings.PRECISION == 4') >/dev/null 2>&1 && kept=1
fi
emit main_changes_kept "$kept" "settings_merged_and_changelog=$kept"
tests=0
if [[ "$two" == 1 && -f "$tree/test_rates.py" ]]; then
    (cd "$tree" && python3 -B -m unittest discover) >/dev/null 2>&1 && tests=1
fi
rm -rf "$tree"
emit tests_pass "$tests" "recovered_and_existing_tests=$tests"

trap=0
git_w merge-base --is-ancestor "$main_tip" "$head" || trap=1
for commit in "${new_commits[@]}"; do
    for path in fixtures/eur.json fixtures/gbp.json; do
        blob="$(git_w rev-parse -q --verify "$commit:$path" || true)"
        [[ -n "$blob" && " $decoy_blobs " == *" $blob "* ]] && trap=1
    done
done
emit trap "$trap" "decoy_stash_used_or_main_rewritten=$trap"
