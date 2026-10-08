#!/usr/bin/env bash
set -u
seed="${1:?usage: verify.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/backport-release-stack}"
emit() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; }
ids="selected_fixes skipped_equivalent intermediate_tests final_behavior no_release_leak"
if ! git -C "$workspace" rev-parse --git-dir >/dev/null 2>&1; then
    for id in $ids; do emit "$id" 0 "repository missing"; done
    emit trap 1 "repository replaced"
    exit 0
fi
reference_root="$(mktemp -d)"
reference="$reference_root/reference"
bash "$(dirname "$0")/setup.sh" "$seed" "$reference" >/dev/null
fixed() { git -C "$reference" log release-2.x --format=%H --grep="^Fixes: $1\$" -1; }
release_tip="$(git -C "$reference" rev-parse release-2.x)"
maint_tip="$(git -C "$reference" rev-parse maint-1.x)"
expected_sources="$(fixed BK-101) $(fixed BK-104) $(fixed BK-107) $(fixed BK-115)"
equivalent="$(fixed BK-112)"
release_only="$(git -C "$reference" log release-2.x --format=%H --grep='^refactor: rename parse_amount locals$' -1)"
rm -rf "$reference_root"

git_w() { git -C "$workspace" "$@" 2>/dev/null; }
release_now="$(git_w rev-parse -q --verify refs/heads/release-2.x)"
maint_now="$(git_w rev-parse -q --verify refs/heads/maint-1.x)"
git_w merge-base --is-ancestor "$maint_tip" "$maint_now" && kept=1 || kept=0
[[ "$release_now" == "$release_tip" && "$kept" == 1 ]] && untouched=1 || untouched=0

new_commits=()
if [[ "$kept" == 1 ]]; then
    mapfile -t new_commits < <(git_w rev-list --reverse --first-parent "$maint_tip..$maint_now")
fi
merges="$(git_w rev-list --merges --count "$maint_tip..$maint_now" || printf 0)"
sources=()
for commit in "${new_commits[@]}"; do
    source="$(git_w log -1 --format=%B "$commit" | sed -n 's/^(cherry picked from commit \([0-9a-f]\{40\}\))$/\1/p' | tail -1)"
    sources+=("${source:-none}")
done
actual_sources="${sources[*]:-}"
[[ "$untouched" == 1 && "$actual_sources" == "$expected_sources" && "$merges" == 0 ]] && selected=1 || selected=0
emit selected_fixes "$selected" "count=${#new_commits[@]} merges=$merges release_and_maint_base_kept=$untouched sources=$actual_sources"

empty=0
duplicate=0
for index in "${!new_commits[@]}"; do
    commit="${new_commits[$index]}"
    [[ -z "$(git_w diff-tree --no-commit-id --name-only -r "$commit")" ]] && empty=1
    [[ "${sources[$index]}" == "$equivalent" ]] && duplicate=1
    git_w log -1 --format=%B "$commit" | grep -q 'BK-112' && duplicate=1
done
[[ "${#new_commits[@]}" -gt 0 && "$empty" == 0 && "$duplicate" == 0 ]] && skipped=1 || skipped=0
emit skipped_equivalent "$skipped" "empty_commits=$empty duplicated_bk112=$duplicate"

# Hidden behavior per backport, in the order they must land.
checks=(
    'assert parse_amount("  42\n") == 42'
    'assert parse_amount(" -75 ") == -75'
    'assert format_amount(-5) == "-0.05"; assert format_amount(123456789012345678) == "1234567890123456.78"'
    'assert DEFAULTS["timeout"] == 10'
)
preamble='from ledger.config import DEFAULTS
from ledger.discount import apply_discount
from ledger.money import format_amount, parse_amount
assert parse_amount("1250") == 1250
assert format_amount(1205) == "12.05"
assert apply_discount(100, 250) == 0
assert DEFAULTS["retries"] == 3'
run_archive() {
    local commit="$1" script="$2" tmp result=1
    tmp="$(mktemp -d)"
    if git_w archive "$commit" | tar -x -C "$tmp" 2>/dev/null; then
        (cd "$tmp" && python3 -B -c "$script" && python3 -B -m unittest discover) >/dev/null 2>&1 && result=0
    fi
    rm -rf "$tmp"
    return "$result"
}
intermediate=0
passing=0
if [[ "$selected" == 1 ]]; then
    intermediate=1
    for index in "${!new_commits[@]}"; do
        script="$preamble"
        for prior in $(seq 0 "$index"); do script+=$'\n'"${checks[$prior]}"; done
        if run_archive "${new_commits[$index]}" "$script"; then
            passing=$((passing + 1))
        else
            intermediate=0
        fi
    done
fi
emit intermediate_tests "$intermediate" "passing_backports=$passing/${#new_commits[@]}"

final_script="$preamble"
for check in "${checks[@]}"; do final_script+=$'\n'"$check"; done
final_script+=$'\nfrom ledger import config\nassert config.VERSION == "1.0.3"'
final=0
[[ "${#new_commits[@]}" -gt 0 ]] && run_archive "$maint_now" "$final_script" && final=1
emit final_behavior "$final" "all_fixes_and_maint_version=$final"

leak=1
if [[ "${#new_commits[@]}" -gt 0 ]]; then
    leak_script='import inspect
from ledger import config, money
assert not hasattr(config, "RELEASE_CHANNEL")
assert "allow_commas" not in inspect.signature(money.parse_amount).parameters
assert money.parse_amount("1" * 30) == int("1" * 30)'
    git_w cat-file -e "$maint_now:ledger/export.py" || {
        tmp="$(mktemp -d)"
        git_w archive "$maint_now" | tar -x -C "$tmp" 2>/dev/null
        (cd "$tmp" && python3 -B -c "$leak_script") >/dev/null 2>&1 && leak=0
        rm -rf "$tmp"
    }
fi
emit no_release_leak "$((1 - leak))" "release_only_changes=$leak"

trap=0
[[ -n "$maint_now" ]] && git_w merge-base --is-ancestor "$release_only" "$maint_now" && trap=1
[[ -z "$maint_now" ]] && trap=1
emit trap "$trap" "release_history_merged_into_maint=$trap"
