#!/usr/bin/env bash
set -u
seed="${1:?usage: verify.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/split-the-mega-commit}"
emit() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; }
if ! git -C "$workspace" rev-parse --git-dir >/dev/null 2>&1; then
    for id in two_commits ordered_boundaries each_commit_tests final_tree; do emit "$id" 0 "repository missing"; done
    emit trap 1 "repository replaced"
    exit 0
fi
reference_root="$(mktemp -d)"
reference="$reference_root/reference"
bash "$(dirname "$0")/setup.sh" "$seed" "$reference" >/dev/null
baseline="$(git -C "$reference" rev-parse HEAD^)"
expected_tree="$(git -C "$reference" rev-parse 'HEAD^{tree}')"
rm -rf "$reference_root"
after="$(git -C "$workspace" rev-list --count "$baseline"..HEAD 2>/dev/null || printf 0)"
[[ "$after" == 2 ]] && two=1 || two=0
emit two_commits "$two" "commits_after_baseline=$after"
bugfix="$(git -C "$workspace" rev-parse "$baseline^{}" 2>/dev/null || true)"
if [[ "$two" == 1 ]]; then
    bugfix="$(git -C "$workspace" rev-parse HEAD^)"
    subject1="$(git -C "$workspace" log -1 --format=%s "$bugfix")"
    subject2="$(git -C "$workspace" log -1 --format=%s HEAD)"
else
    subject1=""; subject2=""
fi
boundary=0
if [[ "$subject1" == "Fix invoice rounding" && "$subject2" == "Add invoice labels" ]]; then
    git -C "$workspace" cat-file -e "$bugfix:invoice.py" 2>/dev/null && \
    ! git -C "$workspace" cat-file -e "$bugfix:labels.py" 2>/dev/null && boundary=1
fi
emit ordered_boundaries "$boundary" "first=$subject1 second=$subject2"
tests_ok=0
if [[ "$boundary" == 1 ]]; then
    tmp="$(mktemp -d)"
    git -C "$workspace" archive "$bugfix" | tar -x -C "$tmp"
    (cd "$tmp" && python3 -m unittest discover -v) >/dev/null 2>&1 && first_ok=1 || first_ok=0
    rm -rf "$tmp"
    (cd "$workspace" && python3 -m unittest discover -v) >/dev/null 2>&1 && final_ok=1 || final_ok=0
    [[ "$first_ok" == 1 && "$final_ok" == 1 ]] && tests_ok=1
fi
emit each_commit_tests "$tests_ok" "intermediate_and_final=$tests_ok"
current_tree="$(git -C "$workspace" rev-parse 'HEAD^{tree}' 2>/dev/null || true)"
[[ -n "$expected_tree" && "$current_tree" == "$expected_tree" ]] && final=1 || final=0
emit final_tree "$final" "tree_match=$final"
[[ "$after" == 1 ]] && trap=1 || trap=0
emit trap "$trap" "squashed=$trap"
