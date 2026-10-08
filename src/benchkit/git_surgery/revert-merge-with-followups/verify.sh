#!/usr/bin/env bash
set -u
seed="${1:?usage: verify.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/revert-merge-with-followups}"
emit() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; }
ids="revert_commit topology_preserved regression_removed both_parents_kept tests_pass"
if ! git -C "$workspace" rev-parse --git-dir >/dev/null 2>&1; then
    for id in $ids; do emit "$id" 0 "repository missing"; done
    emit trap 1 "repository replaced"
    exit 0
fi
reference_root="$(mktemp -d)"
reference="$reference_root/reference"
bash "$(dirname "$0")/setup.sh" "$seed" "$reference" >/dev/null
followups_tip="$(git -C "$reference" rev-parse main)"
faulty="$(git -C "$reference" log main --merges --format=%H --grep="^Merge branch 'pricing-engine'$" -1)"
mainline="$(git -C "$reference" rev-parse "$faulty^1")"
healthy="$(git -C "$reference" log main --merges --format=%H --grep="^Merge branch 'catalog-search'$" -1)"
declare -A test_blobs=()
for path in test_pricing.py test_bulk.py test_search.py; do
    test_blobs[$path]="$(git -C "$reference" rev-parse "main:$path")"
done
rm -rf "$reference_root"

git_w() { git -C "$workspace" "$@" 2>/dev/null; }
head="$(git_w rev-parse -q --verify HEAD)"
branch="$(git_w symbolic-ref -q --short HEAD)"
main="$(git_w rev-parse -q --verify refs/heads/main)"
on_top=0
[[ -n "$main" && "$(git_w rev-parse -q --verify "$main^")" == "$followups_tip" ]] && on_top=1
parents="$(git_w rev-list --parents -n 1 "$main" | wc -w)"
message="$(git_w log -1 --format=%B "$main" | tr '\n' ' ' | tr -s ' ')"
references=0
[[ "$message" == *"This reverts commit $faulty, reversing changes made to $mainline."* ]] && references=1
revert=0
[[ "$on_top" == 1 && "$parents" == 2 && "$references" == 1 ]] && revert=1
emit revert_commit "$revert" "single_commit_on_followups=$on_top references_merge_and_mainline=$references"

topology=0
if [[ "$on_top" == 1 ]]; then
    first_parent="$(git_w rev-list --first-parent "$main" | tr '\n' ' ')"
    merge_parents="$(git_w rev-list --parents -n 1 "$faulty" | wc -w)"
    [[ "$first_parent" == *"$faulty"* && "$first_parent" == *"$healthy"* && "$merge_parents" == 3 && "$branch" == main && "$head" == "$main" ]] && topology=1
fi
emit topology_preserved "$topology" "merges_on_first_parent=$topology branch=$branch"

tree="$(mktemp -d)"
[[ -n "$main" ]] && git_w archive "$main" | tar -x -C "$tree" 2>/dev/null
behavior() { (cd "$tree" && python3 -B -c "$1") >/dev/null 2>&1; }
regression=0
if [[ "$on_top" == 1 ]] && behavior '
from pricing import price
assert price("plum", 150) == 8100
assert price("pear", 100) == 8500
assert price("apple", 1000) == 108000'; then
    regression=1
fi
emit regression_removed "$regression" "large_orders_charged=$regression"

parents_kept=0
bulk=0
tax=0
behavior '
from pricing import BULK_QTY, BULK_RATE, price
assert (BULK_QTY, BULK_RATE) == (8, 90)
assert price("apple", 8) == 864 and price("apple", 7) == 840' && bulk=1
behavior '
from pricing import receipt, total_with_tax
from search import search
assert total_with_tax([("pear", 1)]) == 103
assert receipt([("apple", 8)]) == "net=864 gross=933"
assert search("p") == ["pear", "plum"]
try:
    from pricing import price
    price("pear", 0)
except ValueError:
    pass
else:
    raise AssertionError("quantity validation lost")' && tax=1
[[ "$on_top" == 1 && "$bulk" == 1 && "$tax" == 1 ]] && parents_kept=1
emit both_parents_kept "$parents_kept" "pricing_engine_side=$bulk main_side_and_followups=$tax"

tests=0
blobs=1
for path in "${!test_blobs[@]}"; do
    [[ "$(git_w rev-parse "$main:$path")" == "${test_blobs[$path]}" ]] || blobs=0
done
if [[ "$on_top" == 1 && "$blobs" == 1 ]]; then
    (cd "$tree" && python3 -B -m unittest discover) >/dev/null 2>&1 && tests=1
fi
rm -rf "$tree"
emit tests_pass "$tests" "tests_unchanged=$blobs"

trap=0
git_w merge-base --is-ancestor "$followups_tip" "$main" || trap=1
git_w merge-base --is-ancestor "$faulty" "$main" || trap=1
[[ "$bulk" == 1 && "$tax" == 1 ]] || trap=1
emit trap "$trap" "history_reset_or_parent_behavior_lost=$trap"
