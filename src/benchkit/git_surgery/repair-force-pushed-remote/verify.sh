#!/usr/bin/env bash
set -u
seed="${1:?usage: verify.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/repair-force-pushed-remote}"
emit() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; }
ids="remote_intact lineage_restored followup_retained healthy_refs_untouched tests_pass"
origin="$workspace/origin.git"
if [[ "$(git -C "$origin" rev-parse --is-bare-repository 2>/dev/null)" != true ]]; then
    for id in $ids; do emit "$id" 0 "bare origin missing"; done
    emit trap 1 "origin replaced or removed"
    exit 0
fi
reference_root="$(mktemp -d)"
reference="$reference_root/reference"
bash "$(dirname "$0")/setup.sh" "$seed" "$reference" >/dev/null
git_r() { git -C "$reference/origin.git" "$@"; }
mapfile -t history < <(git_r reflog show --format=%H refs/heads/release)
# Newest first: Alice's fix, Bob's force-push, Alice's 1.2 push, first publish.
published="${history[2]}"
rewritten="${history[1]}"
followup="${history[0]}"
first_publish="${history[3]}"
decoys=""
for commit in $(git_r fsck --unreachable --no-reflogs 2>/dev/null | awk '/unreachable commit/{print $3}'); do
    git_r merge-base --is-ancestor "$commit" "$published" || decoys+=" $commit"
done
decoys+=" $(git -C "$reference/alice" reflog --format='%H %gs' | awk '/^[0-9a-f]+ commit: release: prepare 1.2$/{print $1}' | tr '\n' ' ')"
healthy_refs="$(git_r for-each-ref --format='%(objectname) %(refname)' | grep -v ' refs/heads/release$')"
export GIT_INDEX_FILE="$reference_root/index"
git_r read-tree "$published"
git_r diff --binary "$followup^" "$followup" | git_r apply --cached
expected_tree="$(git_r write-tree)"
unset GIT_INDEX_FILE
followup_patch="$(git_r diff "$followup^" "$followup" | git_r patch-id --stable | cut -d' ' -f1)"
rm -rf "$reference_root"

git_o() { git -C "$origin" "$@" 2>/dev/null; }
release="$(git_o rev-parse -q --verify refs/heads/release)"
changed=0
[[ -n "$release" && "$release" != "$followup" ]] && changed=1
reflog="$(git_o reflog show --format=%H refs/heads/release | tr '\n' ' ')"
intact=0
[[ "$changed" == 1 && "$reflog" == *"$first_publish"* && "$reflog" == *"$published"* ]] && git_o cat-file -e "$published" && intact=1
emit remote_intact "$intact" "bare=1 release_reflog_kept=$intact"

parent="$(git_o rev-parse -q --verify "$release^")"
lineage=0
[[ "$parent" == "$published" ]] && lineage=1
git_o merge-base --is-ancestor "$rewritten" "$release" && lineage=0
emit lineage_restored "$lineage" "release_parent=$parent expected=$published"

retained=0
if [[ "$lineage" == 1 ]]; then
    parents="$(git_o rev-list --parents -n 1 "$release" | wc -w)"
    patch="$(git_o diff "$release^" "$release" | git patch-id --stable | cut -d' ' -f1)"
    tree="$(git_o rev-parse "$release^{tree}")"
    [[ "$parents" == 2 && "$patch" == "$followup_patch" && "$tree" == "$expected_tree" ]] && retained=1
fi
emit followup_retained "$retained" "followup_reapplied_on_published_tip=$retained"

current_refs="$(git_o for-each-ref --format='%(objectname) %(refname)' | grep -v ' refs/heads/release$')"
healthy=0
[[ "$changed" == 1 && "$current_refs" == "$healthy_refs" ]] && healthy=1
emit healthy_refs_untouched "$healthy" "other_refs_match=$([[ "$current_refs" == "$healthy_refs" ]] && echo 1 || echo 0)"

tests=0
if [[ "$changed" == 1 ]]; then
    tmp="$(mktemp -d)"
    git_o archive "$release" | tar -x -C "$tmp" 2>/dev/null
    (cd "$tmp" && python3 -B -m unittest discover) >/dev/null 2>&1 \
        && [[ -f "$tmp/test_grand_total.py" && -f "$tmp/CHANGES.md" ]] && tests=1
    rm -rf "$tmp"
fi
emit tests_pass "$tests" "release_tests=$tests"

trap=0
[[ "$current_refs" == "$healthy_refs" ]] || trap=1
[[ -n "$release" ]] || trap=1
[[ "$reflog" == *"$first_publish"* ]] || trap=1
for decoy in $decoys; do
    [[ -n "$release" ]] && git_o merge-base --is-ancestor "$decoy" "$release" && trap=1
done
emit trap "$trap" "healthy_refs_changed_or_wrong_lineage=$trap"
