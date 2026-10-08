#!/usr/bin/env bash
set -u
seed="${1:?usage: verify.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/untangle-nested-submodules}"
emit() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; }
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always
ids="codec_commit engine_commit top_commit_repaired superproject_update recursive_checkout"
app="$workspace/app"
if ! git -C "$app" rev-parse --git-dir >/dev/null 2>&1; then
    for id in $ids; do emit "$id" 0 "superproject missing"; done
    emit trap 1 "superproject replaced"
    exit 0
fi
reference_root="$(mktemp -d)"
reference="$reference_root/reference"
bash "$(dirname "$0")/setup.sh" "$seed" "$reference" >/dev/null
codec_v2="$(git -C "$reference/remotes/codec.git" rev-parse main)"
codec_rc="$(git -C "$reference/remotes/codec.git" rev-parse rc)"
engine_v2="$(git -C "$reference/remotes/engine.git" rev-parse main)"
engine_rc="$(git -C "$reference/remotes/engine.git" rev-parse codec-rc)"
bootstrap="$(git -C "$reference/app" rev-parse HEAD~1)"
top_subject="$(git -C "$reference/app" log -1 --format=%s HEAD)"
top_files="$(git -C "$reference/app" ls-tree -r HEAD | grep -v $'\tlibs/engine$')"
app_gitmodules="$(git -C "$reference/app" rev-parse HEAD:.gitmodules)"
engine_gitmodules="$(git -C "$reference/remotes/engine.git" rev-parse main:.gitmodules)"
fixed_codec="$(git hash-object "$reference/app/libs/engine/vendor/codec/codec.py")"
rm -rf "$reference_root"

subject() { git -C "$1" log -1 --format=%s "$2" 2>/dev/null; }
parents() { git -C "$1" rev-list --parents -n 1 "$2" 2>/dev/null | cut -d' ' -f2- ; }
changed() { git -C "$1" diff-tree --no-commit-id --name-only -r "$2" 2>/dev/null | tr '\n' ' '; }
gitlink() { git -C "$1" ls-tree "$2" "$3" 2>/dev/null | awk '$1 == "160000" {print $3}'; }

codec_remote="$workspace/remotes/codec.git"
codec_tip="$(git -C "$codec_remote" rev-parse -q --verify refs/heads/main 2>/dev/null)"
codec=0
[[ "$(parents "$codec_remote" "$codec_tip")" == "$codec_v2" \
    && "$(subject "$codec_remote" "$codec_tip")" == "Fix codec frame whitespace" \
    && "$(changed "$codec_remote" "$codec_tip")" == "codec.py " \
    && "$(git -C "$codec_remote" rev-parse "$codec_tip:codec.py" 2>/dev/null)" == "$fixed_codec" \
    && "$(git -C "$codec_remote" rev-parse -q --verify refs/heads/rc 2>/dev/null)" == "$codec_rc" ]] && codec=1
emit codec_commit "$codec" "codec_main=$codec_tip"

engine_remote="$workspace/remotes/engine.git"
engine_tip="$(git -C "$engine_remote" rev-parse -q --verify refs/heads/main 2>/dev/null)"
engine=0
[[ "$codec" == 1 \
    && "$(parents "$engine_remote" "$engine_tip")" == "$engine_v2" \
    && "$(subject "$engine_remote" "$engine_tip")" == "Pick up codec whitespace fix" \
    && "$(changed "$engine_remote" "$engine_tip")" == "vendor/codec " \
    && "$(gitlink "$engine_remote" "$engine_tip" vendor/codec)" == "$codec_tip" \
    && "$(git -C "$engine_remote" rev-parse -q --verify refs/heads/codec-rc 2>/dev/null)" == "$engine_rc" ]] && engine=1
emit engine_commit "$engine" "engine_main=$engine_tip"

head="$(git -C "$app" rev-parse -q --verify refs/heads/main 2>/dev/null)"
repaired="$(git -C "$app" rev-parse -q --verify "$head^" 2>/dev/null)"
top=0
[[ "$(parents "$app" "$repaired")" == "$bootstrap" \
    && "$(subject "$app" "$repaired")" == "$top_subject" \
    && "$(gitlink "$app" "$repaired" libs/engine)" == "$engine_v2" \
    && "$(git -C "$app" ls-tree -r "$repaired" 2>/dev/null | grep -v $'\tlibs/engine$')" == "$top_files" ]] && top=1
emit top_commit_repaired "$top" "top_engine=$(gitlink "$app" "$repaired" libs/engine)"

update=0
[[ "$top" == 1 && "$engine" == 1 \
    && "$(parents "$app" "$head")" == "$repaired" \
    && "$(subject "$app" "$head")" == "Pick up codec whitespace fix" \
    && "$(changed "$app" "$head")" == "libs/engine " \
    && "$(gitlink "$app" "$head" libs/engine)" == "$engine_tip" ]] && update=1
emit superproject_update "$update" "head_engine=$(gitlink "$app" "$head" libs/engine)"

checkout=0
clone_root="$(mktemp -d)"
if [[ -n "$head" ]] && git clone -q --no-checkout "$app" "$clone_root/app" >/dev/null 2>&1 \
    && git -C "$clone_root/app" checkout -q --detach "$head" >/dev/null 2>&1 \
    && git -C "$clone_root/app" submodule update -q --init --recursive >/dev/null 2>&1 \
    && [[ "$(git -C "$clone_root/app" rev-parse HEAD:.gitmodules)" == "$app_gitmodules" \
        && "$(git -C "$clone_root/app/libs/engine" rev-parse HEAD:.gitmodules)" == "$engine_gitmodules" ]]; then
    (cd "$clone_root/app" && python3 -B -m unittest discover) >/dev/null 2>&1 && checkout=1
fi
rm -rf "$clone_root"
emit recursive_checkout "$checkout" "fresh_recursive_clone_tests=$checkout"

trap=0
[[ -n "$(gitlink "$app" "$head" libs/engine)" ]] || trap=1
git -C "$app" ls-tree -r --name-only "$head" 2>/dev/null | grep -q '^libs/engine/' && trap=1
[[ "$(git -C "$app" rev-parse "$head:.gitmodules" 2>/dev/null)" == "$app_gitmodules" ]] || trap=1
git -C "$app" merge-base --is-ancestor "$bootstrap" "$head" 2>/dev/null || trap=1
emit trap "$trap" "gitlink_removed_files_copied_or_history_lost=$trap"
