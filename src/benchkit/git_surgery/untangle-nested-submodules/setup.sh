#!/usr/bin/env bash
set -euo pipefail
seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/untangle-nested-submodules}"
slug="$(printf '%s' "$seed:submodules" | sha256sum | cut -c1-12)"
mkdir -p "$workspace"
workspace="$(cd "$workspace" && pwd)"
cd "$workspace"
export GIT_AUTHOR_NAME="BenchKit Generator" GIT_AUTHOR_EMAIL="generator@benchkit.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
# Submodules are cloned from local paths; Git >= 2.38.1 refuses that unless asked.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always
clock=1700900000
tick() {
    clock=$((clock + 60))
    export GIT_AUTHOR_DATE="@$clock +0000" GIT_COMMITTER_DATE="@$clock +0000"
}
configure() {
    git -C "$1" config user.name "BenchKit Generator"
    git -C "$1" config user.email "generator@benchkit.invalid"
    git -C "$1" config commit.gpgSign false
    git -C "$1" config core.autocrlf false
}
commit() {
    local repo="$1"
    shift
    tick
    git -C "$repo" add -A
    git -C "$repo" commit -q "$@"
}
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

mkdir -p remotes
git init -q --bare -b main remotes/codec.git
git init -q --bare -b main remotes/engine.git

# codec: the innermost library.
git init -q -b main "$scratch/codec"
configure "$scratch/codec"
git -C "$scratch/codec" remote add origin "$workspace/remotes/codec.git"
cat > "$scratch/codec/codec.py" <<'PY'
"""Length-prefixed frame codec."""


def encode(payload: str) -> str:
    return f"{len(payload)}:{payload}"


def decode(frame: str) -> str:
    size, _, payload = frame.partition(":")
    return payload[: int(size)]
PY
commit "$scratch/codec" -m "codec: length-prefixed frames"
cat >> "$scratch/codec/codec.py" <<'PY'


def decode_all(stream: str) -> list[str]:
    frames = []
    while stream:
        size, _, rest = stream.partition(":")
        frames.append(rest[: int(size)].strip())
        stream = rest[int(size) :]
    return frames
PY
commit "$scratch/codec" -m "codec: decode frame streams"
tick
git -C "$scratch/codec" push -q origin main
git -C "$scratch/codec" switch -q -c rc
printf '\nVERSION = "2.0-rc"\n' >> "$scratch/codec/codec.py"
commit "$scratch/codec" -m "codec: 2.0 release candidate"
tick
git -C "$scratch/codec" push -q origin rc
codec_v1="$(git -C "$scratch/codec" rev-parse main~1)"
codec_v2="$(git -C "$scratch/codec" rev-parse main)"
codec_rc="$(git -C "$scratch/codec" rev-parse rc)"

# engine: vendors codec as a submodule.
git init -q -b main "$scratch/engine"
configure "$scratch/engine"
git -C "$scratch/engine" remote add origin "$workspace/remotes/engine.git"
git -C "$scratch/engine" submodule add -q ../codec.git vendor/codec
git -C "$scratch/engine/vendor/codec" checkout -q "$codec_v1"
cat > "$scratch/engine/engine.py" <<'PY'
"""Message engine built on the vendored codec."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "codec"))

import codec  # noqa: E402


def roundtrip(message: str) -> str:
    return codec.decode(codec.encode(message))
PY
commit "$scratch/engine" -m "engine: roundtrip messages through codec"
git -C "$scratch/engine/vendor/codec" checkout -q "$codec_v2"
cat >> "$scratch/engine/engine.py" <<'PY'


def unpack(stream: str) -> list[str]:
    return codec.decode_all(stream)
PY
commit "$scratch/engine" -m "engine: use codec stream decoding"
tick
git -C "$scratch/engine" push -q origin main
engine_v1="$(git -C "$scratch/engine" rev-parse main~1)"
engine_v2="$(git -C "$scratch/engine" rev-parse main)"
git -C "$scratch/engine" switch -q -c codec-rc
git -C "$scratch/engine/vendor/codec" checkout -q "$codec_rc"
commit "$scratch/engine" -m "engine: try codec 2.0 release candidate"
tick
git -C "$scratch/engine" push -q origin codec-rc

# app: the superproject.
git init -q -b main app
configure app
printf '__pycache__/\n' > app/.gitignore
git -C app submodule add -q ../remotes/engine.git libs/engine
git -C app/libs/engine checkout -q "$engine_v1"
git -C app add libs/engine
git -C app submodule update -q --init --recursive
cat > app/main.py <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "libs" / "engine"))

import engine  # noqa: E402


def handle(message: str) -> str:
    return engine.roundtrip(message).upper()
PY
cat > app/test_app.py <<'PY'
import unittest

from main import handle


class AppTests(unittest.TestCase):
    def test_handle(self):
        self.assertEqual(handle("ping"), "PING")
PY
printf '# Messaging app %s\n\nClone with `--recurse-submodules`; run `python3 -m unittest discover -v`.\n' "$slug" > app/README.md
commit app -m "bootstrap messaging app $slug"

# The top commit points libs/engine at a commit that only exists in this
# checkout: the same change as engine's published main, committed locally.
engine_dir=app/libs/engine
configure "$engine_dir"
configure "$engine_dir/vendor/codec"
git -C "$engine_dir/vendor/codec" checkout -q "$codec_v2"
git -C "$engine_dir" checkout -q "$engine_v2" -- engine.py
commit "$engine_dir" -m "engine: use codec stream decoding"
cat >> app/main.py <<'PY'


def handle_stream(stream: str) -> list[str]:
    return [frame.upper() for frame in engine.unpack(stream)]
PY
cat >> app/test_app.py <<'PY'


class StreamTests(unittest.TestCase):
    def test_stream(self):
        from main import handle_stream

        self.assertEqual(handle_stream("2:hi3:you"), ["HI", "YOU"])

    def test_frames_keep_whitespace(self):
        from main import handle_stream

        self.assertEqual(handle_stream("3: hi2:yo"), [" HI", "YO"])
PY
commit app -m "Handle message streams with engine and codec 2"

# Uncommitted work in the nested codec checkout stops stripping frame payloads.
python3 - "$engine_dir/vendor/codec/codec.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
old = "        frames.append(rest[: int(size)].strip())\n"
new = "        frames.append(rest[: int(size)])\n"
if old not in text:
    raise SystemExit("setup: codec stream loop not found")
path.write_text(text.replace(old, new))
PY
unset GIT_AUTHOR_DATE GIT_COMMITTER_DATE
