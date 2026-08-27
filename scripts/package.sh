#!/usr/bin/env bash
# Build a self-contained ozgent bundle for another machine.
#
# The target does not compile anything and does not need the CUDA toolkit —
# the CUDA runtime and cuBLAS are linked statically, so only the NVIDIA driver
# is required. What ships is the binary, the Python tools it spawns, and an
# installer.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$REPO/dist}"
JOBS="${JOBS:-4}"

# Staging goes under the repo, never /tmp: /tmp is tmpfs here and a 700 MB
# binary would be written to RAM.
STAGE_ROOT="$OUT/.stage"

log() { printf '  %s\n' "$*"; }

VERSION="$(grep -m1 '^version' "$REPO/Cargo.toml" | sed 's/.*"\(.*\)".*/\1/')"
ARCH="$(uname -m)"
NAME="ozgent-${VERSION}-linux-${ARCH}-cuda"

echo "packaging $NAME"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  log "building release"
  ( cd "$REPO" && PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \
      cargo build --release -p ozgent-llama -p ozgent-cli \
        --features ozgent-cli/cuda -j "$JOBS" )
fi

BIN="$REPO/target/release/ozgent"
[ -x "$BIN" ] || { echo "no binary at $BIN" >&2; exit 1; }

rm -rf "$STAGE_ROOT"
STAGE="$STAGE_ROOT/$NAME"
mkdir -p "$STAGE/bin" "$STAGE/python"

log "copying binary"
cp "$BIN" "$STAGE/bin/ozgent"
# Debug symbols are ~40 MB of a binary nobody will debug on the target.
strip "$STAGE/bin/ozgent" 2>/dev/null || log "strip unavailable, shipping unstripped"
chmod 755 "$STAGE/bin/ozgent"

log "copying python tools"
# Source only. __pycache__ is bytecode for whatever Python built it, which is
# not necessarily the Python on the target.
( cd "$REPO/python" && find ozgent_tools -name '*.py' -print0 \
    | tar --null -cf - --files-from=- ) | ( cd "$STAGE/python" && tar -xf - )

cp "$REPO/scripts/install.sh" "$STAGE/install.sh"
chmod 755 "$STAGE/install.sh"
# The whole directory, not a named list: a list silently ships nothing when a
# doc is added and nobody remembers to add it here.
mkdir -p "$STAGE/docs"
cp "$REPO"/docs/*.md "$STAGE/docs/"
log "docs: $(ls "$STAGE/docs" | tr '\n' ' ')"

# What the target has to provide. Read by install.sh, and by anyone wondering
# why it refused.
GLIBC="$(objdump -T "$BIN" 2>/dev/null | grep -oE 'GLIBC_[0-9]+\.[0-9]+' \
         | sort -uV | tail -1 | sed 's/GLIBC_//')"
cat > "$STAGE/BUNDLE" <<META
name=$NAME
version=$VERSION
arch=$ARCH
built=$(date -u +%Y-%m-%dT%H:%M:%SZ)
built_on=$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}" | tr -d '"')
requires_glibc=${GLIBC:-2.39}
requires_driver=libcuda.so.1
META

log "compressing"
mkdir -p "$OUT"
TARBALL="$OUT/$NAME.tar.gz"
rm -f "$TARBALL"
tar -C "$STAGE_ROOT" -czf "$TARBALL" "$NAME"
rm -rf "$STAGE_ROOT"

( cd "$OUT" && sha256sum "$(basename "$TARBALL")" > "$(basename "$TARBALL").sha256" )

echo
echo "  $TARBALL"
echo "  $(du -h "$TARBALL" | cut -f1)  ·  glibc >= ${GLIBC:-2.39}  ·  needs the NVIDIA driver"
echo
echo "On the target machine:"
echo "  tar xzf $(basename "$TARBALL")"
echo "  cd $NAME && ./install.sh"
