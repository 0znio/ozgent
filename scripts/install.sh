#!/usr/bin/env bash
# Install ozgent from an unpacked bundle. Nothing is compiled here.
#
# Everything lands under one directory so that removing it removes ozgent, and
# so the binary can find its Python tools by looking beside itself. Your own
# data — models, config, logs — lives in ~/ozgent and is never touched.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX=""
FORCE=0

usage() {
  cat <<'USAGE'
Usage: ./install.sh [--prefix DIR] [--force]

  --prefix DIR   where to install (default: /usr/local, using sudo to copy)
  --force        install even if a requirement check fails
  --uninstall    remove a previous installation and exit

Installs to PREFIX/lib/ozgent and links PREFIX/bin/ozgent.
Your models and settings in ~/ozgent are left alone.
USAGE
}

UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# /usr/local, like other programs: on everyone's PATH with nothing to add to a
# dotfile. The copy uses sudo when the directory is root's.
PREFIX="${PREFIX:-/usr/local}"
LIBDIR="$PREFIX/lib/ozgent"
BINLINK="$PREFIX/bin/ozgent"
AS=""
if [ "$(id -u)" -ne 0 ] && ! { [ -w "$PREFIX" ] || { [ ! -e "$PREFIX" ] && mkdir -p "$PREFIX" 2>/dev/null; }; }; then
  if command -v sudo >/dev/null 2>&1; then AS=sudo
  elif command -v doas >/dev/null 2>&1; then AS=doas
  else
    echo "cannot write to $PREFIX and there is no sudo or doas. Run as root, or: --prefix ~/.local" >&2
    exit 1
  fi
fi

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
bad()  { printf '  \033[31mno\033[0m    %s\n' "$*"; }

if [ "$UNINSTALL" = "1" ]; then
  echo "removing ozgent from $PREFIX"
  $AS rm -rf "$LIBDIR"; $AS rm -f "$BINLINK"
  ok "removed. ~/ozgent (models, settings, logs) was left alone."
  exit 0
fi

# Read, never source: the file carries free text such as the build machine's
# name, and sourcing turns "Arch Linux" into a command.
meta() {
  [ -f "$HERE/BUNDLE" ] || return 0
  sed -n "s/^$1=//p" "$HERE/BUNDLE" | head -1
}
version="$(meta version)"
arch="$(meta arch)"
requires_glibc="$(meta requires_glibc)"

echo "ozgent ${version:-?} → $PREFIX"
echo

# ----------------------------------------------------------------- checks
fail=0

case "$(uname -s)" in
  Linux) ok "linux" ;;
  *) bad "this bundle is for Linux; found $(uname -s)"; fail=1 ;;
esac

if [ "$(uname -m)" = "${arch:-x86_64}" ]; then
  ok "$(uname -m)"
else
  bad "bundle is for ${arch:-x86_64}, this machine is $(uname -m)"; fail=1
fi

# glibc. The binary is dynamically linked against the system C library, so a
# target older than the build machine cannot load it at all — better to say so
# now than to hand over something that dies with a symbol error.
need="${requires_glibc:-2.39}"
# Captured before it is picked apart. Piping `ldd --version` into `head` makes
# ldd die of SIGPIPE, and under `pipefail` that fails the whole pipeline — so
# a fallback would run *after* the version had already been printed, and the
# two would concatenate into nonsense.
ldd_out="$(ldd --version 2>/dev/null || true)"
have="$(printf '%s\n' "$ldd_out" | sed -n '1s/.*[^0-9.]\([0-9][0-9]*\.[0-9][0-9]*\).*$/\1/p')"
[ -n "$have" ] || have=0
if [ "$(printf '%s\n%s\n' "$need" "$have" | sort -V | head -1)" = "$need" ]; then
  ok "glibc $have (needs $need)"
else
  bad "glibc $have is too old; this bundle needs $need or newer"
  echo "        Ubuntu 24.04, Debian 13, Fedora 40 and Arch are new enough."
  echo "        Ubuntu 22.04 and Debian 12 are not."
  fail=1
fi

# The NVIDIA driver. libcuda.so.1 is a hard dependency of this build: without
# it the binary will not start even to run on the CPU.
if ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; then
  ok "NVIDIA driver present"
elif [ -e /usr/lib/libcuda.so.1 ] || [ -e /usr/lib64/libcuda.so.1 ]; then
  ok "NVIDIA driver present"
else
  bad "libcuda.so.1 not found — this is the NVIDIA driver, not the CUDA toolkit"
  echo "        This CUDA build cannot start without it."
  fail=1
fi

# Python runs the tools. Missing it costs the tools, not ozgent.
if command -v python3 >/dev/null 2>&1; then
  ok "python3 $(python3 -V 2>&1 | cut -d' ' -f2) (for tools)"
else
  warn "python3 not found — ozgent will run, but its tools will not load"
fi

if [ "$fail" = "1" ] && [ "$FORCE" != "1" ]; then
  echo
  echo "Refusing to install. Re-run with --force to override." >&2
  exit 1
fi

# ----------------------------------------------------------------- install
echo
[ -n "$AS" ] && echo "$PREFIX belongs to root; $AS will ask for your password to copy the files"
$AS mkdir -p "$LIBDIR" "$PREFIX/bin"

# Replace wholesale rather than merge, so an old file from a previous version
# cannot survive into a new install.
$AS rm -rf "$LIBDIR/bin" "$LIBDIR/python" "$LIBDIR/docs"
$AS cp -r "$HERE/bin" "$LIBDIR/bin"
$AS cp -r "$HERE/python" "$LIBDIR/python"
[ -d "$HERE/docs" ] && $AS cp -r "$HERE/docs" "$LIBDIR/docs"
[ -f "$HERE/BUNDLE" ] && $AS cp "$HERE/BUNDLE" "$LIBDIR/BUNDLE"
# Kept alongside so uninstalling does not require finding the bundle again.
$AS cp "$HERE/install.sh" "$LIBDIR/install.sh" 2>/dev/null || true
$AS chmod 755 "$LIBDIR/install.sh" 2>/dev/null || true
$AS chmod 755 "$LIBDIR/bin/ozgent"

# The binary locates its tools by walking up from its own path, so the symlink
# must point into LIBDIR rather than the binary being copied to bin/.
$AS ln -sfn "$LIBDIR/bin/ozgent" "$BINLINK"

# An earlier version of this script defaulted to ~/.local when not root. Its
# link would sit before /usr/local/bin on PATH and keep running the old build.
old="$HOME/.local"
if [ "$PREFIX" != "$old" ] && [ -L "$old/bin/ozgent" ] &&
   [ "$(readlink "$old/bin/ozgent")" = "$old/lib/ozgent/bin/ozgent" ]; then
  rm -rf "$old/lib/ozgent" "$old/bin/ozgent"
  ok "removed the older install in $old"
fi
ok "installed to $LIBDIR"
ok "linked $BINLINK"

# ----------------------------------------------------------------- verify
echo
if ! "$BINLINK" --version >/dev/null 2>&1; then
  echo "installed, but the binary would not run. Try: $BINLINK --version" >&2
  exit 1
fi
ok "$("$BINLINK" --version)"

tools="$("$BINLINK" tools list 2>/dev/null | head -1 || true)"
case "$tools" in
  *tool*) ok "$tools" ;;
  *) warn "tools did not load; run '$BINLINK doctor' to see why" ;;
esac

case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *)
    echo
    warn "$PREFIX/bin is not on your PATH. Add it:"
    echo "        echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.bashrc"
    ;;
esac

cat <<DONE

Next:
  ozgent pull unsloth/Qwen3.5-4B-GGUF:Q4_K_M    download a model
  ozgent                                        chat
  ozgent web                                    web UI on http://127.0.0.1:7333
  ozgent serve                                  API on http://127.0.0.1:7337/v1
  ozgent doctor                                 check the machine

Uninstall with:
  $LIBDIR/install.sh --uninstall
DONE
