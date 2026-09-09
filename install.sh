#!/usr/bin/env bash
# Build and install ozgent from source.
#
# Works out what your machine is, installs what the build needs, picks a GPU
# backend, compiles, and puts the binary on your PATH. Safe to re-run.
#
#   ./install.sh                     figure everything out
#   ./install.sh --backend cpu       no GPU
#   ./install.sh --prefix ~/.local   somewhere else
#   ./install.sh --uninstall         remove it again
#
# This is the from-source installer. `scripts/install.sh` is a different thing:
# it lives inside a release tarball and installs a binary that is already
# built. Nothing here is compiled twice.
set -euo pipefail

REPO_URL="https://github.com/0znio/ozgent"
PREFIX=""
BACKEND="auto"
JOBS=""
ASSUME_YES=0
SKIP_DEPS=0
DRY_RUN=0
UNINSTALL=0

usage() {
  cat <<'USAGE'
Usage: ./install.sh [options]

  --prefix DIR     where to install (default: /usr/local as root, else ~/.local)
  --backend NAME   cuda | vulkan | metal | cpu | auto   (default: auto)
  --jobs N         parallel compile jobs (default: chosen from RAM and cores)
  --skip-deps      do not install system packages
  --yes            do not ask before installing packages
  --dry-run        print what would happen, change nothing
  --uninstall      remove a previous installation and exit
  -h, --help       this

Models and settings live in ~/ozgent and are never touched.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)    PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
    --backend)   BACKEND="${2:?--backend needs a name}"; shift 2 ;;
    --jobs|-j)   JOBS="${2:?--jobs needs a number}"; shift 2 ;;
    --skip-deps) SKIP_DEPS=1; shift ;;
    --yes|-y)    ASSUME_YES=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$BACKEND" in
  auto|cuda|vulkan|metal|cpu) ;;
  *) echo "unknown backend: $BACKEND (try cuda, vulkan, metal, cpu or auto)" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------- output

if [ -t 1 ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; R=$'\033[0m'
else
  B=""; DIM=""; GREEN=""; YELLOW=""; RED=""; R=""
fi
step() { printf '\n%s==>%s %s%s%s\n' "$GREEN" "$R" "$B" "$*" "$R"; }
ok()   { printf '  %sok%s    %s\n' "$GREEN" "$R" "$*"; }
warn() { printf '  %swarn%s  %s\n' "$YELLOW" "$R" "$*"; }
bad()  { printf '  %sno%s    %s\n' "$RED" "$R" "$*"; }
note() { printf '        %s%s%s\n' "$DIM" "$*" "$R"; }
die()  { printf '\n%serror%s %s\n' "$RED" "$R" "$*" >&2; exit 1; }

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '  %swould run%s  %s\n' "$DIM" "$R" "$*"
    return 0
  fi
  "$@"
}

have() { command -v "$1" >/dev/null 2>&1; }

# Ask, unless told not to. Defaults to yes, because someone who ran an
# installer has already said what they want.
confirm() {
  [ "$ASSUME_YES" = "1" ] && return 0
  [ "$DRY_RUN" = "1" ] && return 0
  [ -t 0 ] || return 0
  printf '  %s [Y/n] ' "$1"
  read -r reply
  case "$reply" in [nN]*) return 1 ;; *) return 0 ;; esac
}

# ---------------------------------------------------------------- locations

if [ -z "$PREFIX" ]; then
  if [ "$(id -u)" -eq 0 ]; then PREFIX=/usr/local; else PREFIX="$HOME/.local"; fi
fi
LIBDIR="$PREFIX/lib/ozgent"
BINLINK="$PREFIX/bin/ozgent"

# sudo only where it is actually needed, and not at all as root.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if have sudo; then SUDO="sudo"; elif have doas; then SUDO="doas"; fi
fi

# ---------------------------------------------------------------- uninstall

if [ "$UNINSTALL" = "1" ]; then
  step "Removing ozgent"
  removed=0
  for target in "$LIBDIR" "$BINLINK"; do
    if [ -e "$target" ] || [ -L "$target" ]; then
      if [ -w "$(dirname "$target")" ]; then run rm -rf "$target"
      else run $SUDO rm -rf "$target"
      fi
      ok "removed $target"; removed=1
    fi
  done
  [ "$removed" = "0" ] && warn "nothing installed at $PREFIX"
  note "Your models and settings in ~/ozgent were left alone."
  note "Remove them too with:  rm -rf ~/ozgent"
  exit 0
fi

# ------------------------------------------------------------ what is this

step "Looking at this machine"

OS="$(uname -s)"
ARCH="$(uname -m)"
DISTRO="unknown"; DISTRO_NAME="$OS"; PM=""

if [ "$OS" = "Darwin" ]; then
  DISTRO="macos"; DISTRO_NAME="macOS $(sw_vers -productVersion 2>/dev/null || echo '')"
  PM="brew"
elif [ -r /etc/os-release ]; then
  . /etc/os-release
  DISTRO="${ID:-unknown}"
  DISTRO_NAME="${PRETTY_NAME:-$DISTRO}"
  # ID_LIKE is what makes derivatives work without listing every one of them.
  for candidate in $DISTRO ${ID_LIKE:-}; do
    case "$candidate" in
      arch)                        PM="pacman"; break ;;
      debian|ubuntu)               PM="apt";    break ;;
      fedora|rhel|centos)          PM="dnf";    break ;;
      opensuse*|suse|sles)         PM="zypper"; break ;;
      alpine)                      PM="apk";    break ;;
    esac
  done
fi

ok "$DISTRO_NAME ($ARCH)"
if [ -n "$PM" ]; then ok "package manager: $PM"
else warn "unrecognised distribution — install dependencies yourself, then re-run with --skip-deps"
fi

if [ "$OS" = "Linux" ] && [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
  warn "$ARCH is untested; the build may not work"
fi

# ------------------------------------------------------------------- memory

total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$total_kb" -gt 0 ]; then
  total_gb=$(( total_kb / 1024 / 1024 ))
  ok "${total_gb} GB RAM, $(nproc 2>/dev/null || echo '?') cores"
  [ "$total_gb" -lt 4 ] && warn "under 4 GB — the C++ build may run out of memory"
fi

# ------------------------------------------------------------------ the GPU

step "Choosing a backend"

CUDA_HOME=""
find_nvcc() {
  have nvcc && { CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; return 0; }
  # Arch keeps it out of PATH; the others are where the .run installer lands.
  for d in /opt/cuda /usr/local/cuda /usr/lib/cuda; do
    [ -x "$d/bin/nvcc" ] && { CUDA_HOME="$d"; return 0; }
  done
  return 1
}

has_nvidia_gpu() {
  have nvidia-smi && nvidia-smi -L >/dev/null 2>&1 && return 0
  [ -e /dev/nvidiactl ] && return 0
  return 1
}

has_vulkan() {
  have glslc || return 1
  have vulkaninfo && return 0
  ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1' && return 0
  return 1
}

if [ "$BACKEND" = "auto" ]; then
  if [ "$OS" = "Darwin" ]; then
    BACKEND="metal"; ok "Apple platform — using Metal"
  elif has_nvidia_gpu; then
    ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo detected)"
    if find_nvcc; then BACKEND="cuda"; ok "CUDA toolkit at $CUDA_HOME"
    else BACKEND="cuda"; warn "no CUDA toolkit yet — will try to install one"
    fi
  elif has_vulkan; then
    BACKEND="vulkan"; ok "Vulkan available — using it"
  else
    BACKEND="cpu"; ok "no usable GPU found — building for the CPU"
  fi
else
  ok "backend: $BACKEND (you asked for it)"
  [ "$BACKEND" = "cuda" ] && find_nvcc >/dev/null || true
fi

[ "$BACKEND" = "metal" ] && [ "$OS" != "Darwin" ] && die "Metal only exists on Apple platforms."

case "$BACKEND" in
  cpu) FEATURES="" ;;
  *)   FEATURES="$BACKEND" ;;
esac

# -------------------------------------------------------------- dependencies

# Packages every backend needs: a C++ compiler, CMake for llama.cpp, git,
# curl for rustup, and python3 for the tools.
pkgs_for() {
  case "$PM" in
    pacman) echo "base-devel cmake git curl python pkgconf" ;;
    apt)    echo "build-essential cmake git curl python3 pkg-config libclang-dev" ;;
    dnf)    echo "gcc gcc-c++ make cmake git curl python3 pkgconf-pkg-config clang-devel" ;;
    zypper) echo "gcc gcc-c++ make cmake git curl python3 pkg-config clang-devel" ;;
    apk)    echo "build-base cmake git curl python3 pkgconf clang-dev" ;;
    brew)   echo "cmake git python3" ;;
  esac
}

# bindgen needs libclang; the CUDA and Vulkan builds need their own toolchains.
extra_pkgs_for() {
  case "$BACKEND:$PM" in
    cuda:pacman)  echo "cuda" ;;
    cuda:apt)     echo "nvidia-cuda-toolkit" ;;
    cuda:dnf)     echo "" ;;   # not in the default repos; handled below
    cuda:zypper)  echo "" ;;
    vulkan:pacman) echo "vulkan-headers vulkan-icd-loader shaderc" ;;
    vulkan:apt)    echo "libvulkan-dev glslc spirv-tools" ;;
    vulkan:dnf)    echo "vulkan-headers vulkan-loader-devel glslc" ;;
    vulkan:zypper) echo "vulkan-devel shaderc" ;;
    *) echo "" ;;
  esac
}

install_pkgs() {
  [ $# -eq 0 ] && return 0
  case "$PM" in
    pacman) run $SUDO pacman -S --needed --noconfirm "$@" ;;
    apt)    run $SUDO apt-get update -qq && run $SUDO apt-get install -y "$@" ;;
    dnf)    run $SUDO dnf install -y "$@" ;;
    zypper) run $SUDO zypper --non-interactive install "$@" ;;
    apk)    run $SUDO apk add --no-cache "$@" ;;
    brew)   run brew install "$@" ;;
    *)      return 1 ;;
  esac
}

step "Dependencies"

if [ "$SKIP_DEPS" = "1" ]; then
  warn "skipped (--skip-deps)"
elif [ -z "$PM" ]; then
  warn "no known package manager; install these yourself:"
  note "a C++ compiler, cmake, git, curl, python3, and libclang"
else
  # Only the missing ones, so a re-run is quiet.
  want="$(pkgs_for) $(extra_pkgs_for)"
  missing=""
  need_cmd() { have "$1" || missing="$missing $2"; }
  need_cmd cmake  cmake
  need_cmd git    git
  need_cmd curl   curl
  need_cmd python3 python3
  have cc || have gcc || have clang || missing="$missing compiler"

  if [ -z "$missing" ] && [ "$BACKEND" = "cpu" ]; then
    ok "everything needed is already here"
  else
    # shellcheck disable=SC2086
    set -- $want
    echo "  ${B}$PM${R} will install:"
    note "$*"
    if confirm "Install them?"; then
      install_pkgs "$@" || warn "some packages failed; continuing and hoping"
      ok "packages installed"
    else
      warn "skipped — the build may fail"
    fi
  fi
fi

# CUDA on the distributions that do not ship it.
if [ "$BACKEND" = "cuda" ] && ! find_nvcc; then
  case "$PM" in
    dnf|zypper|apk|"")
      bad "no CUDA toolkit, and $PM does not carry one"
      note "Install it from https://developer.nvidia.com/cuda-downloads, then re-run."
      note "Or build without it:  ./install.sh --backend vulkan"
      note "                      ./install.sh --backend cpu"
      if has_vulkan && confirm "Use Vulkan instead?"; then
        BACKEND="vulkan"; FEATURES="vulkan"; ok "switched to Vulkan"
      elif confirm "Build for the CPU instead?"; then
        BACKEND="cpu"; FEATURES=""; ok "switched to CPU"
      else
        die "nothing to build with."
      fi
      ;;
    *)
      [ "$DRY_RUN" = "1" ] || die "CUDA toolkit still not found after installing. Re-run, or use --backend cpu."
      ;;
  esac
fi

# Arch puts nvcc in /opt/cuda, which is not on anyone's PATH.
if [ "$BACKEND" = "cuda" ] && [ -n "$CUDA_HOME" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export CUDA_PATH="$CUDA_HOME"
  ok "CUDA_PATH=$CUDA_HOME"
fi

# --------------------------------------------------------------------- rust

step "Rust"

if have cargo; then
  ok "$(cargo --version)"
else
  warn "cargo not found"
  if confirm "Install Rust with rustup?"; then
    run sh -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path'
    # shellcheck disable=SC1090,SC1091
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
    have cargo && ok "$(cargo --version)" || die "rustup finished but cargo is still not on PATH."
  else
    die "ozgent is written in Rust; cargo is required. See https://rustup.rs"
  fi
fi

if [ "$DRY_RUN" != "1" ] && have cargo; then
  # 1.85 is the edition-2024 floor. An older toolchain fails deep in a
  # dependency with an error that does not mention the version.
  rust_ver="$(cargo --version | awk '{print $2}')"
  major="${rust_ver%%.*}"; rest="${rust_ver#*.}"; minor="${rest%%.*}"
  if [ "${major:-0}" -eq 1 ] && [ "${minor:-0}" -lt 85 ]; then
    warn "Rust $rust_ver is older than the 1.85 this needs"
    confirm "Update it with rustup?" && run rustup update stable || die "Update Rust, then re-run."
  fi
fi

# ------------------------------------------------------------------- source

step "Source"

find_repo_root() {
  local dir="$1"
  for _ in 1 2 3 4 5; do
    [ -f "$dir/Cargo.toml" ] && grep -q 'ozgent-cli' "$dir/Cargo.toml" 2>/dev/null && { echo "$dir"; return 0; }
    dir="$(dirname "$dir")"
    [ "$dir" = "/" ] && break
  done
  return 1
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || pwd)"
if SRC="$(find_repo_root "$HERE")"; then
  ok "building from $SRC"
elif SRC="$(find_repo_root "$PWD")"; then
  ok "building from $SRC"
else
  # Piped from curl, so there is no checkout to build.
  SRC="${OZGENT_SRC:-$PWD/ozgent}"
  if [ -d "$SRC/.git" ]; then
    ok "updating $SRC"
    run git -C "$SRC" pull --ff-only
  else
    warn "no checkout here"
    confirm "Clone $REPO_URL into $SRC?" || die "Nothing to build. Clone the repository and run ./install.sh inside it."
    run git clone --depth 1 "$REPO_URL" "$SRC"
    ok "cloned into $SRC"
  fi
fi

# --------------------------------------------------------------------- build

step "Building"

if [ -z "$JOBS" ]; then
  JOBS="$(nproc 2>/dev/null || echo 4)"
  # The CUDA compile is memory-hungry: each nvcc can take a gigabyte or more,
  # and an out-of-memory kill halfway through a fifteen-minute build is a
  # miserable way to find that out.
  if [ "$BACKEND" = "cuda" ] && [ "${total_kb:-0}" -gt 0 ]; then
    by_ram=$(( total_kb / 1024 / 1024 / 2 ))
    [ "$by_ram" -lt 1 ] && by_ram=1
    [ "$by_ram" -lt "$JOBS" ] && JOBS="$by_ram"
  fi
fi

if [ -n "$FEATURES" ]; then
  ok "cargo build --release --features $FEATURES -j $JOBS"
else
  ok "cargo build --release -j $JOBS  (CPU only)"
fi
note "llama.cpp is compiled from source; the first build takes 10-30 minutes."

cd "$SRC"
if [ -n "$FEATURES" ]; then
  run cargo build --release --features "$FEATURES" -j "$JOBS"
else
  run cargo build --release -j "$JOBS"
fi

BINARY="$SRC/target/release/ozgent"
if [ "$DRY_RUN" != "1" ]; then
  [ -x "$BINARY" ] || die "the build finished but $BINARY is not there."
  ok "built $(du -h "$BINARY" | cut -f1)"
fi

# ------------------------------------------------------------------ install

step "Installing"

writable_or_sudo() {
  if [ -w "$(dirname "$1")" ] || mkdir -p "$1" 2>/dev/null; then echo ""; else echo "$SUDO"; fi
}
AS="$(writable_or_sudo "$PREFIX")"
[ -n "$AS" ] && note "$PREFIX needs elevation; using $AS"

run $AS mkdir -p "$LIBDIR/bin" "$PREFIX/bin"
# Replaced wholesale rather than merged, so a file from an older version
# cannot survive into a new install.
run $AS rm -rf "$LIBDIR/python" "$LIBDIR/docs"
run $AS cp "$BINARY" "$LIBDIR/bin/ozgent"
run $AS cp -r "$SRC/python" "$LIBDIR/python"
[ -d "$SRC/docs" ] && run $AS cp -r "$SRC/docs" "$LIBDIR/docs"
[ -d "$SRC/bridge" ] && run $AS cp -r "$SRC/bridge" "$LIBDIR/bridge"
run $AS chmod 755 "$LIBDIR/bin/ozgent"

# The binary finds its Python tools by walking up from its own path, which is
# why bin/ozgent is a link into LIBDIR rather than a copy.
run $AS ln -sfn "$LIBDIR/bin/ozgent" "$BINLINK"
ok "installed to $LIBDIR"
ok "linked $BINLINK"

# -------------------------------------------------------------------- check

if [ "$DRY_RUN" = "1" ]; then
  step "Dry run — nothing was changed"
  exit 0
fi

step "Checking"

"$BINLINK" --version >/dev/null 2>&1 || die "installed, but it will not run. Try: $BINLINK --version"
ok "$("$BINLINK" --version)"

tools="$("$BINLINK" tools list 2>/dev/null | head -1 || true)"
case "$tools" in
  *tool*) ok "$tools" ;;
  *)      warn "tools did not load — run '$BINLINK doctor' to see why" ;;
esac

case ":$PATH:" in
  *":$PREFIX/bin:"*) ok "$PREFIX/bin is on your PATH" ;;
  *)
    warn "$PREFIX/bin is not on your PATH"
    note "echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.bashrc && exec \$SHELL"
    ;;
esac

cat <<DONE

${B}Done.${R} Next:

  ozgent pull unsloth/Qwen3.5-4B-GGUF:Q4_K_M    a model to start with
  ozgent                                        chat in the terminal
  ozgent web                                    browser, http://localhost:7333
  ozgent doctor                                 what this machine can do

Uninstall:  $SRC/install.sh --uninstall
DONE
