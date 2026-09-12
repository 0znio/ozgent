#!/usr/bin/env bash
# Build and install ozgent from source.
#
# Works out what your machine is, installs what the build needs, picks a GPU
# backend, compiles, and puts the binary on your PATH. Safe to re-run.
#
#   ./install.sh                     figure everything out
#   ./install.sh --backend cpu       no GPU
#   ./install.sh --prefix ~/.local   somewhere else (no sudo needed)
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
NO_SERVICE=0

usage() {
  cat <<'USAGE'
Usage: ./install.sh [options]

  --prefix DIR     where to install (default: /usr/local, using sudo to copy)
  --backend NAME   cuda | vulkan | metal | cpu | auto   (default: auto)
  --jobs N         parallel compile jobs (default: chosen from RAM and cores)
  --skip-deps      do not install system packages
  --yes            do not ask before installing packages
  --dry-run        print what would happen, change nothing
  --no-service     do not offer to run ozgent in the background
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
    --no-service) NO_SERVICE=1; shift ;;
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

# Whether ozgent already has a service installed, whatever the init system.
# Asking ozgent itself rather than guessing at paths keeps the two from
# disagreeing about where a service lives.
ozgent_service_installed() {
  [ -x "$BINLINK" ] || return 1
  "$BINLINK" daemon status 2>/dev/null | grep -q '^service *[^ ]' &&
    ! "$BINLINK" daemon status 2>/dev/null | grep -q 'not installed'
}

# Pick up a newly installed binary. Each init spells this differently, and a
# failure is not worth stopping an install over.
restart_service() {
  case "$("$BINLINK" daemon status 2>/dev/null | awk '/^init/ {print $2}')" in
    systemd) run systemctl --user restart ozgent 2>/dev/null || true ;;
    launchd) run launchctl kickstart -k "gui/$(id -u)/com.ozgent.daemon" 2>/dev/null || true ;;
    dinit)   run dinitctl restart ozgent 2>/dev/null || true ;;
    *)       note "restart it yourself so it picks up the new binary" ;;
  esac
}

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

# /usr/local, like every other program installed from source: on every
# user's PATH, in every shell, with no line to add to a dotfile. Only the copy
# into it needs root; the build runs as you. `--prefix ~/.local` is there for
# a machine where sudo is not an option.
PREFIX="${PREFIX:-/usr/local}"
LIBDIR="$PREFIX/lib/ozgent"
BINLINK="$PREFIX/bin/ozgent"

# Where earlier versions of this script installed by default when not run as
# root. Found and cleared on the way to the new place — left behind, its link
# would sit earlier on PATH than /usr/local/bin and keep running the old build.
LEGACY_PREFIX="${OZGENT_LEGACY_PREFIX:-$HOME/.local}"

# sudo only where it is actually needed, and not at all as root.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if have sudo; then SUDO="sudo"; elif have doas; then SUDO="doas"; fi
fi

# Whether `$1/lib/ozgent` is an install this script made, and `$1/bin/ozgent`
# the link it made to it — the only thing ever removed from a prefix that is
# not the one being installed to.
ours() {
  [ -x "$1/lib/ozgent/bin/ozgent" ] && [ -L "$1/bin/ozgent" ] &&
    [ "$(readlink "$1/bin/ozgent")" = "$1/lib/ozgent/bin/ozgent" ]
}

# ---------------------------------------------------------------- uninstall

if [ "$UNINSTALL" = "1" ]; then
  step "Removing ozgent"
  removed=0
  # Before the binary goes: `ozgent daemon uninstall` is what knows where the
  # service file is, and it cannot answer once it has been deleted.
  if [ -x "$BINLINK" ]; then
    run "$BINLINK" daemon uninstall >/dev/null 2>&1 || true
  fi
  for target in "$LIBDIR" "$BINLINK"; do
    if [ -e "$target" ] || [ -L "$target" ]; then
      if [ -w "$(dirname "$target")" ]; then run rm -rf "$target"
      else run $SUDO rm -rf "$target"
      fi
      ok "removed $target"; removed=1
    fi
  done
  if [ "$PREFIX" != "$LEGACY_PREFIX" ] && ours "$LEGACY_PREFIX"; then
    run rm -rf "$LEGACY_PREFIX/lib/ozgent" "$LEGACY_PREFIX/bin/ozgent"
    ok "removed the older install in $LEGACY_PREFIX"; removed=1
  fi
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
      void)                        PM="xbps";   break ;;
      gentoo)                      PM="emerge"; break ;;
      solus)                       PM="eopkg";  break ;;
      # Nothing is installed imperatively on NixOS, so there is nothing here
      # to do — the flags below skip the dependency step and say why.
      nixos)                       PM="nix";    break ;;
    esac
  done
  # Older releases of some distributions have no ID_LIKE at all. The command
  # that is present is then the only evidence, and it is good evidence.
  if [ -z "$PM" ]; then
    for candidate in pacman:pacman apt-get:apt dnf:dnf zypper:zypper apk:apk \
                     xbps-install:xbps emerge:emerge eopkg:eopkg; do
      command -v "${candidate%%:*}" >/dev/null 2>&1 && { PM="${candidate##*:}"; break; }
    done
  fi
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

# Needs both halves: the loader to run against, and glslc to compile the
# shaders at build time. Globbed rather than piped, for the reason above.
has_vulkan() {
  local d f
  have glslc || return 1
  have vulkaninfo && return 0
  for d in /usr/lib /usr/lib64 /usr/local/lib /usr/lib/x86_64-linux-gnu \
           /usr/lib/aarch64-linux-gnu; do
    for f in "$d"/libvulkan.so*; do
      [ -e "$f" ] && return 0
    done
  done
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

# What each requirement is called, per package manager. Keyed by the thing
# actually needed rather than by package name, because that is what can be
# tested for — `cmake` either runs or it does not, whereas "is base-devel
# installed" has no portable answer.
pkg_for() {
  case "$1:$PM" in
    cmake:emerge)     echo "dev-build/cmake" ;;
    cmake:*)          echo "cmake" ;;
    git:emerge)       echo "dev-vcs/git" ;;
    git:*)            echo "git" ;;
    curl:emerge)      echo "net-misc/curl" ;;
    curl:*)           echo "curl" ;;

    python:pacman)    echo "python" ;;
    python:xbps)      echo "python3" ;;
    python:emerge)    echo "dev-lang/python" ;;
    python:*)         echo "python3" ;;

    compiler:pacman)  echo "base-devel" ;;
    compiler:apt)     echo "build-essential" ;;
    compiler:dnf)     echo "gcc gcc-c++ make" ;;
    compiler:zypper)  echo "gcc gcc-c++ make" ;;
    compiler:apk)     echo "build-base" ;;
    compiler:xbps)    echo "base-devel" ;;
    compiler:emerge)  echo "sys-devel/gcc" ;;
    compiler:eopkg)   echo "-c system.devel" ;;
    compiler:brew)    echo "" ;;          # Xcode command line tools, not brew

    # bindgen loads libclang at run time; without it the mtmd bindings fail
    # with a message about a shared library rather than about a package.
    libclang:pacman)  echo "clang" ;;
    libclang:apt)     echo "libclang-dev" ;;
    libclang:dnf)     echo "clang-devel" ;;
    libclang:zypper)  echo "clang-devel" ;;
    libclang:apk)     echo "clang-dev" ;;
    libclang:xbps)    echo "clang" ;;
    libclang:emerge)  echo "sys-devel/clang" ;;
    libclang:eopkg)   echo "llvm-clang-devel" ;;
    libclang:brew)    echo "llvm" ;;

    pkgconfig:pacman) echo "pkgconf" ;;
    pkgconfig:apt)    echo "pkg-config" ;;
    pkgconfig:dnf)    echo "pkgconf-pkg-config" ;;
    pkgconfig:xbps)   echo "pkg-config" ;;
    pkgconfig:emerge) echo "dev-util/pkgconf" ;;
    pkgconfig:*)      echo "pkg-config" ;;

    cuda:pacman)      echo "cuda" ;;
    cuda:apt)         echo "nvidia-cuda-toolkit" ;;
    cuda:*)           echo "" ;;          # not in the default repos

    vulkan:pacman)    echo "vulkan-headers vulkan-icd-loader shaderc" ;;
    vulkan:apt)       echo "libvulkan-dev glslc spirv-tools" ;;
    vulkan:dnf)       echo "vulkan-headers vulkan-loader-devel glslc" ;;
    vulkan:zypper)    echo "vulkan-devel shaderc" ;;
    vulkan:xbps)      echo "vulkan-loader-devel Vulkan-Headers shaderc" ;;
    vulkan:*)         echo "" ;;
  esac
}

have_compiler() { have c++ || have g++ || have clang++; }

# A tool can be installed and still not run. A partial upgrade leaves the
# binary in place linked against a library that is no longer there: `command
# -v` says present, and the build says "is `cmake` not installed?" — both
# wrong, in a way that costs an afternoon. So the ones the build depends on
# are actually executed, and the loader's own complaint is shown.
BROKEN_TOOL=""
BROKEN_WHY=""
runs() {
  local out
  if out="$("$@" 2>&1)"; then
    return 0
  fi
  BROKEN_TOOL="$1"
  BROKEN_WHY="$(printf '%s' "$out" | head -1)"
  return 1
}

# Present, runnable, or broken — three outcomes, not two.
check_runnable() {
  local what="$1" label="$2"; shift 2
  if ! have "$1"; then
    NEEDED="$NEEDED $what"
    warn "$label — missing"
    return 0
  fi
  if runs "$@"; then
    ok "$label"
    return 0
  fi
  bad "$label is installed but will not run"
  note "$BROKEN_WHY"
  case "$BROKEN_WHY" in
    *"error while loading shared libraries"*|*"cannot open shared object"*)
      note "A library it needs is missing or the wrong version — usually a"
      note "half-finished system upgrade. Bring the system up to date:"
      case "$PM" in
        pacman) note "  sudo pacman -Syu" ;;
        apt)    note "  sudo apt update && sudo apt full-upgrade" ;;
        dnf)    note "  sudo dnf upgrade" ;;
        zypper) note "  sudo zypper dup" ;;
        *)      note "  (however your distribution does a full upgrade)" ;;
      esac
      ;;
  esac
  die "$1 cannot run. Fix that, then re-run ./install.sh."
}

# bindgen dlopen()s libclang at run time, so what matters is whether the
# shared object exists — not whether a compiler is installed.
#
# Tested by expanding globs rather than by piping `ls` or `ldconfig` into
# `grep`: this script runs under `pipefail`, and a pipeline whose first command
# exits non-zero fails even when the grep matched. `ls a b` where only `a`
# exists is exactly that, and it reported libclang missing on a machine that
# had three copies of it.
have_libclang() {
  local d f
  for d in /usr/lib /usr/lib64 /usr/local/lib /usr/lib/x86_64-linux-gnu \
           /usr/lib/aarch64-linux-gnu /usr/lib/llvm*/lib \
           /opt/homebrew/opt/llvm/lib /usr/local/opt/llvm/lib; do
    for f in "$d"/libclang.so* "$d"/libclang.dylib; do
      [ -e "$f" ] && return 0
    done
  done
  if have llvm-config; then
    d="$(llvm-config --libdir 2>/dev/null || true)"
    [ -n "$d" ] && { [ -e "$d/libclang.so" ] || [ -e "$d/libclang.dylib" ]; } && return 0
  fi
  return 1
}

install_pkgs() {
  [ $# -eq 0 ] && return 0
  case "$PM" in
    # `-Syu`, not `-S`. Arch does not support partial upgrades: installing a
    # package against a stale database pulls in a build that expects libraries
    # newer than the ones on disk, and the result is a binary that exists and
    # will not load — `cmake: error while loading shared libraries`. Doing that
    # to someone's machine while installing something else is not acceptable,
    # so the system is brought up to date first. That is what the prompt warns
    # about.
    pacman) run $SUDO pacman -Syu --needed --noconfirm "$@" ;;
    apt)    run $SUDO apt-get update -qq && run $SUDO apt-get install -y "$@" ;;
    dnf)    run $SUDO dnf install -y "$@" ;;
    zypper) run $SUDO zypper --non-interactive install "$@" ;;
    apk)    run $SUDO apk add --no-cache "$@" ;;
    xbps)   run $SUDO xbps-install -Sy "$@" ;;
    emerge) run $SUDO emerge --noreplace "$@" ;;
    eopkg)  run $SUDO eopkg install -y "$@" ;;
    brew)   run brew install "$@" ;;
    # NixOS installs nothing imperatively; `nix-shell -p` is the equivalent
    # and is the user's call, not this script's.
    nix)    return 1 ;;
    *)      return 1 ;;
  esac
}

step "Dependencies"

# What this machine is missing, by requirement. Reported either way, so
# "already installed" is visible rather than inferred from silence.
NEEDED=""
# The test is passed as a command and its arguments, not as one string: a
# quoted "have cmake" is a command with a space in its name, and bash says so
# in a way that reads exactly like cmake being absent.
check() {
  what="$1"; label="$2"; shift 2
  if "$@"; then
    ok "$label"
  else
    NEEDED="$NEEDED $what"
    warn "$label — missing"
  fi
}
check_runnable cmake "cmake" cmake --version
check compiler  "C++ compiler" have_compiler
check git       "git"          have git
check curl      "curl"         have curl
check python    "python3"      have python3
check libclang  "libclang"     have_libclang
have pkg-config || have pkgconf || NEEDED="$NEEDED pkgconfig"

case "$BACKEND" in
  cuda)   find_nvcc || NEEDED="$NEEDED cuda" ;;
  vulkan) has_vulkan || NEEDED="$NEEDED vulkan" ;;
esac

# shellcheck disable=SC2086
set -- $NEEDED
if [ $# -eq 0 ]; then
  ok "nothing to install"
elif [ "$SKIP_DEPS" = "1" ]; then
  warn "missing:$NEEDED — but --skip-deps was given"
elif [ -z "$PM" ]; then
  bad "missing:$NEEDED, and there is no known package manager here"
  note "Install them by hand, then re-run with --skip-deps."
  exit 1
elif [ "$PM" = "nix" ]; then
  # Nothing on NixOS is installed imperatively, so offering to try would only
  # fail confusingly. The shell that does work is one line, so it is given.
  bad "missing:$NEEDED"
  note "On NixOS, build inside a shell that has them:"
  note "  nix-shell -p cmake gcc clang pkg-config python3 git curl --run './install.sh --skip-deps'"
  exit 1
else
  packages=""
  for requirement in "$@"; do
    packages="$packages $(pkg_for "$requirement")"
  done
  # shellcheck disable=SC2086
  set -- $packages
  if [ $# -eq 0 ]; then
    warn "missing:$NEEDED, but $PM has no package for it here"
  else
    echo "  ${B}$PM${R} will install:"
    note "$*"
    [ "$PM" = "pacman" ] && note "and update the rest of the system, because Arch does not support partial upgrades"
    if confirm "Install them?"; then
      # Not "continuing and hoping": a package manager that failed turns into
      # a compiler error ten minutes later, and the real reason is here.
      install_pkgs "$@" || die "$PM could not install those. Fix that, then re-run."
      ok "installed"
    else
      warn "skipped — the build will probably fail"
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

# Checked here rather than trusted from the step above: packages can install
# and still leave nothing on PATH — a distribution that puts the compiler
# somewhere unusual, a `brew` that needs `brew link`, a sudo that succeeded for
# a different user. Failing now names the missing tool; failing later produces
# an error from inside a build script that names a crate instead.
if [ "$DRY_RUN" != "1" ]; then
  blocked=""
  { have cmake && runs cmake --version; } || blocked="$blocked cmake"
  have_compiler   || blocked="$blocked a-C++-compiler"
  have_libclang   || blocked="$blocked libclang"
  [ "$BACKEND" = "cuda" ] && { find_nvcc || blocked="$blocked nvcc"; }
  if [ -n "$blocked" ]; then
    bad "cannot build: not on PATH:$blocked"
    note "Installed but not found? Check PATH, then re-run."
    note "Or build without a GPU:  ./install.sh --backend cpu"
    die "missing build tools."
  fi
  ok "toolchain: cmake $(cmake --version | head -1 | awk '{print $3}'), $( (c++ --version 2>/dev/null || g++ --version) | head -1 | cut -c1-40)"
fi

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

# Saying which it is answers the question anyone re-running this has.
if [ -x "$LIBDIR/bin/ozgent" ]; then
  step "Updating the install at $LIBDIR"
  note "replacing $("$LIBDIR/bin/ozgent" --version 2>/dev/null || echo "an earlier build")"
else
  step "Installing"
fi

writable_or_sudo() {
  if [ -w "$1" ] || { [ ! -e "$1" ] && mkdir -p "$1" 2>/dev/null; }; then echo ""; else echo "$SUDO"; fi
}
AS="$(writable_or_sudo "$PREFIX")"
if [ -n "$AS" ]; then
  note "$PREFIX belongs to root; $AS will ask for your password to copy the files"
elif [ ! -w "$PREFIX" ]; then
  die "cannot write to $PREFIX and there is no sudo or doas. Run as root, or: ./install.sh --prefix ~/.local"
fi

run $AS mkdir -p "$LIBDIR/bin" "$PREFIX/bin"
# Replaced wholesale rather than merged, so a file from an older version
# cannot survive into a new install.
run $AS rm -rf "$LIBDIR/python" "$LIBDIR/docs"

# Unlinked before it is replaced. Copying over a binary that is currently
# executing fails with ETXTBSY — "text file busy" — so updating while
# `ozgent web` or `ozgent gateway` is running would otherwise stop here.
# Unlinking never fails that way, and a process already running keeps the
# inode it started from until it exits.
run $AS rm -f "$LIBDIR/bin/ozgent"
run $AS cp "$BINARY" "$LIBDIR/bin/ozgent"
run $AS cp -r "$SRC/python" "$LIBDIR/python"
[ -d "$SRC/docs" ] && run $AS cp -r "$SRC/docs" "$LIBDIR/docs"

# Updated in place rather than replaced, for two reasons. `cp -r a/bridge
# b/bridge` copies *into* the target once it exists, so a second run would
# nest it as b/bridge/bridge. And `ozgent gateway whatsapp` puts a
# node_modules under here — thirty megabytes the user was told to install,
# which an update should not silently throw away.
if [ -d "$SRC/bridge" ]; then
  run $AS mkdir -p "$LIBDIR/bridge"
  run $AS cp -a "$SRC/bridge/." "$LIBDIR/bridge/"
fi
run $AS chmod 755 "$LIBDIR/bin/ozgent"

# Moving from the old default. The WhatsApp bridge's node_modules is carried
# over first — the user installed it on purpose and should not have to again.
if [ "$PREFIX" != "$LEGACY_PREFIX" ] && ours "$LEGACY_PREFIX"; then
  old_modules="$LEGACY_PREFIX/lib/ozgent/bridge/whatsapp/node_modules"
  if [ -d "$old_modules" ] && [ ! -d "$LIBDIR/bridge/whatsapp/node_modules" ]; then
    run $AS mkdir -p "$LIBDIR/bridge/whatsapp"
    run $AS cp -a "$old_modules" "$LIBDIR/bridge/whatsapp/"
  fi
  run rm -rf "$LEGACY_PREFIX/lib/ozgent" "$LEGACY_PREFIX/bin/ozgent"
  ok "moved from $LEGACY_PREFIX to $PREFIX (the old copy is removed)"
  MOVED=1
fi

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

# The shell remembers where it last found `ozgent`; after a move it would
# keep looking in the old place.
if [ "${MOVED:-0}" = "1" ]; then
  note "if 'ozgent' says 'no such file' in an open shell, run: hash -r"
fi

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

# ------------------------------------------------------------- the daemon

# ozgent knows which init system this is and how to write a service for it —
# systemd, OpenRC, runit, s6, dinit or launchd. All this has to decide is
# whether to ask.
step "Background service"

if [ "$NO_SERVICE" = "1" ]; then
  note "skipped (--no-service)"
elif [ "${MOVED:-0}" = "1" ] || ozgent_service_installed; then
  # Already there: re-running the installer replaces the binary underneath it,
  # so the service only needs a nudge to pick the new one up.
  ok "a service is already installed"
  restart_service
else
  note "One ozgent in the background means scheduled jobs run, Telegram and"
  note "WhatsApp are answered, and nothing loads a second copy of the model."
  note "It holds no model at all until something asks it a question."
  if confirm "Run ozgent in the background, starting at login?"; then
    run "$BINLINK" daemon install || warn "could not install the service; 'ozgent daemon install' says why"
  else
    note "later:  ozgent daemon install"
  fi
fi

cat <<DONE

${B}Done.${R} Next:

  ozgent pull unsloth/Qwen3.5-4B-GGUF:Q4_K_M    a model to start with
  ozgent                                        chat in the terminal
  ozgent daemon install                         run it in the background
  ozgent web                                    browser, http://localhost:7333
  ozgent doctor                                 what this machine can do

Uninstall:  $SRC/install.sh --uninstall
DONE
