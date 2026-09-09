# Installing ozgent on another machine

Two ways, for two situations.

**From source**, on a machine that can compile: clone the repository and run
`./install.sh`. It handles the distribution's packages, the GPU backend, the
build and the install. That is the normal path and this document is not about
it — `./install.sh --help` is.

**From a bundle**, for a machine that cannot or should not compile: the rest of
this page. Nothing is compiled on the target; the bundle carries a prebuilt
binary, the Python tools it spawns, and an installer.

## Build the bundle (on a machine with the toolchain)

```bash
./scripts/package.sh
```

Produces `dist/ozgent-<version>-linux-x86_64-cuda.tar.gz` (~477 MB) and a
`.sha256` beside it. `SKIP_BUILD=1` reuses the existing release binary instead
of rebuilding.

## Install (on the target)

```bash
tar xzf ozgent-0.1.0-linux-x86_64-cuda.tar.gz
cd ozgent-0.1.0-linux-x86_64-cuda
./install.sh                 # /usr/local as root, ~/.local otherwise
```

Options: `--prefix DIR`, `--force` to install despite a failed check,
`--uninstall` to remove it.

## What the target needs

| requirement | why |
|---|---|
| **glibc 2.39+** | the binary links against the system C library |
| **libstdc++ with GLIBCXX 3.4.31+** | same, for the C++ side |
| **the NVIDIA driver** (`libcuda.so.1`) | a hard dependency of this build |
| python3 | only for the tools; ozgent runs without it |

Ubuntu 24.04, Debian 13, Fedora 40 and Arch are new enough. **Ubuntu 22.04 and
Debian 12 are not** — their glibc is too old and the binary will not load.

The CUDA **toolkit** is *not* required. The CUDA runtime and cuBLAS are linked
statically; only the driver that ships with the graphics card is needed. That
is also why this build cannot start on a machine with no NVIDIA driver at all,
even to run on the CPU: `libcuda.so.1` is resolved at load time.

The installer checks all of this before copying anything, and says which
distributions qualify when the check fails.

## What goes where

```
$PREFIX/lib/ozgent/bin/ozgent      the binary
$PREFIX/lib/ozgent/python/         the tools it spawns
$PREFIX/lib/ozgent/install.sh      kept for --uninstall
$PREFIX/bin/ozgent                 symlink
```

Everything lives under one directory, so removing that directory removes
ozgent. The symlink matters: the binary finds its Python tools by walking up
from its own path, which is why it is linked rather than copied into `bin/`.

Your own data — models, settings, logs — lives in `~/ozgent` and is never
touched by installing, upgrading or uninstalling.

## Verifying

The installer runs these itself and reports them, but by hand:

```bash
ozgent --version
ozgent tools list      # should report the tools that loaded
ozgent doctor          # hardware, backends, what is misconfigured
```

## Upgrading

Unpack the new bundle and run `./install.sh` again. It replaces
`bin/` and `python/` wholesale rather than merging, so a file from an older
version cannot survive into a new install.
