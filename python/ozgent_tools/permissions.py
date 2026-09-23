"""What the model is allowed to touch.

Tools that only read something the user already pointed at need no ceremony.
Tools that write files, run commands, or reach the network are a different
matter: the model decides to use them, and the user finds out afterwards. So
those are off until switched on, and every refusal names the exact setting
that would allow it — a permission system nobody can work out how to grant is
one people disable wholesale.

Configured once, in ``~/ozgent/configs/config.toml``::

    [tools.config.permissions]
    root       = "/home/you/code"   # filesystem boundary for every file tool
    write      = false              # create or modify files
    shell      = false              # run commands
    shell_allow = ["git", "cargo", "ls"]   # and only these, by first word
    network    = true               # fetch pages by URL
    network_allow = []              # empty means any host

An individual tool may still override ``root`` under its own
``[tools.config.<tool>]`` section, which is how it worked before this existed.

Standing configuration is only half of it. ozgent also asks — at the moment of
the call, showing the tool and its arguments — and a person who reads
``run_command(command="git status")`` and says yes has authorised that call as
surely as a config flag would have. So an approved call satisfies the checks
here: refusing to run a command the user just approved, because a setting they
have never seen is off, is a permission system arguing with its own user.

What approval does *not* lift is the shell-metacharacter refusal below, which
is not a permission at all — commands run without a shell, so a pipe in one
would not do what it looks like it does.
"""

from __future__ import annotations

import contextvars
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from .base import ToolError, get_config

#: Set for the duration of a call the user explicitly approved.
#:
#: A context variable rather than a global because calls run concurrently on
#: one event loop: a plain flag set by one call would be visible to every
#: other in flight, which is precisely the sort of hole a permission system
#: must not have.
_APPROVED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ozgent_call_approved", default=False
)


def approving(approved: bool):
    """Mark the current call as approved, as a context manager."""

    class _Scope:
        def __enter__(self) -> None:
            self.token = _APPROVED.set(bool(approved))

        def __exit__(self, *exc: Any) -> None:
            _APPROVED.reset(self.token)

    return _Scope()


def approved() -> bool:
    """Whether a person authorised the call now running."""
    return _APPROVED.get()


#: Where the settings above live, as a tool name so the existing config
#: transport carries it without a second channel.
SECTION = "permissions"


def perms() -> dict[str, Any]:
    return get_config(SECTION)


def root_for(tool_name: str) -> Path:
    """The directory a tool's paths are confined to.

    A tool's own ``root`` wins over the shared one, and the working directory
    ozgent was started in is the fallback — the least surprising boundary, as
    a model asked about "this project" should not reach the rest of the disk.
    """
    own = get_config(tool_name).get("root")
    shared = perms().get("root")
    return Path(own or shared or os.getcwd()).expanduser().resolve()


def protected_dirs() -> list[Path]:
    """Directories no tool may touch, as ozgent passed them to the worker.

    ozgent's own home — its keys, access rules, database, agents and tools —
    and any other directory tools are loaded from. Read afresh on each call:
    cheap, and a test can set it.
    """
    raw = os.environ.get("OZGENT_PROTECTED", "")
    return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p]


#: Where credentials live. A tool reading one hands it to a model that a web
#: page may be steering, so these are refused like ozgent's own home — unless
#: ``[tools.config.permissions] allow_sensitive = true`` says otherwise.
SENSITIVE = (
    "~/.ssh", "~/.gnupg", "~/.aws", "~/.azure", "~/.config/gcloud", "~/.kube",
    "~/.docker", "~/.netrc", "~/.git-credentials", "~/.config/gh", "~/.config/hub",
    "~/.password-store", "~/.local/share/keyrings", "~/.pki", "~/.npmrc", "~/.pypirc",
    "~/.cargo/credentials", "~/.cargo/credentials.toml", "~/.vault-token",
    "~/.terraform.d", "~/.config/op", "~/.mozilla", "~/.config/google-chrome",
    "~/.config/chromium", "~/.config/BraveSoftware", "~/.config/Code/User/globalStorage",
)


def sensitive_dirs() -> list[Path]:
    """Credential locations, unless the operator has allowed them."""
    if perms().get("allow_sensitive", False):
        return []
    return [Path(p).expanduser().resolve() for p in SENSITIVE]


def refuse_protected(resolved: Path, shown: str) -> None:
    """Refuse a path inside a protected directory, approved or not.

    Approval does not lift this, unlike the root boundary: a person approving
    ``write_file`` in the middle of a conversation is judging the file they
    were shown, and a model steered by a web page is exactly what would ask
    to "fix a setting" in ozgent's own configuration. Changing who may use
    this machine happens through ozgent's own commands, never through a tool.
    """
    for directory in protected_dirs():
        if resolved == directory or directory in resolved.parents:
            raise ToolError(
                f"{shown!r} is inside {directory}, where ozgent keeps its own settings, "
                "keys, database and tools. No tool may read or change anything there, "
                "even with approval."
            )
    for directory in sensitive_dirs():
        if resolved == directory or directory in resolved.parents:
            raise ToolError(
                f"{shown!r} is inside {directory}, where credentials are kept. Tools may not "
                f"read or change them, even with approval, unless [tools.config.{SECTION}] "
                "allow_sensitive = true."
            )


def resolve_within(path: str, tool_name: str) -> Path:
    """Resolve `path`, refusing anything outside the tool's root.

    Resolution happens first so that `../` and symlinks cannot step outside;
    comparing the strings beforehand would be trivially defeated.
    """
    root = root_for(tool_name)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ToolError(f"cannot resolve {path!r}: {exc}") from exc

    refuse_protected(resolved, path)
    if resolved != root and root not in resolved.parents:
        # The prompt showed this path. Someone read it and said yes.
        if approved():
            return resolved
        raise ToolError(
            f"{path!r} is outside the permitted directory ({root}). "
            f"Set [tools.config.{SECTION}] root in config.toml to widen it."
        )
    return resolved


def require(flag: str, doing: str) -> None:
    """Refuse unless `flag` is on or the user approved this call."""
    if approved() or perms().get(flag, False):
        return
    raise ToolError(
        f"{doing} is not permitted. Approve it when ozgent asks, or set "
        f"[tools.config.{SECTION}] {flag} = true in ~/ozgent/configs/config.toml."
    )


def check_command(command: str) -> list[str]:
    """Split `command` and check it against the allowlist.

    An allowlist rather than a denylist, and matched on the program rather than
    the whole string: enumerating what is safe is possible, while enumerating
    what is dangerous is not. An empty allowlist permits nothing even with
    ``shell = true``, because "on" and "anything" should not be the same
    setting.
    """
    require("shell", "running commands")
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ToolError(f"cannot parse the command: {exc}") from exc
    if not argv:
        raise ToolError("no command given")

    # Shell metacharacters would let an allowed program introduce a disallowed
    # one. The command is run without a shell, so these cannot work anyway;
    # refusing them plainly beats running something that silently means
    # something else.
    for ch in ";|&><`$\n":
        if ch in command:
            raise ToolError(
                f"{ch!r} is not allowed: commands run without a shell, so pipes, "
                "redirection and chaining have no effect. Run one program."
            )

    # Arguments that name a path into ozgent's own home are refused even with
    # approval. Best effort, and said so: a program that finds files itself
    # (`find ~`) is not caught by looking at its arguments — only a sandbox
    # around the process would be.
    for arg in argv[1:]:
        value = arg.split("=", 1)[1] if arg.startswith("-") and "=" in arg else arg
        if value.startswith(("/", "~", ".")) or os.sep in value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = root_for("run_command") / candidate
            try:
                refuse_protected(candidate.resolve(), value)
            except OSError:
                pass

    # The allowlist answers "what may run without anyone looking". An approved
    # call was looked at — the whole command string was on screen — so it is
    # past the question the allowlist exists to answer.
    if approved():
        return argv

    allow = [str(a) for a in perms().get("shell_allow", [])]
    if not allow:
        raise ToolError(
            f"no commands are allowed. Approve this one when ozgent asks, or list "
            f"them in [tools.config.{SECTION}] shell_allow, e.g. "
            'shell_allow = ["git", "cargo"].'
        )
    program = Path(argv[0]).name
    if program not in allow:
        raise ToolError(
            f"{program!r} is not in the allowlist. Permitted: {', '.join(sorted(allow))}. "
            f"Add it to [tools.config.{SECTION}] shell_allow to permit it."
        )
    return argv


def private_allowed(host: str) -> bool:
    """Whether `host` may resolve to this machine or a private network.

    Off by default: a page the model reads can ask it to fetch anything, and
    "anything" includes this server's own API, a router's admin page and a
    cloud metadata address. An operator who wants a LAN host listed says so
    per host in ``network_allow``, or wholesale with ``network_private``.
    """
    if perms().get("network_private", False):
        return True
    allow = [str(a).lower() for a in perms().get("network_allow", [])]
    host = host.lower()
    return any(host == a or host.endswith("." + a) for a in allow)


def check_host(url: str) -> str:
    """Check a URL against the network allowlist, returning its host."""
    from urllib.parse import urlparse

    require("network", "fetching pages")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError(f"{parsed.scheme or 'that'!r} is not a URL scheme this can fetch; use http or https")
    host = parsed.hostname or ""
    if not host:
        raise ToolError(f"{url!r} has no host")

    allow = [str(a).lower() for a in perms().get("network_allow", [])]
    if approved():
        return host
    if allow and not any(host.lower() == a or host.lower().endswith("." + a) for a in allow):
        raise ToolError(
            f"{host} is not in the allowlist. Permitted: {', '.join(sorted(allow))}. "
            f"Add it to [tools.config.{SECTION}] network_allow, or clear the list to allow any host."
        )
    return host


# ------------------------------------------------------------ command sandbox

#: Readable and executable: where programs, libraries and their data live.
SYSTEM_READ = (
    "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/opt", "/etc",
    "/nix", "/snap", "/proc", "/sys/devices/system/cpu", "/sys/fs/cgroup",
)

#: Toolchains people keep in their home directory, read-only.
HOME_TOOLCHAINS = (
    "~/.cargo/bin", "~/.cargo/registry", "~/.cargo/git", "~/.rustup", "~/.local/bin",
    "~/.local/lib", "~/.local/share/uv", "~/.pyenv", "~/.nvm", "~/.volta", "~/.bun",
    "~/.deno", "~/go", "~/.local/share/mise", "~/.sdkman", "~/.m2", "~/.gradle",
)

#: Devices a program may open.
DEVICES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom")

#: Environment variables passed through. Everything else — keys, tokens, the
#: SSH agent, the session bus, the display — stays behind.
ENV_KEEP = (
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "TZ", "SHELL",
    "CARGO_HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN", "GOPATH", "GOROOT", "GOFLAGS",
    "JAVA_HOME", "VIRTUAL_ENV", "NVM_DIR", "PNPM_HOME",
)

_SECRETISH = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "COOKIE", "SESSION", "AUTH")


def clean_env(tmp: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in ENV_KEEP or k.startswith("LC_")}
    env = {k: v for k, v in env.items() if not any(s in k.upper() for s in _SECRETISH)}
    env.update({"TMPDIR": tmp, "TERM": "dumb", "NO_COLOR": "1", "CARGO_TERM_COLOR": "never"})
    return env


def command_sandbox(program: str, cwd: Path, tmp: str) -> dict[str, Any]:
    """The sandbox a command runs in: see ``ozgent_tools/sandbox.py``.

    Decided here, with every other permission, so a tool never reads the
    flags for itself.
    """
    read = [p for p in SYSTEM_READ if os.path.exists(p)]
    read += [os.path.expanduser(p) for p in HOME_TOOLCHAINS if os.path.exists(os.path.expanduser(p))]
    # A virtualenv's interpreter and its libraries.
    read += sorted({sys.prefix, sys.base_prefix})
    extra = perms().get("shell_read", [])
    read += [os.path.expanduser(str(p)) for p in extra]
    found = shutil.which(program)
    if found:
        read.append(str(Path(found).resolve().parent))
    return {
        "cwd": str(cwd),
        "read": read,
        "write": [str(cwd), tmp],
        "devices": list(DEVICES),
        "protected": [str(p) for p in protected_dirs() + sensitive_dirs()],
        "network": bool(perms().get("shell_network", False)),
        # Off only by an operator's explicit choice in the file.
        "require": bool(perms().get("sandbox", True)),
        "env": clean_env(tmp),
    }
