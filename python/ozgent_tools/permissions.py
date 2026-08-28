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
