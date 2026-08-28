"""Running one allowed command and returning what it printed.

Off unless permitted, and then only programs named in the allowlist. Runs
without a shell, so pipes, redirection and chaining do not work — deliberately,
since they would let a permitted program start one that is not.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from ..base import ToolError, tool
from ..permissions import check_command, resolve_within

#: Seconds a command may run before it is killed.
COMMAND_TIMEOUT = 60.0

#: Characters of output handed back. A test suite can print megabytes, and the
#: tail is the part that says what happened.
MAX_OUTPUT = 8000


@tool(effect="execute")
async def run_command(
    command: Annotated[str, "One program and its arguments, e.g. 'cargo test'. No pipes or redirection."],
) -> dict[str, Any]:
    """Run one allowed command and return what it printed.

    Off unless permitted, and then only programs named in the allowlist.
    """
    argv = check_command(command)
    cwd = resolve_within(".", "run_command")
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ToolError(f"{argv[0]!r} is not installed or not on PATH") from exc
    except OSError as exc:
        raise ToolError(f"cannot run {argv[0]!r}: {exc}") from exc

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=COMMAND_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(
            f"{argv[0]!r} was still running after {COMMAND_TIMEOUT:.0f}s and was stopped"
        ) from None

    return {
        "command": " ".join(argv),
        "exit_code": proc.returncode,
        "stdout": _tail(out.decode("utf-8", "replace")),
        "stderr": _tail(err.decode("utf-8", "replace")),
        "cwd": str(cwd),
    }


def _tail(text: str) -> str:
    """Keep the end, not the beginning: a failing build prints its errors last."""
    if len(text) <= MAX_OUTPUT:
        return text
    return "… earlier output omitted …\n" + text[-MAX_OUTPUT:]
