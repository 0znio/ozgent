"""Running one allowed command and returning what it printed.

Off unless permitted, and then only programs named in the allowlist. Runs
without a shell, so pipes, redirection and chaining do not work — deliberately,
since they would let a permitted program start one that is not.

And inside a sandbox (``ozgent_tools/sandbox.py``), because the allowlist is
about which program starts, not what it then does: an allowed ``git`` can be
told to run any command through its own configuration, an allowed ``cargo``
builds and runs code from the project, an allowed ``python`` is anything. So
the program may read the system and its toolchain, write only the tool folder
and a temporary directory of its own, reach no network unless
``shell_network`` is on, see none of ozgent's files or the user's credentials,
and hold none of the daemon's secrets in its environment. When the time runs
out, everything it started ends with it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

from ..base import ToolError, tool
from ..permissions import check_command, clean_env, command_sandbox, resolve_within

COMMAND_TIMEOUT = 60.0

MAX_OUTPUT = 8000

#: The launcher, run by the same interpreter as the worker.
SANDBOX = Path(__file__).resolve().parent.parent / "sandbox.py"


@tool(effect="execute")
async def run_command(
    command: Annotated[str, "One program and its arguments, e.g. 'cargo test'. No pipes or redirection."],
) -> dict[str, Any]:
    """Run one allowed command and return what it printed.

    Off unless permitted, and then only programs named in the allowlist.
    """
    argv = check_command(command)
    cwd = resolve_within(".", "run_command")
    tmp = tempfile.mkdtemp(prefix="ozgent-cmd-")
    policy = command_sandbox(argv[0], cwd, tmp)
    sandboxed = policy["require"]
    try:
        if sandboxed:
            launch = [sys.executable, "-S", "-E", str(SANDBOX), json.dumps(policy), "--", *argv]
            env = clean_env(tmp)
        else:
            # The operator switched the sandbox off in the file. Still no
            # secrets in the environment.
            launch = argv
            env = clean_env(tmp)
        try:
            proc = await asyncio.create_subprocess_exec(
                *launch,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Its own process group, so a timeout reaches everything it
                # started and not the worker.
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"{argv[0]!r} is not installed or not on PATH") from exc
        except OSError as exc:
            raise ToolError(f"cannot run {argv[0]!r}: {exc}") from exc

        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=COMMAND_TIMEOUT)
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            raise ToolError(
                f"{argv[0]!r} was still running after {COMMAND_TIMEOUT:.0f}s and was stopped"
            ) from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    stderr = err.decode("utf-8", "replace")
    if proc.returncode == 126 and "ozgent sandbox: this kernel has no Landlock" in stderr:
        raise ToolError(
            "this machine cannot sandbox commands (its kernel has no Landlock), so none are run. "
            "An operator who accepts the risk can set [tools.config.permissions] sandbox = false."
        )
    return {
        "command": " ".join(argv),
        "exit_code": proc.returncode,
        "stdout": _tail(out.decode("utf-8", "replace")),
        "stderr": _tail(stderr),
        "cwd": str(cwd),
        "sandboxed": sandboxed,
    }


def _tail(text: str) -> str:
    """Keep the end, not the beginning: a failing build prints its errors last."""
    if len(text) <= MAX_OUTPUT:
        return text
    return "… earlier output omitted …\n" + text[-MAX_OUTPUT:]
