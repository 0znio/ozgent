"""Writing a file.

Off until permitted. Writing is the one built-in that changes something the
user did not ask for directly, so it is deny-by-default and confined to the
same root as reading. Either approve it when ozgent asks, or set it standing::

    [tools.config.permissions]
    root  = "/home/you/code"
    write = true

The write itself goes to a temporary file beside the target and is renamed
into place, so an interrupted write cannot leave a half-written file where a
good one was.
"""

from __future__ import annotations

from typing import Annotated, Any

from ..base import ToolError, get_config, tool
from ..permissions import require, resolve_within

@tool(effect="write")
async def write_file(
    path: Annotated[str, "Path to write, absolute or relative to the project root."],
    content: Annotated[str, "The full text to write."],
    mode: Annotated[str, "'create' fails if the file exists, 'overwrite' replaces it, 'append' adds to the end."] = "create",
) -> dict[str, Any]:
    """Write a file.

    Disabled unless turned on in config, and confined to the configured root.
    """
    # Through `require`, never by reading the flag here. `require` is also
    # where an approval the user gave at the prompt is honoured, and a tool
    # that checks the flag itself silently ignores it — which is exactly what
    # this one did: pressing "yes" wrote nothing, because the yes never
    # reached the only code that was asking.
    #
    # The per-tool `enabled` predates the shared permission and still counts,
    # so an upgrade does not silently revoke something already granted.
    if not get_config("write_file").get("enabled", False):
        require("write", "writing files")
    if mode not in {"create", "overwrite", "append"}:
        raise ToolError(f"unknown mode {mode!r}: use create, overwrite, or append")

    target = resolve_within(path, "write_file")
    if target.exists() and mode == "create":
        raise ToolError(f"{target} already exists; pass mode='overwrite' to replace it")
    if target.is_dir():
        raise ToolError(f"{target} is a directory")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "append":
            with target.open("a", encoding="utf-8") as fh:
                fh.write(content)
        else:
            # Written beside the target and renamed, so an interrupted write
            # cannot leave a half-written file in place of a good one.
            temporary = target.with_name(target.name + ".ozgent-tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(target)
    except OSError as exc:
        raise ToolError(f"cannot write {target}: {exc}") from exc

    written = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return {
        "path": str(target),
        "mode": mode,
        "lines_written": written,
        "bytes": len(content.encode("utf-8")),
    }
