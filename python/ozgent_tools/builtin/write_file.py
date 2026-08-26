"""Writing a file.

Off until permitted. Writing is the one built-in that changes something the
user did not ask for directly, so it is deny-by-default and confined to the
same root as reading::

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
from ..permissions import perms, resolve_within

@tool
async def write_file(
    path: Annotated[str, "Path to write, absolute or relative to the project root."],
    content: Annotated[str, "The full text to write."],
    mode: Annotated[str, "'create' fails if the file exists, 'overwrite' replaces it, 'append' adds to the end."] = "create",
) -> dict[str, Any]:
    """Write a file.

    Disabled unless turned on in config, and confined to the configured root.
    """
    # Either switch turns writing on: the shared `write` permission, or the
    # per-tool `enabled` this had before the shared one existed. Honouring both
    # means an upgrade does not silently revoke a permission already granted.
    settings = get_config("write_file")
    if not (perms().get("write", False) or settings.get("enabled", False)):
        raise ToolError(
            "writing is not permitted. Set [tools.config.permissions] write = true "
            "in ~/ozgent/configs/config.toml to allow it."
        )
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
