"""Listing a directory, so a project can be explored rather than guessed at.

Confined to the same root as reading. Build and dependency directories are
named but not descended into: the answer would otherwise be thousands of files
that say nothing about the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from ..base import ToolError, tool
from ..permissions import resolve_within

#: Entries returned by one listing. Enough to understand a directory, far
#: short of what a node_modules would produce.
MAX_ENTRIES = 200

#: Never descended into. Naming them is useful; their contents are not.
NOISE = frozenset(
    """.git node_modules target build dist .venv venv __pycache__ .mypy_cache
    .pytest_cache .ruff_cache .cargo .next .idea .gradle""".split()
)


@tool(effect="read")
async def list_dir(
    path: Annotated[str, "Directory to list, absolute or relative to the project root."] = ".",
    depth: Annotated[int, "How many levels to descend. 1 lists just this directory."] = 1,
) -> dict[str, Any]:
    """List what is in a directory, so a project can be explored rather than guessed at.

    Confined to the permitted root. Build and dependency directories are named
    but not descended into.
    """
    root = resolve_within(path, "list_dir")
    if not root.exists():
        raise ToolError(f"{root} does not exist")
    if not root.is_dir():
        raise ToolError(f"{root} is a file, not a directory. Use read_file for it.")
    depth = max(1, min(int(depth), 4))

    entries: list[str] = []
    truncated = False

    def walk(directory: Path, level: int, prefix: str) -> None:
        nonlocal truncated
        if level > depth or truncated:
            return
        try:
            children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as exc:
            entries.append(f"{prefix}… cannot read this directory: {exc}")
            return
        for child in children:
            if len(entries) >= MAX_ENTRIES:
                truncated = True
                return
            if child.is_dir():
                skip = child.name in NOISE
                entries.append(f"{prefix}{child.name}/" + ("  (not descended)" if skip else ""))
                if not skip:
                    walk(child, level + 1, prefix + "  ")
            else:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                entries.append(f"{prefix}{child.name}  ({_size(size)})")

    walk(root, 1, "")
    return {
        "path": str(root),
        "entries": entries,
        "truncated": truncated,
        "note": f"stopped at {MAX_ENTRIES} entries" if truncated else None,
    }


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"
