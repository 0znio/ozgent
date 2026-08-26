"""Reading a file, or the parts of it that matter.

A model asking to read a 6,000-line file cannot be given 6,000 lines, and
truncating at line 200 usually returns the imports. When a query is supplied
the file is split into definitions, scored, and — the part that matters —
expanded along its dependencies, so the helper a relevant function calls
arrives with it. See :mod:`ozgent_tools.relevance`.

The directory reads are confined to is set once, under
``[tools.config.permissions] root``. A ``[tools.config.read_file] root``
overrides it for this tool alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from ..base import ToolError, get_config, tool
from ..permissions import resolve_within
from ..relevance import expand_dependencies, score_chunks, split_definitions

#: Files above this are never read whole, even without a query.
DEFAULT_MAX_LINES = 400

#: Refuse to even open something this large; it is not source code.
MAX_BYTES = 8 * 1024 * 1024

SKIP_SUFFIXES = frozenset(
    """.png .jpg .jpeg .gif .webp .bmp .ico .pdf .zip .gz .tar .xz .bz2 .7z .rar
    .exe .dll .so .dylib .a .o .class .jar .pyc .pyo .wasm .bin .dat .db .sqlite
    .mp3 .mp4 .avi .mov .mkv .wav .flac .ttf .otf .woff .woff2 .gguf .safetensors""".split()
)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise ToolError(f"{path} does not exist")
    if path.is_dir():
        listing = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())[:200]
        raise ToolError(f"{path} is a directory. It contains: {', '.join(listing) or 'nothing'}")
    if path.suffix.lower() in SKIP_SUFFIXES:
        raise ToolError(f"{path.name} is a binary format ({path.suffix}); there is no text to read")

    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ToolError(f"{path.name} is {size // 1024 // 1024} MB, too large to read")

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ToolError(f"cannot read {path}: {exc}") from exc
    if b"\0" in data[:8192]:
        raise ToolError(f"{path.name} looks binary; there is no text to read")
    return data.decode("utf-8", errors="replace")


def _numbered(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{start + i:>{width}} | {line}" for i, line in enumerate(lines))


@tool
async def read_file(
    path: Annotated[str, "Path to the file, absolute or relative to the project root."],
    query: Annotated[
        str | None,
        "What you are looking for. Supply this for large files: the most relevant "
        "definitions are returned along with whatever they depend on, instead of "
        "the first N lines.",
    ] = None,
    start_line: Annotated[int | None, "Read from this line (1-based) instead of searching."] = None,
    end_line: Annotated[int | None, "Read up to this line, with start_line."] = None,
) -> dict[str, Any]:
    """Read a file, or the parts of it relevant to a query.

    Small files come back whole. For a large file, give `query` and you will get
    the definitions that match together with the ones they call, each labelled
    with its line numbers, and a list of what was left out.
    """
    settings = get_config("read_file")
    target = resolve_within(path, "read_file")
    text = _read_text(target)
    lines = text.splitlines()
    max_lines = int(settings.get("max_lines") or DEFAULT_MAX_LINES)

    # An explicit range wins: the caller has already decided what it wants.
    if start_line is not None:
        first = max(1, int(start_line))
        last = min(len(lines), int(end_line) if end_line else first + max_lines - 1)
        window = lines[first - 1 : last]
        return {
            "path": str(target),
            "total_lines": len(lines),
            "mode": "range",
            "content": _numbered(window, first),
            "omitted": len(lines) - len(window),
        }

    if len(lines) <= max_lines:
        return {
            "path": str(target),
            "total_lines": len(lines),
            "mode": "whole",
            "content": _numbered(lines, 1),
            "omitted": 0,
        }

    chunks = split_definitions(text)
    outline = [
        {"name": c.name, "start": c.start, "end": c.end}
        for c in chunks
        if c.name
    ]

    # No query on a large file: return the head plus an outline, so the model
    # can see what exists and ask again with something specific.
    if not query or not query.strip():
        head = lines[:max_lines]
        return {
            "path": str(target),
            "total_lines": len(lines),
            "mode": "head",
            "content": _numbered(head, 1),
            "omitted": len(lines) - len(head),
            "outline": outline,
            "note": (
                f"{len(lines)} lines is too many to return. This is the first "
                f"{len(head)}. Call again with `query` describing what you need, "
                "and the relevant definitions will be selected instead."
            ),
        }

    scores = score_chunks(chunks, query)
    if not scores:
        head = lines[:max_lines]
        return {
            "path": str(target),
            "total_lines": len(lines),
            "mode": "head",
            "content": _numbered(head, 1),
            "omitted": len(lines) - len(head),
            "outline": outline,
            "note": f"Nothing in {target.name} matched {query!r}; showing the start of the file.",
        }

    # Take matches by score until the budget is spent, then follow what they
    # depend on. Direct matches are chosen first so expansion never crowds out
    # the thing actually asked for.
    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
    selected: set[int] = set()
    used = 0
    for i in ranked:
        if used + chunks[i].lines > max_lines and selected:
            continue
        selected.add(i)
        used += chunks[i].lines
        if used >= max_lines:
            break

    matched = sorted(selected)
    selected, pulled_in = expand_dependencies(chunks, selected, max_lines, used)

    parts: list[str] = []
    shown = 0
    previous_end = 0
    for i in sorted(selected):
        chunk = chunks[i]
        gap = chunk.start - previous_end - 1
        if previous_end and gap > 0:
            parts.append(f"\n... {gap} lines omitted ...\n")
        parts.append(_numbered(chunk.text.splitlines(), chunk.start))
        shown += chunk.lines
        previous_end = chunk.end
    if previous_end < len(lines):
        parts.append(f"\n... {len(lines) - previous_end} lines omitted ...")

    return {
        "path": str(target),
        "total_lines": len(lines),
        "mode": "relevant",
        "content": "\n".join(parts),
        "omitted": len(lines) - shown,
        "matched": [chunks[i].name or f"lines {chunks[i].start}-{chunks[i].end}" for i in matched],
        "pulled_in": pulled_in,
        "outline": outline,
    }
