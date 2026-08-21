"""Reading and writing files.

``read_file`` is the interesting one. A model asking to read a 6,000-line file
cannot be given 6,000 lines, and truncating at line 200 usually returns the
imports. When a query is supplied the file is instead split into definitions,
scored, and — the part that matters — expanded along its dependencies, so the
helper a relevant function calls arrives with it. See
:mod:`ozgent_tools.relevance`.

Configure in ``~/ozgent/configs/config.toml``::

    [tools.config.read_file]
    root = "/home/you/code"     # refuse anything outside this
    max_lines = 400             # per read

    [tools.config.write_file]
    root = "/home/you/code"
    enabled = true              # writing is off unless turned on
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from ..base import ToolError, get_config, tool
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


def _root(settings: dict[str, Any]) -> Path:
    """The directory reads and writes are confined to.

    Defaults to the working directory ozgent was started in, which is the least
    surprising boundary: a model asked about "this project" should not be able
    to reach the rest of the disk.
    """
    return Path(settings.get("root") or os.getcwd()).expanduser().resolve()


def _resolve(path: str, settings: dict[str, Any]) -> Path:
    """Resolve `path` and refuse anything outside the configured root.

    Resolution happens before the check so `../` and symlinks cannot be used to
    step outside — comparing the strings first would be trivially defeated.
    """
    root = _root(settings)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ToolError(f"cannot resolve {path!r}: {exc}") from exc

    if resolved != root and root not in resolved.parents:
        raise ToolError(
            f"{path!r} is outside the permitted directory ({root}). "
            "Set [tools.config.read_file] root in config.toml to widen it."
        )
    return resolved


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
    target = _resolve(path, settings)
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


@tool
async def write_file(
    path: Annotated[str, "Path to write, absolute or relative to the project root."],
    content: Annotated[str, "The full text to write."],
    mode: Annotated[str, "'create' fails if the file exists, 'overwrite' replaces it, 'append' adds to the end."] = "create",
) -> dict[str, Any]:
    """Write a file.

    Disabled unless turned on in config, and confined to the configured root.
    """
    settings = get_config("write_file")
    if not settings.get("enabled", False):
        raise ToolError(
            "writing is disabled. Set [tools.config.write_file] enabled = true "
            "in ~/ozgent/configs/config.toml to allow it."
        )
    if mode not in {"create", "overwrite", "append"}:
        raise ToolError(f"unknown mode {mode!r}: use create, overwrite, or append")

    target = _resolve(path, settings)
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
