"""Picking the parts of a large file that actually answer a question.

Reading a big file whole is usually impossible and always wasteful, but naive
truncation is worse: it cuts mid-definition and drops the one function that
mattered. Slicing by keyword alone is not much better, because the part that
answers a question frequently is not the part that mentions it — a caller
scores highly on the query while the behaviour lives in the callee, two hundred
lines away and sharing none of the query's words.

So this works in three passes:

1. **Split into definitions.** Chunks follow the file's own structure, so a
   returned function is always whole.
2. **Score against the query.** Plain lexical scoring, with rarer terms worth
   more — a chunk matching ``parse_manifest`` should beat one matching ``self``.
3. **Follow dependencies.** Each selected chunk's referenced symbols are
   resolved against the definitions in the file, and those chunks are pulled in
   too, transitively. This is the step that finds the callee nobody searched
   for, and is why a keyword slice is not enough.

Everything here is pure text processing with no parser per language: a real
parse would be better but would need a grammar for every language a user might
open, and the heuristics below degrade into "return the whole file" rather than
into something wrong.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# A definition's opening line, across the languages people actually paste.
# Ordered longest-prefix first so `async def` is not read as a bare statement.
DEFINITION_PATTERNS = [
    # Python, and anything with a `def`/`class` keyword.
    re.compile(r"^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>\w+)"),
    re.compile(r"^(?P<indent>\s*)class\s+(?P<name>\w+)"),
    # Rust.
    re.compile(r"^(?P<indent>\s*)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+|const\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*fn\s+(?P<name>\w+)"),
    re.compile(r"^(?P<indent>\s*)(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|impl|mod|union)\s+(?:<[^>]*>\s*)?(?P<name>\w+)"),
    # C, C++, Java, Go, TypeScript.
    re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(?P<name>\w+)"),
    re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:abstract\s+)?(?:class|interface|type|enum)\s+(?P<name>\w+)"),
    re.compile(r"^(?P<indent>\s*)func\s+(?:\([^)]*\)\s*)?(?P<name>\w+)"),
    # `const handler = (…) => {`, `let x = function`.
    re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|\w+\s*=>)"),
]

IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

# Words too common to carry meaning, in prose or in code.
STOPWORDS = frozenset("""
a an and are as at be but by can do does for from has have how i if in into is it
its of on or that the their then there these this to was what when where which
who why will with you your
self this true false none null void return if else while for in let const var
def class fn func function import from export public private static async await
new not and or is str int bool dict list set type object value key item
""".split())


@dataclass
class Chunk:
    """One definition, or a run of top-level code between definitions."""

    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    name: str | None
    text: str
    #: Identifiers this chunk mentions, used to follow dependencies.
    refs: set[str] = field(default_factory=set)

    @property
    def lines(self) -> int:
        return self.end - self.start + 1


#: A chunk larger than this is split into its nested definitions instead.
#: A 200-line `impl` block or a long class is not a useful unit of relevance —
#: what the reader wants is the method inside it.
MAX_CHUNK_LINES = 80


def split_definitions(source: str) -> list[Chunk]:
    """Split a file into definition-shaped chunks.

    A definition runs until a line that is neither blank nor indented deeper
    than its own opening line — which is exactly right for Python and a decent
    approximation everywhere else, because brace languages indent their bodies
    too. Text before the first definition becomes its own chunk so imports and
    module docstrings are never lost.

    Large definitions are split again into their own nested definitions, with
    the opening line kept as a separate chunk for context. Without that, an
    `impl` block or a long class returns as one indivisible slab and defeats
    the point of selecting anything.
    """
    lines = source.splitlines()
    if not lines:
        return []
    chunks = _segment(lines, 0, len(lines) - 1)
    return [c for c in chunks if c.text.strip()]


def _definitions_in(lines: list[str], lo: int, hi: int) -> list[tuple[int, str, int]]:
    """Outermost definitions within [lo, hi], as (index, name, indent)."""
    found: list[tuple[int, str, int]] = []
    for i in range(lo, hi + 1):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith(("#", "//", "*")):
            continue
        for pattern in DEFINITION_PATTERNS:
            m = pattern.match(line)
            if m:
                found.append((i, m.group("name"), len(m.group("indent").expandtabs(4))))
                break
    if not found:
        return []

    # Keep only those at the shallowest indent; anything deeper is a body.
    outermost = min(f[2] for f in found)
    return [f for f in found if f[2] == outermost]


def _segment(lines: list[str], lo: int, hi: int, depth: int = 0) -> list[Chunk]:
    """Chunk the region [lo, hi], recursing into oversized definitions."""
    if lo > hi:
        return []

    tops = _definitions_in(lines, lo, hi)
    if not tops:
        return [_chunk(lines, lo, hi, None)]

    chunks: list[Chunk] = []
    if tops[0][0] > lo:
        chunks.append(_chunk(lines, lo, tops[0][0] - 1, None))

    for idx, (line_no, name, indent) in enumerate(tops):
        end = _body_end(lines, line_no, indent)
        following = tops[idx + 1][0] - 1 if idx + 1 < len(tops) else hi
        end = min(end, following)

        size = end - line_no + 1
        inner = _definitions_in(lines, line_no + 1, end) if size > MAX_CHUNK_LINES else []
        if inner and depth < 3:
            # Keep the opening line(s) so the nested pieces stay attributable to
            # the class or impl they came from.
            header_end = inner[0][0] - 1
            if header_end >= line_no:
                chunks.append(_chunk(lines, line_no, header_end, name))
            chunks.extend(_segment(lines, inner[0][0], end, depth + 1))
        else:
            chunks.append(_chunk(lines, line_no, end, name))

        if end < following:
            # The gap can itself hold definitions — a `_body_end` that stopped
            # early leaves the rest of an impl block here, and emitting it raw
            # produced one indivisible 376-line chunk. Recursion terminates
            # because each level consumes at least one definition.
            if _definitions_in(lines, end + 1, following):
                chunks.extend(_segment(lines, end + 1, following, depth))
            else:
                chunks.append(_chunk(lines, end + 1, following, None))

    return chunks


def _body_end(lines: list[str], start: int, indent: int) -> int:
    """Last line of the definition opening at `start`."""
    end = start
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= indent and not line.lstrip().startswith(("}", ")", "]")):
            break
        end = i
    # Trailing closing brace of a brace language belongs to the definition.
    for i in range(end + 1, min(end + 3, len(lines))):
        if lines[i].strip() in {"}", "};", ")", ");", "]"}:
            end = i
        else:
            break
    return end


def _chunk(lines: list[str], start: int, end: int, name: str | None) -> Chunk:
    text = "\n".join(lines[start : end + 1])
    refs = {t for t in IDENTIFIER.findall(text) if t not in STOPWORDS}
    if name:
        refs.discard(name)
    return Chunk(start=start + 1, end=end + 1, name=name, text=text, refs=refs)


def tokenize(text: str) -> list[str]:
    """Query and chunk text reduced to comparable terms.

    ``parse_manifest`` also yields ``parse`` and ``manifest``, so a query
    written either way finds the definition.
    """
    out: list[str] = []
    for raw in IDENTIFIER.findall(text.lower()):
        if raw in STOPWORDS or len(raw) < 2:
            continue
        out.append(raw)
        parts = [p for p in re.split(r"_+", raw) if len(p) > 2]
        camel = [p.lower() for p in re.findall(r"[A-Z]?[a-z0-9]+", raw) if len(p) > 2]
        for piece in parts + camel:
            if piece != raw and piece not in STOPWORDS:
                out.append(piece)
    return out


def score_chunks(chunks: list[Chunk], query: str) -> dict[int, float]:
    """Score every chunk against the query. Higher is more relevant.

    Rare terms count for more, so a query naming a specific symbol is not
    drowned out by chunks that merely use common vocabulary. A name match is
    weighted heavily: a chunk *called* what you asked for is almost always what
    you meant, whatever its body says.
    """
    terms = tokenize(query)
    if not terms:
        return {}

    documents = [set(tokenize(c.text)) for c in chunks]
    total = len(chunks) or 1
    idf = {
        term: math.log(1 + total / (1 + sum(term in d for d in documents)))
        for term in set(terms)
    }

    scores: dict[int, float] = {}
    for i, chunk in enumerate(chunks):
        counts = documents[i]
        score = sum(idf[t] for t in set(terms) if t in counts)
        if chunk.name:
            name_terms = set(tokenize(chunk.name))
            score += 3.0 * sum(idf[t] for t in set(terms) if t in name_terms)
        # Normalise gently by size so a huge chunk does not win on volume.
        score /= 1 + math.log1p(chunk.lines) / 4
        if score > 0:
            scores[i] = score
    return scores


def expand_dependencies(
    chunks: list[Chunk],
    selected: set[int],
    budget_lines: int,
    used_lines: int,
    max_hops: int = 2,
) -> tuple[set[int], list[str]]:
    """Pull in the definitions the selected chunks depend on.

    This is the step that makes the result usable rather than merely relevant:
    a function that answers the question is not much help if the helper it
    delegates to is missing. Expansion is breadth-first so direct dependencies
    are taken before their dependencies, and it stops at the line budget rather
    than dragging in the whole file.

    Returns the enlarged selection and a note of what was added and why.
    """
    by_name: dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        if chunk.name and chunk.name not in by_name:
            by_name[chunk.name] = i

    notes: list[str] = []
    frontier = set(selected)
    for _ in range(max_hops):
        if used_lines >= budget_lines:
            break
        wanted: list[tuple[int, str, str]] = []  # (chunk index, symbol, needed by)
        for i in sorted(frontier):
            for ref in sorted(chunks[i].refs):
                target = by_name.get(ref)
                if target is None or target in selected:
                    continue
                wanted.append((target, ref, chunks[i].name or f"lines {chunks[i].start}-{chunks[i].end}"))

        if not wanted:
            break

        added: set[int] = set()
        for target, symbol, needed_by in wanted:
            if target in selected:
                continue
            if used_lines + chunks[target].lines > budget_lines:
                continue
            selected.add(target)
            added.add(target)
            used_lines += chunks[target].lines
            notes.append(f"{chunks[target].name or 'block'} (used by {needed_by})")
        if not added:
            break
        frontier = added

    return selected, notes
