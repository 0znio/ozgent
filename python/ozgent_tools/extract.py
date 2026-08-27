"""Pull the readable part out of a web page.

A page is mostly not its article. Wikipedia's markup opens with sixty lines of
navigation; a news site wraps four paragraphs in menus, share buttons, related
links and a newsletter box. Stripping tags and handing over the first 12,000
characters therefore gives a model a table of contents and no story -- which
looks like a working fetch and reads like a broken one.

Three strategies, best first:

1. **JSON-LD.** News sites publish ``articleBody`` in a ``<script
   type="application/ld+json">`` block because search engines read it. It is
   the article, already clean, with no markup to guess at.
2. **Scoring.** Otherwise, score each block by how much text it holds against
   how much of that text is inside links -- navigation is nearly all link, prose
   is nearly none. This is the readability heuristic, and it is the one that
   works on pages nobody has seen before.
3. **Whole page.** When neither finds anything, strip the tags and return
   everything, which is what the old version always did.

Standard library only, deliberately: the distributed bundle ships this source
with no site-packages, so a dependency here is a tool that works on the
development machine and fails on every installed one.
"""

from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser

#: Elements whose contents are never prose.
DROP = {
    "script", "style", "noscript", "svg", "canvas", "iframe", "form",
    "nav", "header", "footer", "aside", "template", "button", "select",
    "option", "figure", "figcaption", "video", "audio", "map", "object",
}

#: Elements that introduce a line break when flattened.
BLOCKS = {
    "p", "div", "section", "article", "main", "li", "tr", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "td", "dd", "dt",
}

#: Blocks worth scoring as a possible article body.
CANDIDATES = {"div", "section", "article", "main", "td", "blockquote", "pre"}

#: Class and id fragments that mark chrome, whatever the site.
CHROME = re.compile(
    r"nav|menu|sidebar|side-bar|footer|header|masthead|comment|share|social|"
    r"promo|advert|\bad\b|ads\b|banner|subscribe|newsletter|cookie|consent|"
    r"related|recommend|popup|modal|breadcrumb|pagination|paginate|toolbar|"
    r"widget|sponsor|trending|most-read|tags?\b|meta\b|skip",
    re.IGNORECASE,
)

#: ...and fragments that mark the thing we are looking for.
CONTENT = re.compile(
    r"article|articlebody|story|storybody|post-?body|post-?content|entry-?content|"
    r"main-?content|page-?content|content-?body|\bprose\b|markdown|"
    r"^content$|^body$|^text$|readable",
    re.IGNORECASE,
)


class _Node:
    """A block element, its text, and enough bookkeeping to score it."""

    __slots__ = ("tag", "ident", "parent", "children", "chunks", "link_chars", "dropped")

    def __init__(self, tag: str, ident: str, parent: "_Node | None"):
        self.tag = tag
        self.ident = ident
        self.parent = parent
        self.children: list[_Node] = []
        #: Text belonging directly to this node, not to a child block.
        self.chunks: list[str] = []
        self.link_chars = 0
        self.dropped = False

    def text(self) -> str:
        """This node and everything under it, flattened to lines."""
        parts: list[str] = []
        own = " ".join(" ".join(self.chunks).split())
        if own:
            parts.append(own)
        for child in self.children:
            if child.dropped:
                continue
            got = child.text()
            if got:
                parts.append(got)
        joined = "\n".join(parts)
        return joined

    def char_count(self) -> int:
        return sum(len(c) for c in self.chunks) + sum(
            c.char_count() for c in self.children if not c.dropped
        )

    def total_link_chars(self) -> int:
        return self.link_chars + sum(
            c.total_link_chars() for c in self.children if not c.dropped
        )


class _Reader(HTMLParser):
    """Build a tree of blocks, discarding everything that is not prose."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("[root]", "", None)
        self.node = self.root
        self.title = ""
        #: Nesting depth inside an element we are throwing away. Counted
        #: rather than flagged: <nav> containing <div> containing <nav> has to
        #: stay dropped until the outermost one closes.
        self.muted = 0
        self.in_title = False
        self.in_link = 0
        self.scripts: list[str] = []
        self.in_script: str | None = None
        self.meta: dict[str, str] = {}

    # ------------------------------------------------------------- parsing

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.lower(): (v or "") for k, v in attrs}

        if tag == "script" and "ld+json" in attr.get("type", ""):
            self.in_script = ""
            return
        if tag == "meta":
            key = attr.get("property") or attr.get("name") or ""
            if key and attr.get("content"):
                self.meta.setdefault(key.lower(), attr["content"])
            return
        if tag == "title":
            self.in_title = True
            return

        if self.muted:
            # Still track depth so the matching close tag is recognised.
            if tag in DROP:
                self.muted += 1
            return

        if tag in DROP:
            self.muted = 1
            return

        if tag == "a":
            self.in_link += 1
            return

        if tag in BLOCKS:
            ident = f"{attr.get('class', '')} {attr.get('id', '')}".strip()
            child = _Node(tag, ident, self.node)
            # Chrome is dropped by name where the markup says what it is. A
            # positive match wins a negative one: "article-related-nav" is
            # navigation, but "content" inside "ad-wrapper" is not rescued --
            # so the check is on this element's own attributes only.
            if ident and CHROME.search(ident) and not CONTENT.search(ident):
                child.dropped = True
            self.node.children.append(child)
            self.node = child

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
            return
        if self.in_script is not None and tag == "script":
            self.scripts.append(self.in_script)
            self.in_script = None
            return
        if self.muted:
            if tag in DROP:
                self.muted -= 1
            return
        if tag == "a":
            self.in_link = max(0, self.in_link - 1)
            return
        if tag in BLOCKS:
            # Walk up to the nearest matching open block rather than assuming
            # the tree is well formed: real pages close tags they never opened.
            node = self.node
            while node is not self.root and node.tag != tag:
                node = node.parent or self.root
            if node is not self.root and node.parent is not None:
                self.node = node.parent

    def handle_data(self, data: str) -> None:
        if self.in_script is not None:
            self.in_script += data
            return
        if self.in_title:
            self.title += data
            return
        if self.muted or not data.strip():
            return
        self.node.chunks.append(data)
        if self.in_link:
            self.node.link_chars += len(data.strip())


def _score(node: _Node) -> float:
    """How much this block looks like an article body.

    Length is the signal and link density is the correction: a sidebar of forty
    links has plenty of characters and almost all of them are anchor text.
    """
    chars = node.char_count()
    if chars < 200:
        return 0.0

    density = node.total_link_chars() / chars
    if density > 0.5:
        return 0.0

    score = chars * (1.0 - density)
    if node.tag in ("article", "main"):
        score *= 1.5
    elif node.tag in ("td", "blockquote"):
        score *= 0.6
    if node.ident and CONTENT.search(node.ident):
        score *= 1.6
    # Paragraphs are what prose is made of; a div of forty one-line divs is
    # usually a listing.
    paragraphs = sum(1 for c in _walk(node) if c.tag == "p" and c.char_count() > 80)
    score *= 1.0 + min(paragraphs, 10) * 0.08
    return score


def _walk(node: _Node):
    for child in node.children:
        if child.dropped:
            continue
        yield child
        yield from _walk(child)


def _from_json_ld(scripts: list[str]) -> str:
    """The article body a page publishes for search engines, if it does."""
    for raw in scripts:
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        for entry in _flatten_ld(data):
            if not isinstance(entry, dict):
                continue
            body = entry.get("articleBody") or entry.get("text")
            if isinstance(body, str) and len(body.strip()) > 200:
                return body.strip()
    return ""


def _flatten_ld(data):
    """JSON-LD arrives as an object, a list, or an @graph of either."""
    if isinstance(data, list):
        for item in data:
            yield from _flatten_ld(item)
    elif isinstance(data, dict):
        yield data
        graph = data.get("@graph")
        if graph is not None:
            yield from _flatten_ld(graph)


_BLANKS = re.compile(r"\n{3,}")


def _tidy(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    kept = [line for line in lines if line]
    return _BLANKS.sub("\n\n", "\n".join(kept))


def extract(body: str) -> dict[str, object]:
    """Reduce a page to its title, its prose, and how that was decided.

    ``strategy`` is reported so a thin result is diagnosable rather than
    merely disappointing.
    """
    try:
        reader = _Reader()
        reader.feed(body)
        reader.close()
    except Exception:  # noqa: BLE001 - malformed markup must not be fatal
        return {
            "title": "",
            "text": _tidy(strip_tags(body)),
            "strategy": "tags",
            "description": "",
        }

    title = " ".join(reader.title.split())
    description = reader.meta.get("og:description") or reader.meta.get("description") or ""

    article = _from_json_ld(reader.scripts)
    if article:
        return {
            "title": title or reader.meta.get("og:title", ""),
            "text": _tidy(article),
            "strategy": "json-ld",
            "description": description,
        }

    best, best_score = None, 0.0
    for node in _walk(reader.root):
        if node.tag not in CANDIDATES:
            continue
        got = _score(node)
        if got > best_score:
            best, best_score = node, got

    whole = _tidy(reader.root.text())
    if best is not None:
        chosen = _tidy(best.text())
        # A winner that holds almost everything on the page has not actually
        # discriminated; and one that holds almost nothing has over-trimmed.
        if len(chosen) >= 200 and len(chosen) >= len(whole) * 0.05:
            return {
                "title": title,
                "text": chosen,
                "strategy": "readability",
                "description": description,
            }

    return {"title": title, "text": whole, "strategy": "whole-page", "description": description}


_SCRIPTY = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_BREAKS = re.compile(r"<(br|/p|/div|/li|/h[1-6])\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def strip_tags(body: str) -> str:
    """Last resort: remove the markup and keep the words."""
    if "<" not in body:
        return body
    stripped = _SCRIPTY.sub(" ", body)
    stripped = _BREAKS.sub("\n", stripped)
    stripped = _TAG.sub(" ", stripped)
    return html.unescape(stripped)
