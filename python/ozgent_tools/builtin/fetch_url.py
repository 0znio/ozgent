"""Reading a URL as text, for following up on what a search returned.

Off unless network access is permitted, and restricted to the allowlist when
one is set.

What the naive version got wrong, in order of how often it bit:

- It advertised brotli and could not always decode it, so pages died in the
  transport with a message that named no cause.
- It sent `ozgent/0.1` as its user agent, which a good many sites answer with
  401 or 403.
- It stripped tags and returned the first 12,000 characters, which on any real
  page is the navigation menu. The fetch succeeded and the answer was useless,
  which is worse than failing.
- It assumed HTML. A JSON API, a PDF or an RSS feed came back as noise.

Each is handled below, and every refusal now says which one it was.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any

from ..base import ToolError, tool
from ..extract import extract, strip_tags
from ..http import HttpError, fetch_page
from ..permissions import check_host

#: Characters of a page handed back.
MAX_PAGE = 12000

#: Below this, a page that returned 200 has probably not returned its content.
THIN = 200


@tool
async def fetch_url(
    url: Annotated[str, "The http or https URL to read."],
    query: Annotated[
        str, "Optional: what you are looking for. Long pages are trimmed to the parts that match."
    ] = "",
) -> dict[str, Any]:
    """Read a web page, article, JSON API, or feed as text.

    Extracts the article and leaves out navigation, so what comes back is the
    content rather than the menus around it. Give `query` on a long page to get
    the passages that match it instead of the first few thousand characters.

    Off unless network access is permitted, and restricted to the allowlist
    when one is set.
    """
    host = check_host(url)

    try:
        page = await fetch_page(url)
    except HttpError as exc:
        raise ToolError(_explain(host, exc), retryable=exc.status in (429, 503)) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as advice
        raise ToolError(
            f"could not reach {host}: {type(exc).__name__}: {exc}", retryable=True
        ) from exc

    parsed = _read(page)
    text = parsed["text"]

    if not text.strip():
        raise ToolError(
            f"{host} returned {page.kind or 'no content type'} with nothing readable in it. "
            "The page is probably built by JavaScript, which this cannot run."
        )

    if query.strip() and len(text) > MAX_PAGE:
        text, focused = _focus(text, query), True
    else:
        focused = False

    result = {
        "url": page.url,
        "host": host,
        "title": parsed["title"],
        "kind": page.kind,
        "strategy": parsed["strategy"],
        "text": text[:MAX_PAGE],
        "truncated": len(text) > MAX_PAGE,
    }
    if focused:
        result["focused_on"] = query
    if parsed["strategy"] == "whole-page" and len(text) < THIN:
        # Say so rather than let a model treat a stub as the whole story.
        result["note"] = (
            "little text found; the page may require JavaScript or a subscription"
        )
    return result


def _read(page) -> dict[str, str]:
    """Turn a fetched resource into text according to what it actually is."""
    kind = page.kind

    if kind in ("application/json", "text/json") or kind.endswith("+json"):
        try:
            return {
                "title": "",
                "text": json.dumps(json.loads(page.text()), indent=2, ensure_ascii=False),
                "strategy": "json",
            }
        except ValueError:
            # Served as JSON but is not; fall through and treat it as text.
            return {"title": "", "text": page.text(), "strategy": "text"}

    if kind == "application/pdf" or page.body[:5] == b"%PDF-":
        return {"title": "", "text": _pdf_text(page.body), "strategy": "pdf"}

    if kind in ("text/plain", "text/markdown", "text/csv") or kind.startswith("text/x-"):
        return {"title": "", "text": page.text(), "strategy": "text"}

    if kind in ("application/xml", "text/xml") or kind.endswith("+xml"):
        # RSS and Atom are XML whose useful part is the entries; the tag
        # stripper gets those out without a feed parser.
        return {"title": "", "text": strip_tags(page.text()), "strategy": "xml"}

    body = page.text()
    if kind and not kind.startswith("text/") and "html" not in kind:
        raise ToolError(
            f"{page.url} is {kind}, which has no text to read."
        )

    parsed = extract(body)
    text = str(parsed["text"])
    # A description is a poor article but a good stub, and beats returning
    # nothing when the body did not survive extraction.
    if len(text) < THIN and parsed["description"]:
        text = f"{parsed['description']}\n\n{text}".strip()
    return {"title": str(parsed["title"]), "text": text, "strategy": str(parsed["strategy"])}


def _pdf_text(body: bytes) -> str:
    """Text from a PDF, if anything in the environment can read one.

    No dependency is required, so this is best-effort by design: a clear
    refusal is more useful than a page of binary.
    """
    for module, call in (
        ("pypdf", lambda m: m.PdfReader),
        ("PyPDF2", lambda m: m.PdfReader),
    ):
        try:
            mod = __import__(module)
        except ImportError:
            continue
        try:
            import io

            reader = call(mod)(io.BytesIO(body))
            return "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:  # noqa: BLE001 - a broken PDF is not a crash
            break
    raise ToolError(
        "this is a PDF and no PDF reader is installed. `pip install pypdf` to read them."
    )


_PARAGRAPH_BREAK = re.compile(r"\n{2,}")


def _blocks(text: str) -> list[str]:
    """Split into the largest units that still let a page be trimmed.

    Blank lines where the page has them, single lines where it does not: the
    extractor joins adjacent blocks with one newline, so splitting only on
    blank lines saw the whole article as a single paragraph and could never
    drop any of it.
    """
    paragraphs = [p for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
    if len(paragraphs) > 2:
        return paragraphs
    return [line for line in text.splitlines() if line.strip()]


def _focus(text: str, query: str) -> str:
    """Keep the paragraphs that match `query`, in their original order.

    Whole paragraphs rather than matching lines: a sentence lifted out of its
    surroundings is exactly the kind of context-free quote a model then
    misreads.
    """
    words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    if not words:
        return text

    paragraphs = _blocks(text)
    scored = []
    for i, para in enumerate(paragraphs):
        seen = {w for w in re.findall(r"\w+", para.lower())}
        hits = len(words & seen)
        if hits:
            scored.append((hits, -i, i, para))

    if not scored:
        return text

    scored.sort(reverse=True)
    keep, used = set(), 0
    for hits, _, i, para in scored:
        if used + len(para) > MAX_PAGE:
            break
        keep.add(i)
        used += len(para)

    if not keep:
        return text
    return "\n\n".join(paragraphs[i] for i in sorted(keep))


def _explain(host: str, exc: HttpError) -> str:
    """Say what the server did, and what would actually help."""
    status = exc.status
    if status in (401, 403):
        return (
            f"{host} refused the request (HTTP {status}). It blocks automated readers "
            "or requires a subscription; try another source for the same story."
        )
    if status == 404:
        return f"{host} has no page at that address (HTTP 404). Check the URL."
    if status == 429:
        return f"{host} is rate limiting (HTTP 429). Wait before trying it again."
    if status in (500, 502, 503, 504):
        return f"{host} is having trouble (HTTP {status}). Retried already; try later."
    if status == 0:
        return f"could not reach {host}: {exc}"
    return f"{host} returned HTTP {status}."
