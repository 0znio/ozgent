"""Reading a web page as text, for following up on what a search returned.

Off unless network access is permitted, and restricted to the allowlist when
one is set. The markup is stripped rather than parsed: the model needs the
words, and the tags between them are tokens it would pay to skim.
"""

from __future__ import annotations

import html
import re
from typing import Annotated, Any

from ..base import ToolError, tool
from ..http import request
from ..permissions import check_host

#: Characters of a fetched page handed back.
MAX_PAGE = 12000


@tool
async def fetch_url(
    url: Annotated[str, "The http or https URL to read."],
) -> dict[str, Any]:
    """Read a web page as text, for following up on something a search returned.

    Off unless network access is permitted, and restricted to the allowlist
    when one is set.
    """
    host = check_host(url)
    try:
        body = await request(url, headers={"Accept": "text/html,text/plain,*/*"})
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as advice
        raise ToolError(f"could not fetch {url}: {exc}", retryable=True) from exc

    text = _to_text(body)
    if not text.strip():
        raise ToolError(
            f"{host} returned nothing readable. The page may be built by JavaScript, "
            "which this cannot run."
        )
    return {
        "url": url,
        "host": host,
        "text": text[:MAX_PAGE],
        "truncated": len(text) > MAX_PAGE,
    }


_SCRIPTY = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_BREAKS = re.compile(r"<(br|/p|/div|/li|/h[1-6])\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n{3,}")


def _to_text(body: str) -> str:
    """Strip a page down to its prose.

    Not a parser and not trying to be one.
    """
    if "<" not in body:
        return body
    stripped = _SCRIPTY.sub(" ", body)
    stripped = _BREAKS.sub("\n", stripped)
    stripped = _TAG.sub(" ", stripped)
    stripped = html.unescape(stripped)
    lines = [" ".join(line.split()) for line in stripped.splitlines()]
    return _BLANKS.sub("\n\n", "\n".join(line for line in lines if line))
