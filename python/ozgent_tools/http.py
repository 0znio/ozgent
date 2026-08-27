"""Minimal async HTTP helper.

Uses ``httpx`` when it is installed and falls back to the standard library
otherwise, so a fresh ozgent install needs no pip step to search the web.
"""

from __future__ import annotations

import asyncio
import json as _json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

#: Sent on API calls, where a bare identifier is correct and welcome.
USER_AGENT = "ozgent/0.1 (+https://github.com/ozgent)"

#: Sent when reading a page meant for a person.
#:
#: Not a disguise: a great many sites answer an unrecognised agent with 401 or
#: 403 regardless of robots.txt, and Reuters answers this one 200 where it
#: answers `ozgent/0.1` with 401. The request is the same either way -- one
#: page, on demand, because the user asked for it.
BROWSER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

#: Headers a browser sends that servers check for.
BROWSER_HEADERS = {
    "User-Agent": BROWSER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Deliberately not `br`. httpx advertises brotli whenever brotlicffi is
    # importable, and brotlicffi rejects some real responses outright --
    # "decoder process called with data when 'can_accept_more_data()' is
    # False" is what Times of India returned. gzip is universal and has no
    # such failure mode.
    "Accept-Encoding": "gzip, deflate",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

try:  # pragma: no cover - availability depends on the environment
    import httpx

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    _HAS_HTTPX = False


class HttpError(Exception):
    def __init__(self, status: int, message: str, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


async def request(
    url: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    data: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> str:
    """Perform a request and return the body as text."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    hdrs.update(headers or {})

    body: bytes | None = None
    if json_body is not None:
        body = _json.dumps(json_body).encode()
        hdrs["Content-Type"] = "application/json"
    elif data is not None:
        body = urllib.parse.urlencode(data).encode()
        hdrs["Content-Type"] = "application/x-www-form-urlencoded"

    if _HAS_HTTPX:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.request(method, url, headers=hdrs, content=body)
            if resp.status_code >= 400:
                raise HttpError(resp.status_code, f"HTTP {resp.status_code}", resp.text[:500])
            return resp.text

    return await asyncio.to_thread(_blocking_request, url, method, hdrs, body, timeout)


def _blocking_request(
    url: str, method: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> str:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise HttpError(exc.code, f"HTTP {exc.code}", detail) from exc
    except urllib.error.URLError as exc:
        raise HttpError(0, f"network error: {exc.reason}") from exc


async def get_json(url: str, **kw: Any) -> Any:
    return _json.loads(await request(url, **kw))


async def post_json(url: str, **kw: Any) -> Any:
    return _json.loads(await request(url, method="POST", **kw))


class Page:
    """A fetched resource: its bytes, what the server said it was, where it ended up."""

    __slots__ = ("url", "status", "content_type", "body", "encoding")

    def __init__(self, url: str, status: int, content_type: str, body: bytes, encoding: str):
        self.url = url
        self.status = status
        self.content_type = content_type
        self.body = body
        self.encoding = encoding

    @property
    def kind(self) -> str:
        """The content type without its parameters, lowercased."""
        return self.content_type.split(";")[0].strip().lower()

    def text(self) -> str:
        """Decode the body, preferring the charset the server declared."""
        for codec in (self.encoding, "utf-8", "cp1252", "latin-1"):
            if not codec:
                continue
            try:
                return self.body.decode(codec)
            except (UnicodeDecodeError, LookupError):
                continue
        return self.body.decode("utf-8", errors="replace")


#: Status codes worth trying again: rate limits and transient server faults.
RETRYABLE = {408, 425, 429, 500, 502, 503, 504}

#: Refuse a body larger than this rather than feed a model 80 MB of video.
MAX_BYTES = 8 * 1024 * 1024


async def fetch_page(
    url: str,
    *,
    timeout: float = 25.0,
    attempts: int = 3,
    headers: dict[str, str] | None = None,
) -> Page:
    """Fetch one URL as a browser would, retrying what is worth retrying.

    Raises `HttpError` with the final status when the server keeps refusing.
    """
    hdrs = dict(BROWSER_HEADERS)
    hdrs.update(headers or {})
    last: Exception | None = None

    for attempt in range(attempts):
        try:
            return await _fetch_once(url, hdrs, timeout)
        except HttpError as exc:
            last = exc
            if exc.status not in RETRYABLE:
                raise
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised after
            last = exc
        if attempt + 1 < attempts:
            # Brief and increasing. A rate limit answered instantly three times
            # is three refusals, not three chances.
            await asyncio.sleep(0.6 * (2**attempt))

    raise last if last else HttpError(0, "request failed")


async def _fetch_once(url: str, hdrs: dict[str, str], timeout: float) -> Page:
    if _HAS_HTTPX:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, verify=True
        ) as client:
            resp = await client.get(url, headers=hdrs)
            body = resp.content[:MAX_BYTES]
            if resp.status_code >= 400:
                raise HttpError(
                    resp.status_code,
                    f"HTTP {resp.status_code} {resp.reason_phrase}".strip(),
                    resp.text[:500],
                )
            return Page(
                url=str(resp.url),
                status=resp.status_code,
                content_type=resp.headers.get("content-type", ""),
                body=body,
                encoding=resp.charset_encoding or "",
            )

    return await asyncio.to_thread(_blocking_fetch, url, hdrs, timeout)


def _blocking_fetch(url: str, hdrs: dict[str, str], timeout: float) -> Page:
    # urllib does not decompress, so ask for what it can actually read.
    plain = dict(hdrs)
    plain["Accept-Encoding"] = "identity"
    req = urllib.request.Request(url, headers=plain, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Page(
                url=resp.geturl(),
                status=resp.status,
                content_type=resp.headers.get("Content-Type", ""),
                body=resp.read(MAX_BYTES),
                encoding=resp.headers.get_content_charset() or "",
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise HttpError(exc.code, f"HTTP {exc.code}", detail) from exc
    except urllib.error.URLError as exc:
        raise HttpError(0, f"network error: {exc.reason}") from exc
