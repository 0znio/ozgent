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

USER_AGENT = "ozgent/0.1 (+https://github.com/ozgent)"

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
