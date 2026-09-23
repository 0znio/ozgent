"""Minimal async HTTP helper.

Uses ``httpx`` when it is installed and falls back to the standard library
otherwise, so a fresh ozgent install needs no pip step to search the web.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json as _json
import socket
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


# ------------------------------------------------------------------ SSRF guard
#
# A tool fetches whatever URL the model names, and the model reads web pages.
# A page can therefore send it anywhere the worker can reach: this server's own
# API on 127.0.0.1, a router's admin page, a printer, a cloud metadata address
# that hands out credentials. So every hop of every request must land on a
# public address, unless the operator has allowed private ones — checked when
# the name is resolved, again on every redirect, and once more against the
# address the connection actually reached, which closes the gap a DNS answer
# that changes between the check and the connect (rebinding) would leave.

#: Redirects followed before giving up.
MAX_REDIRECTS = 8

_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def is_public(ip: str) -> bool:
    """Whether `ip` is an address on the public internet.

    IPv4 carried inside IPv6 — mapped, 6to4, Teredo, NAT64 — is judged by the
    IPv4 address it carries, or ``::ffff:127.0.0.1`` would pass as global.
    """
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return is_public(str(addr.ipv4_mapped))
        if addr.sixtofour is not None:
            return is_public(str(addr.sixtofour))
        if addr.teredo is not None:
            return is_public(str(addr.teredo[1]))
        if addr in _NAT64:
            return is_public(str(ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)))
    return addr.is_global and not addr.is_multicast


def _private_allowed(host: str) -> bool:
    from .permissions import private_allowed  # imported late: permissions imports tool config

    try:
        return private_allowed(host)
    except Exception:  # noqa: BLE001 - outside a tool call there is no config: stay strict
        return False


def _refusal(host: str, ip: str) -> HttpError:
    return HttpError(
        0,
        f"{host} is at {ip}, which is this machine or a private network. Tools may not fetch "
        "from there unless the host is listed in [tools.config.permissions] network_allow, "
        "or network_private = true.",
    )


async def check_url(url: str) -> None:
    """Refuse `url` unless it is http(s) to a public address."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise HttpError(0, f"{parsed.scheme or 'that'!r} is not a URL scheme this can fetch")
    host = parsed.hostname or ""
    if not host:
        raise HttpError(0, f"{url!r} has no host")
    if _private_allowed(host):
        return
    try:
        ipaddress.ip_address(host)
        literal = [host]
    except ValueError:
        literal = []
    if literal:
        addresses = literal
    else:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise HttpError(0, f"cannot resolve {host}: {exc}") from exc
        addresses = sorted({info[4][0] for info in infos})
    for ip in addresses:
        if not is_public(ip):
            raise _refusal(host, ip)


def _check_url_blocking(url: str) -> None:
    asyncio.run(check_url(url))


def _check_peer(resp: Any, url: str) -> None:
    """Refuse a response from a non-public address, whatever DNS said first."""
    stream = resp.extensions.get("network_stream")
    peer = stream.get_extra_info("server_addr") if stream is not None else None
    host = urllib.parse.urlsplit(url).hostname or ""
    if peer and not is_public(str(peer[0])) and not _private_allowed(host):
        raise _refusal(host, str(peer[0]))


async def _send(client: Any, method: str, url: str, headers: dict[str, str], body: bytes | None) -> Any:
    """Send a request, following redirects by hand so each hop is checked.

    Returns an open streaming response; the caller reads and closes it.
    """
    for _ in range(MAX_REDIRECTS + 1):
        await check_url(url)
        request = client.build_request(method, url, headers=headers, content=body)
        resp = await client.send(request, stream=True)
        try:
            _check_peer(resp, url)
        except HttpError:
            await resp.aclose()
            raise
        if not resp.is_redirect:
            return resp
        location = resp.headers.get("location", "")
        await resp.aclose()
        if not location:
            raise HttpError(resp.status_code, "a redirect with no location")
        url = urllib.parse.urljoin(url, location)
        # 303, and 301/302 after a POST, become a GET, as browsers do.
        if resp.status_code == 303 or (resp.status_code in (301, 302) and method != "GET"):
            method, body = "GET", None
    raise HttpError(0, f"more than {MAX_REDIRECTS} redirects")


async def _read_capped(resp: Any, limit: int) -> bytes:
    """The body, stopping at `limit` rather than downloading what follows it."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in resp.aiter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= limit:
            break
    return b"".join(chunks)[:limit]


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """urllib's redirect handler, checking each hop the same way."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        try:
            _check_url_blocking(newurl)
        except HttpError as exc:
            raise urllib.error.URLError(str(exc)) from exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_GuardedRedirects)


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
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resp = await _send(client, method, url, hdrs, body)
            try:
                raw = await _read_capped(resp, MAX_BYTES)
            finally:
                await resp.aclose()
            text = raw.decode(resp.encoding or "utf-8", errors="replace")
            if resp.status_code >= 400:
                raise HttpError(resp.status_code, f"HTTP {resp.status_code}", text[:500])
            return text

    await check_url(url)
    return await asyncio.to_thread(_blocking_request, url, method, hdrs, body, timeout)


def _blocking_request(
    url: str, method: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> str:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read(MAX_BYTES).decode(charset, errors="replace")
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
            timeout=timeout, follow_redirects=False, verify=True
        ) as client:
            resp = await _send(client, "GET", url, hdrs, None)
            try:
                body = await _read_capped(resp, MAX_BYTES)
            finally:
                await resp.aclose()
            if resp.status_code >= 400:
                raise HttpError(
                    resp.status_code,
                    f"HTTP {resp.status_code} {resp.reason_phrase}".strip(),
                    body[:500].decode("utf-8", errors="replace"),
                )
            return Page(
                url=str(resp.url),
                status=resp.status_code,
                content_type=resp.headers.get("content-type", ""),
                body=body,
                encoding=resp.charset_encoding or "",
            )

    await check_url(url)
    return await asyncio.to_thread(_blocking_fetch, url, hdrs, timeout)


def _blocking_fetch(url: str, hdrs: dict[str, str], timeout: float) -> Page:
    # urllib does not decompress, so ask for what it can actually read.
    plain = dict(hdrs)
    plain["Accept-Encoding"] = "identity"
    req = urllib.request.Request(url, headers=plain, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
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
