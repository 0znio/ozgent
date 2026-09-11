"""Reading Reddit: searching posts, listing a subreddit, reading a thread.

Two ways in, and the difference matters:

- **With API credentials** (a free "script" app from
  https://www.reddit.com/prefs/apps), requests go to Reddit's API as that app.
  This is the supported route: 100 requests a minute, and every post comes
  back with its score and comment count.
- **Without them**, Reddit's public Atom feeds. Reddit refuses anonymous JSON
  outright (HTTP 403), but still serves the feeds -- about one request a
  minute, measured, and no scores. Enough for one search; not enough to read
  a search *and* its threads. Every result says which route it came from, and
  a spent window is reported with how long until the next request, rather
  than retried into a longer one.

Configure in ``~/ozgent/configs/config.toml``::

    [tools.config.reddit]
    client_id = "..."          # or $REDDIT_CLIENT_ID
    client_secret = "..."      # or $REDDIT_CLIENT_SECRET
    username = "your_reddit_name"   # names you in the user agent, as Reddit asks

Reddit's API rules require a user agent of the form
``<platform>:<app id>:<version> (by /u/<username>)``, so that is what is sent.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Annotated, Any, Literal

from ..base import ToolError, get_config, tool
from ..extract import strip_tags

Action = Literal["search", "subreddit", "comments"]
Sort = Literal["relevance", "new", "top", "hot", "comments"]
Period = Literal["hour", "day", "week", "month", "year", "all"]

ATOM = {"a": "http://www.w3.org/2005/Atom"}

#: Characters of a post or comment body handed to the model. Long enough for
#: an argument, short enough that twenty of them fit in a prompt.
MAX_BODY = 700

#: Longest wait for a rate limit to lift before giving up on the call. The
#: tool timeout is thirty seconds, and the request itself needs some of it.
MAX_WAIT = 20.0

#: What Reddit last said about the rate-limit window: requests left in it,
#: and when it resets. Anonymous access gets about one feed request per
#: window, so knowing the window before asking is the difference between a
#: short wait and a wasted request that pushes the window back.
_WINDOW = {"remaining": None, "reset_at": 0.0}
_WINDOW_LOCK = threading.Lock()


def _settings() -> dict[str, Any]:
    return get_config("reddit")


def _user_agent() -> str:
    settings = _settings()
    if settings.get("user_agent"):
        return str(settings["user_agent"])
    who = settings.get("username") or os.environ.get("REDDIT_USERNAME") or "ozgent"
    return f"python:ozgent:0.1 (by /u/{who})"


def _credentials() -> tuple[str, str] | None:
    settings = _settings()
    cid = os.environ.get("REDDIT_CLIENT_ID") or settings.get("client_id")
    secret = os.environ.get("REDDIT_CLIENT_SECRET") or settings.get("client_secret")
    return (str(cid), str(secret)) if cid and secret else None


class _Token:
    """An app-only OAuth token, fetched on first use and renewed before expiry."""

    def __init__(self) -> None:
        self.value: str | None = None
        self.expires = 0.0
        self.lock = threading.Lock()

    def get(self, cid: str, secret: str) -> str:
        with self.lock:
            if self.value and time.time() < self.expires - 60:
                return self.value
            auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
            req = urllib.request.Request(
                "https://www.reddit.com/api/v1/access_token",
                data=b"grant_type=client_credentials",
                headers={"Authorization": f"Basic {auth}", "User-Agent": _user_agent()},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                raise ToolError(
                    f"Reddit rejected the API credentials (HTTP {exc.code}). Check "
                    "[tools.config.reddit] client_id and client_secret; the app must be a "
                    "'script' app."
                ) from exc
            if "access_token" not in data:
                raise ToolError(f"Reddit issued no token: {data.get('error', data)}")
            self.value = data["access_token"]
            self.expires = time.time() + float(data.get("expires_in", 3600))
            return self.value


_TOKEN = _Token()


def _note_window(headers: Any) -> None:
    """Remember the rate-limit window Reddit reported on a response."""
    try:
        remaining = float(headers.get("x-ratelimit-remaining"))
        reset = float(headers.get("x-ratelimit-reset"))
    except (TypeError, ValueError):
        return
    with _WINDOW_LOCK:
        _WINDOW["remaining"] = remaining
        _WINDOW["reset_at"] = time.monotonic() + reset


def _wait_for_window() -> None:
    """Sleep until the window resets if it is known to be spent, or refuse."""
    with _WINDOW_LOCK:
        spent = _WINDOW["remaining"] is not None and _WINDOW["remaining"] < 1
        wait = _WINDOW["reset_at"] - time.monotonic()
    if not spent or wait <= 0:
        return
    if wait > MAX_WAIT:
        # Not retryable: the model would be told "this may succeed if
        # retried" beside "the next request is possible in 40 seconds", and
        # it believes the first. It retried straight into the same wall.
        raise ToolError(_rate_limited(round(wait)))
    time.sleep(wait + 0.5)


def _get(url: str, headers: dict[str, str]) -> bytes:
    """GET, pacing to Reddit's rate-limit window, blocking."""
    headers = {"User-Agent": _user_agent(), **headers}
    for attempt in range(2):
        _wait_for_window()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as resp:
                _note_window(resp.headers)
                return resp.read()
        except urllib.error.HTTPError as exc:
            _note_window(exc.headers)
            if exc.code == 429 and attempt == 0:
                # The window is now known, so the next pass waits for it
                # rather than asking again straight away.
                continue
            if exc.code == 429:
                raise ToolError(_rate_limited()) from exc
            if exc.code == 404:
                raise ToolError("Reddit has no such subreddit or post") from exc
            if exc.code == 403:
                raise ToolError(
                    "Reddit refused the request (HTTP 403): the subreddit may be private or "
                    "banned, or Reddit is blocking anonymous access. " + _setup_hint()
                ) from exc
            raise ToolError(f"Reddit answered HTTP {exc.code}", retryable=exc.code >= 500) from exc
        except urllib.error.URLError as exc:
            raise ToolError(f"could not reach Reddit: {exc.reason}", retryable=True) from exc
    raise ToolError(_rate_limited())


def _rate_limited(seconds: int | None = None) -> str:
    when = f"in about {seconds} seconds" if seconds else "in a minute"
    if _credentials():
        return f"Reddit is rate-limiting this app; try again {when}."
    return (
        f"Reddit allows very few anonymous requests; the next is possible {when}. "
        "Carry on with what you have. " + _setup_hint()
    )


def _setup_hint() -> str:
    return (
        "For reliable access, create a free 'script' app at https://www.reddit.com/prefs/apps "
        "and set [tools.config.reddit] client_id and client_secret in config.toml."
    )


def _clip(text: str, limit: int = MAX_BODY) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def post_id(reference: str) -> str:
    """The base-36 id of a post, from a URL, a ``t3_`` fullname or a bare id."""
    reference = reference.strip()
    found = re.search(r"/comments/([a-z0-9]+)", reference, re.I)
    if found:
        return found.group(1).lower()
    bare = reference.removeprefix("t3_")
    if re.fullmatch(r"[a-z0-9]{4,12}", bare, re.I):
        return bare.lower()
    raise ToolError(
        f"{reference!r} is not a Reddit post. Pass the post's URL, or its id from a search result."
    )


def _subreddit(name: str) -> str:
    name = name.strip().removeprefix("/").removeprefix("r/").strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_]{2,21}", name):
        raise ToolError(f"{name!r} is not a subreddit name")
    return name


def _iso(seconds: Any) -> str | None:
    if not isinstance(seconds, (int, float)):
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


# ------------------------------------------------------------------ API route


def shape_api_post(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": d.get("id"),
        "title": d.get("title", ""),
        "subreddit": d.get("subreddit"),
        "author": d.get("author"),
        "score": d.get("score"),
        "upvote_ratio": d.get("upvote_ratio"),
        "comments": d.get("num_comments"),
        "created": _iso(d.get("created_utc")),
        "url": "https://www.reddit.com" + d.get("permalink", ""),
        "snippet": _clip(d.get("selftext") or ""),
        "link": None if d.get("is_self") else d.get("url"),
    }


def shape_api_comments(payload: list[Any], count: int) -> dict[str, Any]:
    post = ((payload[0].get("data") or {}).get("children") or [{}])[0].get("data") or {}
    comments = []
    for child in (payload[1].get("data") or {}).get("children") or []:
        if child.get("kind") != "t1":
            continue  # "more" placeholders, not comments
        c = child.get("data") or {}
        if c.get("body") in ("[deleted]", "[removed]"):
            continue
        comments.append({
            "author": c.get("author"),
            "score": c.get("score"),
            "created": _iso(c.get("created_utc")),
            "text": _clip(c.get("body") or ""),
            "url": "https://www.reddit.com" + c.get("permalink", ""),
        })
        if len(comments) == count:
            break
    return {"post": shape_api_post(post), "comments": comments}


def _api(path: str, params: dict[str, Any], creds: tuple[str, str]) -> Any:
    token = _TOKEN.get(*creds)
    url = f"https://oauth.reddit.com{path}?{urllib.parse.urlencode({**params, 'raw_json': 1})}"
    return json.loads(_get(url, {"Authorization": f"Bearer {token}"}))


# ----------------------------------------------------------------- feed route


def parse_feed(body: bytes) -> list[dict[str, Any]]:
    """Entries of a Reddit Atom feed, as posts or comments."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ToolError(
            "Reddit returned a page instead of a feed, which usually means it is blocking "
            "anonymous access. " + _setup_hint()
        ) from exc
    entries = []
    for e in root.findall("a:entry", ATOM):
        link = e.find("a:link", ATOM)
        category = e.find("a:category", ATOM)
        author = e.find("a:author/a:name", ATOM)
        content = e.find("a:content", ATOM)
        entries.append({
            "id": (e.findtext("a:id", "", ATOM) or "").removeprefix("t3_").removeprefix("t1_"),
            "kind": "comment" if (e.findtext("a:id", "", ATOM) or "").startswith("t1_") else "post",
            "title": e.findtext("a:title", "", ATOM),
            "subreddit": category.get("term") if category is not None else None,
            "author": (author.text or "").removeprefix("/u/") if author is not None else None,
            "created": e.findtext("a:published", None, ATOM) or e.findtext("a:updated", None, ATOM),
            "url": link.get("href") if link is not None else None,
            "text": _clip(strip_tags(content.text or "")) if content is not None else "",
        })
    return entries


def _feed(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    return parse_feed(_get(f"https://www.reddit.com{path}{query}", {"Accept": "application/atom+xml"}))


def feed_posts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": e["id"], "title": e["title"], "subreddit": e["subreddit"],
            "author": e["author"], "created": e["created"], "url": e["url"],
            # Feeds prepend "submitted by /u/x [link] [comments]" to every
            # body; it is chrome, not what the person wrote.
            "snippet": re.sub(r"\s*submitted by\s+/u/\S+.*$", "", e["text"]).strip(),
        }
        for e in entries
        if e["kind"] == "post"
    ]


# ---------------------------------------------------------------------- tool


@tool(effect="read")
async def reddit(
    action: Annotated[
        Action,
        "search: find posts matching `query` (optionally within `subreddit`). "
        "subreddit: a subreddit's current posts. comments: the discussion under one post.",
    ],
    query: Annotated[str, "What to search for; for action 'search'."] = "",
    subreddit: Annotated[str, "Subreddit name without r/, e.g. stocks. Optional for 'search'."] = "",
    post: Annotated[str, "The post's URL or id; for action 'comments'."] = "",
    sort: Annotated[Sort, "Order of results. 'top' with `period` finds the most upvoted."] = "relevance",
    period: Annotated[Period, "Time window for 'search' and 'top'."] = "week",
    count: Annotated[int, "How many posts or comments, 1-25."] = 10,
) -> dict[str, Any]:
    """Read Reddit: search posts, list a subreddit, or read a post's comments. For opinions and discussion."""
    count = max(1, min(int(count), 25))
    creds = _credentials()
    route = "api" if creds else "feed"

    if action == "search":
        if not query.strip():
            raise ToolError("action 'search' needs a query")
        sub = _subreddit(subreddit) if subreddit.strip() else None
        path = f"/r/{sub}/search" if sub else "/search"
        params: dict[str, Any] = {"q": query, "sort": sort if sort != "hot" else "relevance",
                                  "t": period, "limit": count}
        if sub:
            params["restrict_sr"] = 1
        if creds:
            data = await asyncio.to_thread(_api, path, params, creds)
            posts = [shape_api_post(c["data"]) for c in (data.get("data") or {}).get("children") or []]
        else:
            posts = feed_posts(await asyncio.to_thread(_feed, path + ".rss", params))
        return _result(action, route, {"query": query, "subreddit": sub, "results": posts[:count]})

    if action == "subreddit":
        sub = _subreddit(subreddit or query)
        listing = sort if sort in ("new", "top", "hot") else "hot"
        params = {"limit": count, **({"t": period} if listing == "top" else {})}
        if creds:
            data = await asyncio.to_thread(_api, f"/r/{sub}/{listing}", params, creds)
            posts = [shape_api_post(c["data"]) for c in (data.get("data") or {}).get("children") or []]
        else:
            posts = feed_posts(await asyncio.to_thread(_feed, f"/r/{sub}/{listing}/.rss", params))
        return _result(action, route, {"subreddit": sub, "sort": listing, "results": posts[:count]})

    if action == "comments":
        pid = post_id(post or query)
        if creds:
            data = await asyncio.to_thread(
                _api, f"/comments/{pid}", {"limit": count, "depth": 1, "sort": "top"}, creds
            )
            return _result(action, route, shape_api_comments(data, count))
        entries = await asyncio.to_thread(_feed, f"/comments/{pid}/.rss", {"limit": count + 1})
        posts = [e for e in entries if e["kind"] == "post"]
        comments = [
            {"author": e["author"], "created": e["created"], "text": e["text"], "url": e["url"]}
            for e in entries
            if e["kind"] == "comment" and e["text"] not in ("[deleted]", "[removed]")
        ][:count]
        head = feed_posts(posts)[0] if posts else {"id": pid}
        return _result(action, route, {"post": head, "comments": comments})

    raise ToolError(f"unknown action {action!r}")


def _result(action: str, route: str, body: dict[str, Any]) -> dict[str, Any]:
    out = {"action": action, "source": "reddit api" if route == "api" else "reddit feeds", **body}
    if route == "feed":
        out["note"] = "Read from public feeds: no scores or comment counts. " + _setup_hint()
    return out
