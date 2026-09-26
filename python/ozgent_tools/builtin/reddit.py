"""Reading Reddit: searching posts, listing a subreddit, reading a thread.

Three ways in, tried in this order:

- **With API credentials** (a free "script" app from
  https://www.reddit.com/prefs/apps), requests go to Reddit's API as that app:
  100 requests a minute, the supported route.
- **Without them, Reddit's own page fragments** — the server-rendered pieces
  reddit.com loads into its pages (``/svc/shreddit/...``). Anonymous, about
  200 requests per rate-limit window (measured 2026-09-26), with scores,
  comment counts and whole comment threads. Reddit refuses anonymous JSON
  (HTTP 403) and old.reddit.com now asks for a login, so these are what a
  browser without an account is served. A post's own text comes from Arctic
  Shift (arctic-shift.photon-reddit.com), a public archive of Reddit.
- **Reddit's Atom feeds**, the last resort should the fragments change shape:
  about one request a minute, and no scores.

Every result says which route it came from.

Configure in ``~/ozgent/configs/config.toml`` (optional)::

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
import html as htmllib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
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



# ------------------------------------------------------------------ web route

WEB = "https://www.reddit.com"
ARCTIC = "https://arctic-shift.photon-reddit.com/api"

#: Elements with no closing tag, which a depth count must not wait for.
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class _Fragments(HTMLParser):
    """Posts and comments out of a page fragment.

    A post is a ``<shreddit-post>`` whose attributes carry everything but the
    body, which is the text of its ``slot="text-body"`` child; a comment is a
    ``<shreddit-comment>`` with its text in a ``slot="comment"`` child.
    Parsed rather than matched, because comments nest inside comments.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.posts: list[dict[str, Any]] = []
        self.comments: list[dict[str, Any]] = []
        self._text_into: dict[str, Any] | None = None
        self._text_depth = 0
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag not in _VOID:
            self._depth += 1
        if tag == "shreddit-post":
            self.posts.append({**a, "_text": []})
        elif tag == "shreddit-comment":
            self.comments.append({**a, "_text": []})
        elif self._text_into is None and a.get("slot") in ("text-body", "comment"):
            owner = self.comments if a["slot"] == "comment" else self.posts
            if owner:
                self._text_into = owner[-1]
                self._text_depth = self._depth
        if tag in ("p", "br", "li") and self._text_into is not None:
            self._text_into["_text"].append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if self._text_into is not None and self._depth == self._text_depth:
            self._text_into = None
        self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._text_into is not None:
            self._text_into["_text"].append(data)


def _number(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _stamp(value: str | None) -> str | None:
    """``2026-09-26T00:40:52.000000+0000`` as ``2026-09-26T00:40:52Z``."""
    if not value:
        return None
    found = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", value)
    return f"{found.group(1)}Z" if found else value


def parse_listing(page: str) -> list[dict[str, Any]]:
    """The posts of a subreddit listing fragment."""
    parser = _Fragments()
    parser.feed(page)
    posts = []
    for p in parser.posts:
        if not p.get("id", "").startswith("t3_"):
            continue
        link = p.get("content-href", "")
        posts.append({
            "id": p["id"].removeprefix("t3_"),
            "title": htmllib.unescape(p.get("post-title", "")),
            "subreddit": p.get("subreddit-name"),
            "author": p.get("author"),
            "score": _number(p.get("score")),
            "upvote_ratio": round(float(p["upvote-ratio"]), 2) if p.get("upvote-ratio") else None,
            "comments": _number(p.get("comment-count")),
            "created": _stamp(p.get("created-timestamp")),
            "url": WEB + p.get("permalink", ""),
            "snippet": _clip("".join(p["_text"])),
            "link": None if "/comments/" in link or not link else link,
        })
    return posts


def parse_comments(page: str) -> list[dict[str, Any]]:
    """The comments of a thread fragment, in reading order, replies indented
    by ``depth``."""
    parser = _Fragments()
    parser.feed(page)
    comments = []
    for c in parser.comments:
        text = _clip("".join(c["_text"]))
        if not text or text in ("[deleted]", "[removed]"):
            continue
        comments.append({
            "author": c.get("author"),
            "score": _number(c.get("score")),
            "depth": _number(c.get("depth")) or 0,
            "created": _stamp(c.get("created")),
            "text": text,
            "url": WEB + c.get("permalink", "") if c.get("permalink") else None,
        })
    return comments


def parse_search(page: str) -> tuple[list[dict[str, Any]], str | None]:
    """The posts of a search fragment, and the fragment of the next page.

    Each result is announced by a tracking context naming the post — its id,
    title, subreddit, author and a snippet — and followed by its vote and
    comment counts and its age.
    """
    marks = [
        (m.start(), m.group(1))
        for m in re.finditer(r'data-faceplate-tracking-context="([^"]*)"', page)
    ]
    posts: list[dict[str, Any]] = []
    starts: list[int] = []
    for at, raw in marks:
        try:
            ctx = json.loads(htmllib.unescape(raw))
        except ValueError:
            continue
        post = ctx.get("post") or {}
        if (ctx.get("action_info") or {}).get("type") != "post" or not post.get("id"):
            continue
        if any(p["id"] == post["id"].removeprefix("t3_") for p in posts):
            continue
        starts.append(at)
        posts.append({
            "id": post["id"].removeprefix("t3_"),
            "title": post.get("title", ""),
            "subreddit": (ctx.get("subreddit") or {}).get("name"),
            "author": (ctx.get("profile") or {}).get("name"),
            "snippet": _clip((ctx.get("search") or {}).get("snippet") or ""),
        })
    for i, post in enumerate(posts):
        segment = page[starts[i] : starts[i + 1] if i + 1 < len(starts) else len(page)]
        row = segment.find('data-testid="search-counter-row"')
        numbers = re.findall(r'<faceplate-number[^>]*number="([^"]*)"', segment[row : row + 2000]) if row >= 0 else []
        post["score"] = _number(numbers[0]) if numbers else None
        post["comments"] = _number(numbers[1]) if len(numbers) > 1 else None
        ts = re.search(r'<faceplate-timeago[^>]*ts="([^"]*)"', segment)
        post["created"] = _stamp(ts.group(1)) if ts else None
        post["url"] = f"{WEB}/r/{post['subreddit']}/comments/{post['id']}/" if post["subreddit"] else f"{WEB}/comments/{post['id']}/"
    more = re.search(r'<faceplate-partial[^>]*src="(/svc/shreddit/search/[^"]*cursor=[^"]*)"', page)
    return posts, htmllib.unescape(more.group(1)) if more else None


def _web(path_and_query: str) -> str:
    return _get(WEB + path_and_query, {"Accept": "text/html", "Accept-Language": "en-US,en;q=0.9"}).decode("utf-8", "replace")


def web_listing(sub: str, listing: str, period: str, count: int) -> list[dict[str, Any]]:
    params = {"name": sub, **({"t": period} if listing == "top" else {})}
    return parse_listing(_web(f"/svc/shreddit/community-more-posts/{listing}/?{urllib.parse.urlencode(params)}"))[:count]


def web_search(query: str, sub: str | None, sort: str, period: str, count: int) -> list[dict[str, Any]]:
    params = {"q": query, "type": "posts", "sort": sort, "t": period}
    path = f"/svc/shreddit/r/{sub}/search/" if sub else "/svc/shreddit/search/"
    page = _web(f"{path}?{urllib.parse.urlencode(params)}")
    posts, more = parse_search(page)
    # About seven to a page; a few more pages for a larger count.
    for _ in range(3):
        if len(posts) >= count or not more:
            break
        extra, more = parse_search(_web(more))
        posts += [p for p in extra if all(p["id"] != q["id"] for q in posts)]
    return posts[:count]


def arctic_post(pid: str) -> dict[str, Any] | None:
    """A post's title and text from the Arctic Shift archive. Its score and
    comment count are as they stood when archived, minutes after posting, so
    they are left out."""
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{ARCTIC}/posts/ids?ids={pid}", headers={"User-Agent": _user_agent()}),
            timeout=15,
        ) as resp:
            data = (json.loads(resp.read()).get("data") or [None])[0]
    except (urllib.error.URLError, ValueError, OSError):
        return None
    if not data:
        return None
    return {
        "id": pid,
        "title": data.get("title", ""),
        "subreddit": data.get("subreddit"),
        "author": data.get("author"),
        "created": _iso(data.get("created_utc")),
        "url": WEB + (data.get("permalink") or f"/comments/{pid}/"),
        "text": _clip(data.get("selftext") or "", MAX_BODY * 3),
        "link": None if data.get("is_self") else data.get("url"),
    }


def arctic_search(query: str, sub: str | None, count: int) -> list[dict[str, Any]]:
    params = {"query": query, "limit": count, "sort": "desc", **({"subreddit": sub} if sub else {})}
    with urllib.request.urlopen(
        urllib.request.Request(f"{ARCTIC}/posts/search?{urllib.parse.urlencode(params)}", headers={"User-Agent": _user_agent()}),
        timeout=20,
    ) as resp:
        rows = json.loads(resp.read()).get("data") or []
    return [
        {
            "id": d.get("id"), "title": d.get("title", ""), "subreddit": d.get("subreddit"),
            "author": d.get("author"), "created": _iso(d.get("created_utc")),
            "url": WEB + (d.get("permalink") or f"/comments/{d.get('id')}/"),
            "snippet": _clip(d.get("selftext") or ""),
        }
        for d in rows
    ]

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

    if action == "search":
        if not query.strip():
            raise ToolError("action 'search' needs a query")
        sub = _subreddit(subreddit) if subreddit.strip() else None
        order = sort if sort != "hot" else "relevance"
        if creds:
            path = f"/r/{sub}/search" if sub else "/search"
            params: dict[str, Any] = {"q": query, "sort": order, "t": period, "limit": count}
            if sub:
                params["restrict_sr"] = 1
            data = await asyncio.to_thread(_api, path, params, creds)
            posts = [shape_api_post(c["data"]) for c in (data.get("data") or {}).get("children") or []]
            return _result(action, "api", {"query": query, "subreddit": sub, "results": posts[:count]})
        route, posts = await _first_of(
            ("web", lambda: web_search(query, sub, order, period, count)),
            ("archive", lambda: arctic_search(query, sub, count)),
            ("feed", lambda: feed_posts(_feed(f"/r/{sub}/search.rss" if sub else "/search.rss",
                                              {"q": query, "sort": order, "t": period, "limit": count,
                                               **({"restrict_sr": 1} if sub else {})}))),
        )
        return _result(action, route, {"query": query, "subreddit": sub, "results": posts[:count]})

    if action == "subreddit":
        sub = _subreddit(subreddit or query)
        listing = sort if sort in ("new", "top", "hot") else "hot"
        if creds:
            params = {"limit": count, **({"t": period} if listing == "top" else {})}
            data = await asyncio.to_thread(_api, f"/r/{sub}/{listing}", params, creds)
            posts = [shape_api_post(c["data"]) for c in (data.get("data") or {}).get("children") or []]
            return _result(action, "api", {"subreddit": sub, "sort": listing, "results": posts[:count]})
        route, posts = await _first_of(
            ("web", lambda: web_listing(sub, listing, period, count)),
            ("feed", lambda: feed_posts(_feed(f"/r/{sub}/{listing}/.rss",
                                              {"limit": count, **({"t": period} if listing == "top" else {})}))),
        )
        return _result(action, route, {"subreddit": sub, "sort": listing, "results": posts[:count]})

    if action == "comments":
        pid = post_id(post or query)
        if creds:
            data = await asyncio.to_thread(
                _api, f"/comments/{pid}", {"limit": count, "depth": 1, "sort": "top"}, creds
            )
            return _result(action, "api", shape_api_comments(data, count))
        try:
            page = await asyncio.to_thread(_web, f"/svc/shreddit/comments/r/all/t3_{pid}")
            comments = parse_comments(page)
        except ToolError:
            comments = []
        if comments:
            head = await asyncio.to_thread(arctic_post, pid) or {"id": pid, "url": f"{WEB}/comments/{pid}/"}
            return _result(action, "web", {"post": head, "comments": comments[:count]})
        entries = await asyncio.to_thread(_feed, f"/comments/{pid}/.rss", {"limit": count + 1})
        posts = [e for e in entries if e["kind"] == "post"]
        comments = [
            {"author": e["author"], "created": e["created"], "text": e["text"], "url": e["url"]}
            for e in entries
            if e["kind"] == "comment" and e["text"] not in ("[deleted]", "[removed]")
        ][:count]
        head = feed_posts(posts)[0] if posts else {"id": pid}
        return _result(action, "feed", {"post": head, "comments": comments})

    raise ToolError(f"unknown action {action!r}")


async def _first_of(*routes: tuple[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """The first route that returns anything, and its name.

    A route that fails, or finds nothing, gives way to the next: an empty
    page from Reddit more often means its markup changed than that nothing
    matched, and the archive or the feeds will say which. The last route's
    answer stands, empty or not; its error is raised if it failed.
    """
    last: Exception | None = None
    for i, (name, fetch) in enumerate(routes):
        try:
            found = await asyncio.to_thread(fetch)
        except (ToolError, urllib.error.URLError, ValueError, OSError) as exc:
            last = exc
            continue
        if found or i == len(routes) - 1:
            return name, found
    if isinstance(last, ToolError):
        raise last
    raise ToolError(f"could not read Reddit: {last}", retryable=True)


def _result(action: str, route: str, body: dict[str, Any]) -> dict[str, Any]:
    source = {
        "api": "reddit api",
        "web": "reddit.com (public pages)",
        "archive": "arctic shift archive of reddit",
        "feed": "reddit feeds",
    }[route]
    out = {"action": action, "source": source, **body}
    if route == "feed":
        out["note"] = "Read from public feeds: no scores or comment counts. " + _setup_hint()
    elif route == "archive":
        out["note"] = "From an archive of Reddit: scores and comment counts are as archived, not current."
    return out
