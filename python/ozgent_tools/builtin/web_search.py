"""Web search across interchangeable providers.

Every provider is normalised to one result shape, so switching providers never
changes what the model sees and never invalidates a prompt that depended on the
old format.

Configure in ``~/ozgent/configs/config.toml``::

    [tools.config.web_search]
    provider = "brave"          # brave | tavily | duckduckgo
    max_results = 5

    [tools.config.web_search.brave]
    api_key = "..."             # or set BRAVE_API_KEY

    [tools.config.web_search.tavily]
    api_key = "..."             # or set TAVILY_API_KEY
"""

from __future__ import annotations

import html
import os
import re
from typing import Annotated, Any, Awaitable, Callable, Literal

from ..base import ToolError, get_config, tool
from ..http import HttpError, get_json, post_json, request

Category = Literal["web", "news"]

# provider name -> coroutine(query, category, count, settings) -> payload
PROVIDERS: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}


def provider(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        PROVIDERS[name] = fn
        return fn

    return wrap


def _api_key(settings: dict[str, Any], provider_name: str, env: str) -> str:
    """Prefer the environment over the config file, so a key never has to be
    written to disk to be used."""
    key = os.environ.get(env) or (settings.get(provider_name) or {}).get("api_key")
    if not key:
        raise ToolError(
            f"web_search provider {provider_name!r} needs an API key. "
            f"Set ${env}, or add [tools.config.web_search.{provider_name}] "
            f"api_key to config.toml."
        )
    return str(key)


def _result(
    title: str, url: str, snippet: str = "", published: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    out = {"title": title.strip(), "url": url.strip(), "snippet": snippet.strip()}
    if published:
        out["published"] = published
    if source:
        out["source"] = source
    return out


# --------------------------------------------------------------------- brave


@provider("brave")
async def _brave(query: str, category: str, count: int, settings: dict[str, Any]) -> dict[str, Any]:
    key = _api_key(settings, "brave", "BRAVE_API_KEY")
    endpoint = "news" if category == "news" else "web"
    url = f"https://api.search.brave.com/res/v1/{endpoint}/search"

    payload = await get_json(
        url,
        params={"q": query, "count": min(count, 20)},
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
    )

    container = payload.get("results") if category == "news" else (payload.get("web") or {}).get("results")
    results = []
    for item in container or []:
        results.append(
            _result(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=_strip_tags(item.get("description", "")),
                published=item.get("age") or item.get("page_age"),
                source=(item.get("meta_url") or {}).get("hostname"),
            )
        )
    return {"results": results}


# -------------------------------------------------------------------- tavily


@provider("tavily")
async def _tavily(query: str, category: str, count: int, settings: dict[str, Any]) -> dict[str, Any]:
    key = _api_key(settings, "tavily", "TAVILY_API_KEY")
    opts = settings.get("tavily") or {}

    body: dict[str, Any] = {
        "query": query,
        "topic": "news" if category == "news" else "general",
        "max_results": min(count, 20),
        "search_depth": opts.get("search_depth", "basic"),
        # Tavily's synthesised answer is genuinely useful context, and costs
        # nothing extra on the basic tier.
        "include_answer": opts.get("include_answer", True),
    }
    if opts.get("days"):
        body["days"] = opts["days"]

    payload = await post_json(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}"},
        json_body=body,
    )

    results = [
        _result(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
            published=item.get("published_date"),
        )
        for item in payload.get("results", [])
    ]
    out: dict[str, Any] = {"results": results}
    if payload.get("answer"):
        out["answer"] = payload["answer"]
    return out


# ---------------------------------------------------------------- duckduckgo


@provider("duckduckgo")
async def _duckduckgo(query: str, category: str, count: int, settings: dict[str, Any]) -> dict[str, Any]:
    """Keyless fallback. Scrapes the lite endpoint, so it is the least reliable
    of the three but needs no account."""
    if category == "news":
        query = f"{query} news"

    body = await request(
        "https://lite.duckduckgo.com/lite/",
        method="POST",
        data={"q": query},
        headers={"Accept": "text/html", "Content-Type": "application/x-www-form-urlencoded"},
    )

    results = [
        _result(title=title, url=url, snippet=snippet)
        for title, url, snippet in _parse_ddg_lite(body)[:count]
    ]
    if not results:
        raise ToolError(
            "duckduckgo returned no parseable results; it may be rate-limiting. "
            "Configure the brave or tavily provider for reliable search.",
            retryable=True,
        )
    return {"results": results}


_DDG_LINK = re.compile(
    r'<a[^>]+class="result-link"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET = re.compile(
    r'<td[^>]+class="result-snippet"[^>]*>(?P<snippet>.*?)</td>', re.IGNORECASE | re.DOTALL
)


def _parse_ddg_lite(body: str) -> list[tuple[str, str, str]]:
    links = _DDG_LINK.finditer(body)
    snippets = _DDG_SNIPPET.findall(body)
    out: list[tuple[str, str, str]] = []
    for i, match in enumerate(links):
        url = _unwrap_ddg_redirect(html.unescape(match.group("url")))
        title = _strip_tags(match.group("title"))
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        if url.startswith("http"):
            out.append((title, url, snippet))
    return out


def _unwrap_ddg_redirect(url: str) -> str:
    """DuckDuckGo wraps outbound links in ``//duckduckgo.com/l/?uddg=...``."""
    if "uddg=" not in url:
        return url
    import urllib.parse

    parsed = urllib.parse.urlparse(url if url.startswith("http") else f"https:{url}")
    target = urllib.parse.parse_qs(parsed.query).get("uddg")
    return urllib.parse.unquote(target[0]) if target else url


_TAG = re.compile(r"<[^>]+>")


def _strip_tags(text: str) -> str:
    return html.unescape(_TAG.sub("", text or "")).strip()


# ----------------------------------------------------------------- the tool


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "provider": {"type": "string"},
        "query": {"type": "string"},
        "category": {"type": "string"},
        "answer": {"type": "string", "description": "Synthesised answer, when the provider supplies one."},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                    "published": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["title", "url"],
            },
        },
    },
    "required": ["provider", "query", "results"],
}


@tool(output_schema=OUTPUT_SCHEMA)
async def web_search(
    query: Annotated[str, "What to search for. Use plain keywords, not a question."],
    category: Annotated[Category, "'news' restricts to recent news coverage."] = "web",
    count: Annotated[int, "How many results to return, 1-20."] = 5,
) -> dict[str, Any]:
    """Look up current facts on the web: news, prices, events, or anything after your training cutoff.

    Only the first line of this docstring reaches the model, so it has to carry
    the whole rule. The previous wording ended "...or any claim you would
    otherwise have to guess at", which invited a search for anything the model
    felt unsure about — including describing an image it had already been shown.
    """
    # Which engine runs the search is the user's configuration, never the
    # model's choice. It was briefly a parameter, and models duly overrode a
    # configured Brave key with duckduckgo — spending the user's setup on a
    # provider they had not chosen. Nothing about picking an engine needs the
    # model's judgement, so it is not offered one.
    settings = get_config("web_search")
    name = (settings.get("provider") or "duckduckgo").lower()

    fn = PROVIDERS.get(name)
    if fn is None:
        raise ToolError(
            f"unknown search provider {name!r}. Available: {', '.join(sorted(PROVIDERS))}"
        )

    count = max(1, min(int(count), 20))

    try:
        payload = await fn(query, category, count, settings)
    except HttpError as exc:
        if exc.status in (401, 403):
            raise ToolError(f"{name} rejected the API key (HTTP {exc.status})") from exc
        if exc.status == 429:
            raise ToolError(f"{name} is rate-limiting; try again shortly", retryable=True) from exc
        raise ToolError(f"{name} search failed: {exc} {exc.body}".strip(), retryable=True) from exc

    return {"provider": name, "query": query, "category": category, **payload}
