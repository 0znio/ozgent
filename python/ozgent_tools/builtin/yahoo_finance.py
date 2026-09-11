"""Market data from Yahoo Finance: quotes, price history, fundamentals, news.

No API key. Yahoo's public endpoints serve anyone with a session cookie, and
the quote and fundamentals endpoints additionally want a "crumb" -- a token
bound to that cookie. Both are fetched once per worker and reused; a crumb
Yahoo stops accepting is refreshed once and the call retried.

What bit, in order of how long it took to find:

- Yahoo answers 429 "Too Many Requests" to clients it does not like, whatever
  the actual rate. A bare library user agent gets it on every call, and so
  does a browser user agent sent from a client that is plainly not a browser.
  A plain ``Mozilla/5.0 (compatible; ozgent/...)`` is accepted, and says
  truthfully what is asking.
- The crumb is useless without the cookie it was issued with, so the two
  live in one cookie jar and are always used together.
- Numbers come back as ``{"raw": 1.23, "fmt": "1.23"}`` from some endpoints and
  bare from others. Everything handed to the model is flattened to the raw
  value, so it never has to parse "3.2T".

Configure in ``~/ozgent/configs/config.toml`` (all optional)::

    [tools.config.yahoo_finance]
    region = "US"
    lang = "en-US"
"""

from __future__ import annotations

import asyncio
import http.cookiejar
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated, Any, Literal

from .. import indicators
from ..base import ToolError, get_config, tool

USER_AGENT = "Mozilla/5.0 (compatible; ozgent/0.1; +https://github.com/0znio/ozgent)"

Action = Literal["quote", "history", "technicals", "fundamentals", "news", "search"]
Range = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

#: The bar size that suits each range: enough points to see the shape, few
#: enough that the history fits in a prompt.
INTERVALS = {
    "1d": "5m", "5d": "30m", "1mo": "1d", "3mo": "1d", "6mo": "1d",
    "1y": "1wk", "2y": "1wk", "5y": "1mo", "10y": "1mo", "ytd": "1d", "max": "3mo",
}

#: Most rows of history returned. Beyond this the series is thinned evenly,
#: keeping the first and last bars, because those are the ones a change is
#: measured between.
MAX_ROWS = 60

#: Symbols one quote call may ask about.
MAX_SYMBOLS = 10

#: quoteSummary modules asked for by "fundamentals".
MODULES = (
    "price,summaryDetail,financialData,defaultKeyStatistics,"
    "recommendationTrend,calendarEvents,assetProfile"
)

#: Seconds a crumb is trusted before it is fetched again anyway.
CRUMB_TTL = 6 * 3600


class _Session:
    """One cookie jar and the crumb that goes with it, shared by every call."""

    def __init__(self) -> None:
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        # `*/*` rather than JSON: the crumb endpoint answers text/plain, and
        # asking it for JSON gets a 406 instead of a crumb.
        self.opener.addheaders = [("User-Agent", USER_AGENT), ("Accept", "*/*")]
        self.crumb: str | None = None
        self.fetched = 0.0
        # Calls run in worker threads; two racing to fetch a crumb would each
        # reset the other's cookie.
        self.lock = threading.Lock()

    def get(self, url: str, timeout: float = 20.0) -> tuple[int, str]:
        try:
            with self.opener.open(url, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")[:500]

    def ensure_crumb(self, force: bool = False) -> str:
        with self.lock:
            if self.crumb and not force and time.time() - self.fetched < CRUMB_TTL:
                return self.crumb
            self.jar.clear()
            # This answers 404, and that is fine: the response still sets the
            # session cookie, which is all it is visited for.
            self.get("https://fc.yahoo.com/")
            status, body = self.get("https://query1.finance.yahoo.com/v1/test/getcrumb")
            crumb = body.strip()
            if status != 200 or not crumb or "<" in crumb or " " in crumb:
                raise ToolError(
                    f"Yahoo Finance did not issue a session (HTTP {status}): {crumb[:80]}",
                    retryable=True,
                )
            self.crumb, self.fetched = crumb, time.time()
            return crumb


_SESSION = _Session()


def _fetch(path: str, params: dict[str, Any], *, crumb: bool = False) -> Any:
    """GET a Yahoo endpoint and parse its JSON, blocking."""
    settings = get_config("yahoo_finance")
    params = {"lang": settings.get("lang", "en-US"), "region": settings.get("region", "US"), **params}

    for attempt in range(2):
        query = dict(params)
        if crumb:
            query["crumb"] = _SESSION.ensure_crumb(force=attempt > 0)
        url = f"https://query1.finance.yahoo.com{path}?{urllib.parse.urlencode(query)}"
        status, body = _SESSION.get(url)
        # A crumb Yahoo has stopped honouring comes back as 401 with
        # "Invalid Crumb"; a fresh session fixes it, a second failure will not.
        if crumb and status in (401, 403) and attempt == 0:
            continue
        if status == 429:
            raise ToolError("Yahoo Finance is rate-limiting; try again shortly", retryable=True)
        if status == 404:
            raise ToolError(_not_found(body))
        if status >= 400:
            raise ToolError(f"Yahoo Finance answered HTTP {status}: {body[:200]}", retryable=status >= 500)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ToolError(f"Yahoo Finance returned something that is not JSON: {body[:120]}") from exc
    raise ToolError("Yahoo Finance refused the session twice", retryable=True)


def _not_found(body: str) -> str:
    try:
        detail = json.loads(body)
        for key in ("chart", "quoteSummary", "finance"):
            err = (detail.get(key) or {}).get("error") or {}
            if err.get("description"):
                return f"{err['description']}. Use action 'search' to find the right symbol."
    except (json.JSONDecodeError, AttributeError):
        pass
    return "no such symbol. Use action 'search' to find the right one."


def raw(value: Any) -> Any:
    """Flatten Yahoo's ``{"raw": x, "fmt": "..."}`` pairs to the raw value."""
    if isinstance(value, dict):
        if "raw" in value:
            return value["raw"]
        if not value:
            return None
        return {k: raw(v) for k, v in value.items() if k not in ("maxAge",)}
    if isinstance(value, list):
        return [raw(v) for v in value]
    return value


def _iso(seconds: Any) -> str | None:
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def _round(x: Any, places: int = 4) -> Any:
    return round(x, places) if isinstance(x, float) else x


# ---------------------------------------------------------------- actions

#: The quote fields worth a model's attention, under the names it gets.
QUOTE_FIELDS = {
    "symbol": "symbol",
    "longName": "name",
    "quoteType": "type",
    "fullExchangeName": "exchange",
    "currency": "currency",
    "marketState": "market_state",
    "regularMarketPrice": "price",
    "regularMarketChange": "change",
    "regularMarketChangePercent": "change_percent",
    "regularMarketPreviousClose": "previous_close",
    "regularMarketOpen": "open",
    "regularMarketDayHigh": "day_high",
    "regularMarketDayLow": "day_low",
    "regularMarketVolume": "volume",
    "averageDailyVolume3Month": "avg_volume_3m",
    "fiftyTwoWeekHigh": "week52_high",
    "fiftyTwoWeekLow": "week52_low",
    "fiftyDayAverage": "avg_50d",
    "twoHundredDayAverage": "avg_200d",
    "marketCap": "market_cap",
    "trailingPE": "pe_trailing",
    "forwardPE": "pe_forward",
    "epsTrailingTwelveMonths": "eps_ttm",
    "epsForward": "eps_forward",
    "dividendYield": "dividend_yield",
    "trailingAnnualDividendYield": "dividend_yield_trailing",
    "priceToBook": "price_to_book",
    "averageAnalystRating": "analyst_rating",
    "preMarketPrice": "pre_market_price",
    "postMarketPrice": "post_market_price",
}


def shape_quote(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for src, dst in QUOTE_FIELDS.items():
        if item.get(src) is not None:
            out[dst] = _round(raw(item[src]))
    if "name" not in out and item.get("shortName"):
        out["name"] = item["shortName"]
    when = _iso(raw(item.get("regularMarketTime")))
    if when:
        out["as_of"] = when
    return out


def _quote(symbols: list[str]) -> dict[str, Any]:
    data = _fetch("/v7/finance/quote", {"symbols": ",".join(symbols)}, crumb=True)
    results = (data.get("quoteResponse") or {}).get("result") or []
    if not results:
        raise ToolError(f"no quote for {', '.join(symbols)}. Use action 'search' to find the symbol.")
    found = {r.get("symbol") for r in results}
    out: dict[str, Any] = {"quotes": [shape_quote(r) for r in results]}
    missing = [s for s in symbols if s not in found]
    if missing:
        out["not_found"] = missing
    return out


def thin(rows: list[Any], limit: int = MAX_ROWS) -> list[Any]:
    """Evenly thin a series to at most `limit` rows, keeping both ends."""
    if len(rows) <= limit:
        return rows
    step = (len(rows) - 1) / (limit - 1)
    return [rows[round(i * step)] for i in range(limit)]


def shape_history(data: dict[str, Any], range_: str, interval: str) -> dict[str, Any]:
    result = ((data.get("chart") or {}).get("result") or [None])[0]
    if not result:
        raise ToolError("no price history returned for that symbol and range")
    meta = result.get("meta") or {}
    stamps = result.get("timestamp") or []
    quote = (((result.get("indicators") or {}).get("quote")) or [{}])[0]

    rows = []
    for i, t in enumerate(stamps):
        close = (quote.get("close") or [None] * len(stamps))[i]
        if close is None:
            continue  # a bar with no trades, which Yahoo reports as nulls
        rows.append({
            "time": _iso(t),
            "open": _round((quote.get("open") or [None] * len(stamps))[i]),
            "high": _round((quote.get("high") or [None] * len(stamps))[i]),
            "low": _round((quote.get("low") or [None] * len(stamps))[i]),
            "close": _round(close),
            "volume": (quote.get("volume") or [None] * len(stamps))[i],
        })
    if not rows:
        raise ToolError("the price history for that range is empty")

    first, last = rows[0]["close"], rows[-1]["close"]
    highs = [r["high"] for r in rows if r["high"] is not None]
    lows = [r["low"] for r in rows if r["low"] is not None]
    volumes = [r["volume"] for r in rows if r["volume"]]
    summary = {
        "start": rows[0]["time"],
        "end": rows[-1]["time"],
        "first_close": first,
        "last_close": last,
        "change": _round(last - first),
        "change_percent": _round((last - first) / first * 100, 2) if first else None,
        "high": max(highs) if highs else None,
        "low": min(lows) if lows else None,
        "avg_volume": int(sum(volumes) / len(volumes)) if volumes else None,
        "bars": len(rows),
    }
    return {
        "symbol": meta.get("symbol"),
        "currency": meta.get("currency"),
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
        "range": range_,
        "interval": interval,
        "summary": summary,
        "rows": thin(rows),
    }


def _history(symbol: str, range_: str) -> dict[str, Any]:
    interval = INTERVALS[range_]
    data = _fetch(
        f"/v8/finance/chart/{urllib.parse.quote(symbol)}",
        {"range": range_, "interval": interval, "includePrePost": "false"},
    )
    return shape_history(data, range_, interval)


def daily_bars(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Every daily bar in a chart response, oldest first, unthinned."""
    result = ((data.get("chart") or {}).get("result") or [None])[0]
    if not result:
        raise ToolError("no price history returned for that symbol")
    stamps = result.get("timestamp") or []
    quote = (((result.get("indicators") or {}).get("quote")) or [{}])[0]
    column = lambda k: quote.get(k) or [None] * len(stamps)  # noqa: E731
    opens, highs, lows, closes, volumes = (column(k) for k in ("open", "high", "low", "close", "volume"))
    return [
        {"time": _iso(t), "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i], "volume": volumes[i]}
        for i, t in enumerate(stamps)
        if closes[i] is not None
    ]


def _technicals(symbol: str) -> dict[str, Any]:
    # Two years of daily bars: the 200-day average needs 200 of them, and a
    # year beyond that gives the 52-week range and the swing levels room.
    data = _fetch(
        f"/v8/finance/chart/{urllib.parse.quote(symbol)}",
        {"range": "2y", "interval": "1d", "includePrePost": "false"},
    )
    meta = (((data.get("chart") or {}).get("result") or [{}])[0] or {}).get("meta") or {}
    try:
        reading = indicators.analyse(daily_bars(data))
    except ValueError as e:
        raise ToolError(f"{symbol}: {e}") from None
    return {
        "symbol": meta.get("symbol") or symbol,
        "currency": meta.get("currency"),
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
        **reading,
    }


def shape_fundamentals(data: dict[str, Any]) -> dict[str, Any]:
    result = ((data.get("quoteSummary") or {}).get("result") or [None])[0]
    if not result:
        raise ToolError("no fundamentals returned for that symbol")
    out: dict[str, Any] = {}

    profile = raw(result.get("assetProfile") or {}) or {}
    if profile:
        about = profile.get("longBusinessSummary") or ""
        out["profile"] = {
            k: v for k, v in {
                "sector": profile.get("sector"),
                "industry": profile.get("industry"),
                "country": profile.get("country"),
                "employees": profile.get("fullTimeEmployees"),
                "website": profile.get("website"),
                "summary": about[:600] + ("…" if len(about) > 600 else ""),
            }.items() if v
        }

    keep = {
        "price": ("longName", "currency", "regularMarketPrice", "marketCap", "exchangeName"),
        "summaryDetail": (
            "previousClose", "dayLow", "dayHigh", "fiftyTwoWeekLow", "fiftyTwoWeekHigh",
            "trailingPE", "forwardPE", "dividendYield", "payoutRatio", "beta",
            "priceToSalesTrailing12Months", "averageVolume",
        ),
        "financialData": (
            "targetMeanPrice", "targetHighPrice", "targetLowPrice", "numberOfAnalystOpinions",
            "recommendationKey", "recommendationMean", "totalRevenue", "revenueGrowth",
            "earningsGrowth", "grossMargins", "operatingMargins", "profitMargins",
            "returnOnEquity", "returnOnAssets", "totalCash", "totalDebt", "debtToEquity",
            "currentRatio", "freeCashflow", "operatingCashflow", "ebitda",
        ),
        "defaultKeyStatistics": (
            "enterpriseValue", "pegRatio", "priceToBook", "trailingEps", "forwardEps",
            "sharesOutstanding", "floatShares", "shortPercentOfFloat", "heldPercentInsiders",
            "heldPercentInstitutions", "52WeekChange", "SandP52WeekChange",
            "mostRecentQuarter", "lastFiscalYearEnd",
        ),
    }
    for module, fields in keep.items():
        block = raw(result.get(module) or {}) or {}
        picked = {f: _round(block.get(f)) for f in fields if block.get(f) not in (None, {}, "")}
        for f in ("mostRecentQuarter", "lastFiscalYearEnd"):
            if f in picked:
                picked[f] = _iso(picked[f])
        if picked:
            out[module] = picked

    trend = (raw(result.get("recommendationTrend") or {}) or {}).get("trend") or []
    if trend:
        out["analyst_recommendations"] = [
            {k: t.get(k) for k in ("period", "strongBuy", "buy", "hold", "sell", "strongSell")}
            for t in trend[:4]
        ]

    calendar = raw(result.get("calendarEvents") or {}) or {}
    earnings = calendar.get("earnings") or {}
    dates = [_iso(d) for d in earnings.get("earningsDate") or [] if _iso(d)]
    if dates or calendar.get("exDividendDate"):
        out["calendar"] = {
            k: v for k, v in {
                "next_earnings": dates,
                "earnings_estimate_avg": earnings.get("earningsAverage"),
                "revenue_estimate_avg": earnings.get("revenueAverage"),
                "ex_dividend": _iso(calendar.get("exDividendDate")),
            }.items() if v
        }
    return out


def _fundamentals(symbol: str) -> dict[str, Any]:
    data = _fetch(
        f"/v10/finance/quoteSummary/{urllib.parse.quote(symbol)}",
        {"modules": MODULES},
        crumb=True,
    )
    return {"symbol": symbol, **shape_fundamentals(data)}


def shape_search(data: dict[str, Any]) -> dict[str, Any]:
    quotes = [
        {
            k: v for k, v in {
                "symbol": q.get("symbol"),
                "name": q.get("longname") or q.get("shortname"),
                "type": q.get("quoteType"),
                "exchange": q.get("exchDisp") or q.get("exchange"),
                "sector": q.get("sectorDisp") or q.get("sector"),
            }.items() if v
        }
        for q in data.get("quotes") or []
        if q.get("symbol")
    ]
    news = [
        {
            k: v for k, v in {
                "title": n.get("title"),
                "publisher": n.get("publisher"),
                "published": _iso(n.get("providerPublishTime")),
                "url": n.get("link"),
                "tickers": n.get("relatedTickers"),
            }.items() if v
        }
        for n in data.get("news") or []
        if n.get("title")
    ]
    return {"symbols": quotes, "news": news}


def _search(query: str, quotes: int, news: int) -> dict[str, Any]:
    data = _fetch(
        "/v1/finance/search",
        {"q": query, "quotesCount": quotes, "newsCount": news, "enableFuzzyQuery": "false"},
    )
    return shape_search(data)


def _symbols(symbol: str) -> list[str]:
    parts = [s.strip().upper() for s in symbol.replace(" ", ",").split(",") if s.strip()]
    if not parts:
        raise ToolError("give a symbol, e.g. AAPL, or use action 'search' with a company name")
    if len(parts) > MAX_SYMBOLS:
        raise ToolError(f"at most {MAX_SYMBOLS} symbols per call")
    return parts


@tool(effect="read")
async def yahoo_finance(
    action: Annotated[
        Action,
        "quote: current price and key stats. history: price series over `range`. "
        "technicals: RSI, MACD, moving averages, Bollinger bands, ATR, support and "
        "resistance, 52-week range, returns and volume trend, from daily prices. "
        "fundamentals: financials, valuation, analyst targets, earnings dates. "
        "news: recent headlines. search: find the symbol for a company name.",
    ],
    symbol: Annotated[
        str, "Ticker, e.g. AAPL, BTC-USD, ^GSPC, RELIANCE.NS. Comma-separated for several quotes."
    ] = "",
    query: Annotated[str, "Company or topic to look up; for action 'search'."] = "",
    range: Annotated[Range, "Period for action 'history'."] = "1mo",
    count: Annotated[int, "How many headlines or search matches, 1-20."] = 8,
) -> dict[str, Any]:
    """Live stock, ETF, index and crypto data: quotes, price history, technical indicators, fundamentals, news, symbol search."""
    # One tool with an `action` rather than five tools: small models pick the
    # right action from one description far more reliably than they pick the
    # right tool out of five near-identical ones.
    count = max(1, min(int(count), 20))
    if action == "search":
        text = (query or symbol).strip()
        if not text:
            raise ToolError("action 'search' needs a query, e.g. query='nvidia'")
        result = await asyncio.to_thread(_search, text, count, 0)
        return {"action": action, "query": text, "symbols": result["symbols"]}

    # Headlines can be asked for by topic as well as by ticker; a model
    # researching "NVIDIA" reasonably puts that in `query`.
    if action == "news" and not symbol.strip() and query.strip():
        result = await asyncio.to_thread(_search, query.strip(), 0, count)
        if not result["news"]:
            raise ToolError(f"no recent headlines for {query!r}; try web_search with category 'news'")
        return {
            "action": action,
            "query": query.strip(),
            "results": [
                {"title": n["title"], "url": n.get("url", ""), "snippet": n.get("publisher", ""),
                 "published": n.get("published")}
                for n in result["news"]
            ],
        }

    symbols = _symbols(symbol)
    if action == "quote":
        return {"action": action, **await asyncio.to_thread(_quote, symbols)}
    if len(symbols) > 1:
        raise ToolError(f"action {action!r} takes one symbol; call it once per symbol")
    one = symbols[0]
    if action == "history":
        if range not in INTERVALS:
            raise ToolError(f"range must be one of {', '.join(INTERVALS)}")
        return {"action": action, **await asyncio.to_thread(_history, one, range)}
    if action == "technicals":
        return {"action": action, **await asyncio.to_thread(_technicals, one)}
    if action == "fundamentals":
        return {"action": action, **await asyncio.to_thread(_fundamentals, one)}
    if action == "news":
        result = await asyncio.to_thread(_search, one, 0, count)
        # Yahoo's news search matches loosely, so headlines actually tagged
        # with the symbol go first; the rest are still context.
        result["news"].sort(key=lambda n: one not in (n.get("tickers") or []))
        if not result["news"]:
            raise ToolError(f"no recent headlines for {one}; try web_search with category 'news'")
        # Shaped like web_search results so a front end renders them the same.
        return {
            "action": action,
            "symbol": one,
            "results": [
                {"title": n["title"], "url": n.get("url", ""), "snippet": n.get("publisher", ""),
                 "published": n.get("published")}
                for n in result["news"]
            ],
        }
    raise ToolError(f"unknown action {action!r}")
