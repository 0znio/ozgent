"""Technical indicators, computed from daily bars.

Plain Python on purpose: the tool worker has no third-party dependencies, and
these are a few loops each. Every indicator uses its textbook definition —
Wilder's smoothing for RSI and ATR, a 12/26/9 EMA MACD, 20-day Bollinger bands
at two standard deviations — so a number here can be checked against any
charting site.

What the model is handed is the latest reading of each, plus a few plain-words
judgements ("above its 200-day average", "RSI overbought") that a small model
would otherwise get wrong from the raw numbers. The judgements are rules, not
advice, and say which rule they are.
"""

from __future__ import annotations

from typing import Any, Sequence


def sma(values: Sequence[float], n: int) -> float | None:
    """Simple moving average of the last `n` values."""
    if len(values) < n or n <= 0:
        return None
    return sum(values[-n:]) / n


def ema_series(values: Sequence[float], n: int) -> list[float]:
    """Exponential moving average, seeded with the SMA of the first `n`."""
    if len(values) < n:
        return []
    k = 2 / (n + 1)
    out = [sum(values[:n]) / n]
    for v in values[n:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(closes: Sequence[float], n: int = 14) -> float | None:
    """Relative strength index with Wilder's smoothing."""
    if len(closes) <= n:
        return None
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = sum(gains[:n]) / n
    avg_loss = sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):
        avg_gain = (avg_gain * (n - 1) + g) / n
        avg_loss = (avg_loss * (n - 1) + l) / n
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, float] | None:
    """MACD line, signal line and histogram, latest values."""
    if len(closes) < slow + signal:
        return None
    ema_fast = ema_series(closes, fast)
    ema_slow = ema_series(closes, slow)
    # Align: the slow EMA starts `slow - fast` bars later than the fast one.
    offset = slow - fast
    line = [f - s for f, s in zip(ema_fast[offset:], ema_slow)]
    sig = ema_series(line, signal)
    if not sig:
        return None
    hist = line[-1] - sig[-1]
    prev_hist = line[-2] - sig[-2] if len(sig) >= 2 else hist
    return {"macd": line[-1], "signal": sig[-1], "histogram": hist, "previous_histogram": prev_hist}


def bollinger(closes: Sequence[float], n: int = 20, width: float = 2.0) -> dict[str, float] | None:
    """Bollinger bands around the `n`-day average."""
    if len(closes) < n:
        return None
    window = closes[-n:]
    mid = sum(window) / n
    var = sum((c - mid) ** 2 for c in window) / n
    sd = var ** 0.5
    return {"middle": mid, "upper": mid + width * sd, "lower": mid - width * sd}


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], n: int = 14) -> float | None:
    """Average true range with Wilder's smoothing: typical daily movement."""
    if len(closes) <= n:
        return None
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    value = sum(trs[:n]) / n
    for tr in trs[n:]:
        value = (value * (n - 1) + tr) / n
    return value


def swing_levels(highs: Sequence[float], lows: Sequence[float], price: float, span: int = 5,
                 lookback: int = 130, tolerance: float = 0.015) -> dict[str, list[float]]:
    """Support and resistance from recent swing lows and highs.

    A swing low is a bar lower than the `span` bars either side of it; nearby
    swings (within `tolerance` of each other) are merged into one level.
    Levels below the price are support, above it resistance, nearest first.
    """
    start = max(span, len(lows) - lookback)
    points: list[float] = []
    for i in range(start, len(lows) - span):
        if lows[i] == min(lows[i - span:i + span + 1]):
            points.append(lows[i])
        if highs[i] == max(highs[i - span:i + span + 1]):
            points.append(highs[i])
    clusters: list[list[float]] = []
    for p in sorted(points):
        if clusters and abs(p - clusters[-1][-1]) <= clusters[-1][-1] * tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = [(sum(c) / len(c), len(c)) for c in clusters]
    support = sorted((lv for lv in levels if lv[0] < price), key=lambda lv: price - lv[0])[:3]
    resistance = sorted((lv for lv in levels if lv[0] > price), key=lambda lv: lv[0] - price)[:3]
    return {"support": [lv[0] for lv in support], "resistance": [lv[0] for lv in resistance]}


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return (a - b) / b * 100


def _r(x: float | None, places: int = 2) -> float | None:
    return None if x is None else round(x, places)


def analyse(bars: list[dict[str, Any]]) -> dict[str, Any]:
    """Every indicator's latest reading, with plain-words signals.

    `bars` are daily, oldest first, each with open/high/low/close/volume.
    """
    bars = [b for b in bars if b.get("close") is not None]
    if len(bars) < 30:
        raise ValueError(f"only {len(bars)} daily bars; technicals need at least 30")
    closes = [float(b["close"]) for b in bars]
    highs = [float(b.get("high") if b.get("high") is not None else b["close"]) for b in bars]
    lows = [float(b.get("low") if b.get("low") is not None else b["close"]) for b in bars]
    volumes = [float(b.get("volume") or 0) for b in bars]
    price = closes[-1]

    averages = {f"sma_{n}": _r(sma(closes, n)) for n in (20, 50, 200)}
    ema12, ema26 = ema_series(closes, 12), ema_series(closes, 26)
    averages["ema_12"] = _r(ema12[-1]) if ema12 else None
    averages["ema_26"] = _r(ema26[-1]) if ema26 else None
    rsi14 = rsi(closes)
    m = macd(closes)
    bands = bollinger(closes)
    atr14 = atr(highs, lows, closes)
    levels = swing_levels(highs, lows, price)

    returns = {
        label: _r(_pct(price, closes[-1 - n])) if len(closes) > n else None
        for label, n in (("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252))
    }
    vol20, vol50 = sma(volumes, 20), sma(volumes, 50)

    signals: list[str] = []
    s50, s200 = sma(closes, 50), sma(closes, 200)
    if s200 is not None:
        signals.append(f"price is {'above' if price > s200 else 'below'} its 200-day average (long-term trend {'up' if price > s200 else 'down'})")
    if s50 is not None and s200 is not None:
        prev50, prev200 = sma(closes[:-1], 50), sma(closes[:-1], 200)
        if prev50 is not None and prev200 is not None and (prev50 - prev200) * (s50 - s200) < 0:
            signals.append("50-day average just crossed " + ("above the 200-day (golden cross)" if s50 > s200 else "below the 200-day (death cross)"))
        else:
            signals.append(f"50-day average is {'above' if s50 > s200 else 'below'} the 200-day")
    if rsi14 is not None:
        if rsi14 >= 70:
            signals.append(f"RSI {rsi14:.0f}: overbought by the usual 70 rule")
        elif rsi14 <= 30:
            signals.append(f"RSI {rsi14:.0f}: oversold by the usual 30 rule")
        else:
            signals.append(f"RSI {rsi14:.0f}: neither overbought nor oversold")
    if m:
        side = "above" if m["macd"] > m["signal"] else "below"
        turning = (m["histogram"] > 0) != (m["previous_histogram"] > 0)
        signals.append(f"MACD is {side} its signal line" + (" (just crossed — momentum turning)" if turning else ""))
    if bands:
        if price > bands["upper"]:
            signals.append("price is above the upper Bollinger band (stretched)")
        elif price < bands["lower"]:
            signals.append("price is below the lower Bollinger band (stretched)")
    if vol20 and vol50:
        ratio = vol20 / vol50
        if ratio > 1.25:
            signals.append(f"volume rising: 20-day average is {ratio:.1f}× the 50-day")
        elif ratio < 0.8:
            signals.append(f"volume fading: 20-day average is {ratio:.1f}× the 50-day")

    return {
        "price": _r(price),
        "as_of": bars[-1].get("time"),
        "bars": len(bars),
        "moving_averages": averages,
        "price_vs": {k.replace("sma_", "sma"): _r(_pct(price, sma(closes, int(k.split('_')[1])))) for k in ("sma_20", "sma_50", "sma_200")},
        "rsi_14": _r(rsi14, 1),
        "macd": {k: _r(v, 3) for k, v in m.items() if k != "previous_histogram"} if m else None,
        "bollinger_20": {k: _r(v) for k, v in bands.items()} if bands else None,
        "atr_14": _r(atr14),
        "atr_percent": _r(_pct(price + (atr14 or 0), price)) if atr14 else None,
        "support": [_r(x) for x in levels["support"]],
        "resistance": [_r(x) for x in levels["resistance"]],
        "high_52w": _r(max(highs[-252:])),
        "low_52w": _r(min(lows[-252:])),
        "from_52w_high_percent": _r(_pct(price, max(highs[-252:]))),
        "returns_percent": returns,
        "volume": {
            "last": int(volumes[-1]),
            "avg_20d": int(vol20) if vol20 else None,
            "avg_50d": int(vol50) if vol50 else None,
        },
        "signals": signals,
        "note": "Rules of thumb read off the numbers above, not advice.",
    }
