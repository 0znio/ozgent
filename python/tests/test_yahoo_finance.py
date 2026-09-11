"""Shaping tests for yahoo_finance.

The endpoints are not a contract, so these pin the shapes Yahoo actually
serves -- trimmed from real responses -- and what the model is handed in
return. The network half is exercised by hand; see the module docstring for
what each endpoint needs.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ozgent_tools.base import ToolError
from ozgent_tools.builtin.yahoo_finance import (
    MAX_ROWS, _symbols, raw, shape_fundamentals, shape_history, shape_quote, shape_search, thin,
)

CHART = {"chart": {"result": [{
    "meta": {"currency": "USD", "symbol": "NVDA", "fullExchangeName": "NasdaqGS"},
    "timestamp": [1757000000, 1757086400, 1757172800],
    "indicators": {"quote": [{
        "open": [100.0, 101.0, None], "high": [102.0, 104.5, None],
        "low": [99.0, 100.5, None], "close": [101.0, 104.0, None],
        "volume": [1000, 3000, None],
    }]},
}], "error": None}}


class Shaping(unittest.TestCase):
    def test_raw_flattens_formatted_pairs_at_any_depth(self):
        assert raw({"a": {"raw": 1.5, "fmt": "1.50"}, "b": [{"raw": 2}], "maxAge": 1}) == {"a": 1.5, "b": [2]}
        assert raw({}) is None

    def test_a_quote_keeps_the_fields_a_model_needs_under_plain_names(self):
        q = shape_quote({
            "symbol": "NVDA", "longName": "NVIDIA Corporation", "regularMarketPrice": 218.36,
            "regularMarketChangePercent": -2.374123, "marketCap": 5272738725888,
            "regularMarketTime": 1757534400, "someNoise": "x",
        })
        assert q["price"] == 218.36 and q["name"] == "NVIDIA Corporation"
        assert q["change_percent"] == -2.3741
        assert q["as_of"] == "2025-09-10T20:00:00Z"
        assert "someNoise" not in q

    def test_history_skips_empty_bars_and_summarises_the_change(self):
        h = shape_history(CHART, "5d", "1d")
        assert h["summary"]["bars"] == 2
        assert h["summary"]["first_close"] == 101.0 and h["summary"]["last_close"] == 104.0
        assert h["summary"]["change_percent"] == 2.97
        assert h["summary"]["high"] == 104.5 and h["summary"]["low"] == 99.0
        assert h["summary"]["avg_volume"] == 2000

    def test_an_empty_history_is_an_error_not_an_empty_answer(self):
        with self.assertRaises(ToolError):
            shape_history({"chart": {"result": []}}, "1d", "5m")

    def test_thinning_keeps_both_ends_and_the_limit(self):
        rows = list(range(500))
        out = thin(rows)
        assert len(out) == MAX_ROWS and out[0] == 0 and out[-1] == 499
        assert thin([1, 2, 3]) == [1, 2, 3]

    def test_fundamentals_flatten_and_drop_what_is_missing(self):
        out = shape_fundamentals({"quoteSummary": {"result": [{
            "assetProfile": {"sector": "Technology", "longBusinessSummary": "x" * 900},
            "financialData": {"targetMeanPrice": {"raw": 250.0, "fmt": "250"}, "totalDebt": {}},
            "recommendationTrend": {"trend": [{"period": "0m", "strongBuy": 20, "buy": 30,
                                               "hold": 5, "sell": 0, "strongSell": 1}]},
            "calendarEvents": {"earnings": {"earningsDate": [{"raw": 1760000000}]}},
        }]}})
        assert out["profile"]["sector"] == "Technology"
        assert out["profile"]["summary"].endswith("…") and len(out["profile"]["summary"]) == 601
        assert out["financialData"] == {"targetMeanPrice": 250.0}
        assert out["analyst_recommendations"][0]["strongBuy"] == 20
        assert out["calendar"]["next_earnings"] == ["2025-10-09T08:53:20Z"]

    def test_search_returns_symbols_and_headlines(self):
        out = shape_search({
            "quotes": [{"symbol": "NVDA", "longname": "NVIDIA Corporation", "quoteType": "EQUITY"},
                       {"longname": "no symbol"}],
            "news": [{"title": "Chips", "publisher": "Wire", "link": "https://x", "providerPublishTime": 1757534400}],
        })
        assert out["symbols"] == [{"symbol": "NVDA", "name": "NVIDIA Corporation", "type": "EQUITY"}]
        assert out["news"][0]["published"] == "2025-09-10T20:00:00Z"

    def test_symbols_are_split_uppercased_and_bounded(self):
        assert _symbols("nvda, aapl") == ["NVDA", "AAPL"]
        assert _symbols("brk-b btc-usd") == ["BRK-B", "BTC-USD"]
        with self.assertRaises(ToolError):
            _symbols(" , ")
        with self.assertRaises(ToolError):
            _symbols(",".join(["A"] * 11))


if __name__ == "__main__":
    unittest.main()
