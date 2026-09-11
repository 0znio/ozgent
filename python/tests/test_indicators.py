"""Indicators against values worked by hand and against their definitions."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ozgent_tools import indicators as ind
from ozgent_tools.builtin.yahoo_finance import daily_bars


def bars_from(closes, spread=1.0, volume=1000):
    return [
        {"time": f"d{i}", "open": c, "high": c + spread, "low": c - spread, "close": c, "volume": volume}
        for i, c in enumerate(closes)
    ]


class Averages(unittest.TestCase):
    def test_sma_is_the_mean_of_the_last_n(self):
        assert ind.sma([1, 2, 3, 4, 5], 3) == 4
        assert ind.sma([1, 2], 3) is None

    def test_ema_is_seeded_with_the_sma_and_tracks_a_constant(self):
        series = ind.ema_series([10.0] * 30, 12)
        assert all(abs(v - 10.0) < 1e-9 for v in series)
        assert ind.ema_series([1, 2], 5) == []


class Oscillators(unittest.TestCase):
    def test_rsi_is_100_when_it_only_rises_and_0_when_it_only_falls(self):
        assert ind.rsi(list(range(1, 40))) == 100.0
        assert ind.rsi(list(range(40, 1, -1))) < 1e-9

    def test_rsi_matches_wilders_worked_example(self):
        # The fourteen-period example from Wilder (1978), as reproduced by
        # StockCharts: 44.34 ... 46.28 gives RSI 70.46 on the first reading.
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
                  46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
        assert abs(ind.rsi(closes) - 70.46) < 0.1

    def test_macd_is_zero_on_a_flat_series_and_positive_on_a_rise(self):
        flat = ind.macd([50.0] * 60)
        assert abs(flat["macd"]) < 1e-9 and abs(flat["histogram"]) < 1e-9
        rising = ind.macd([float(i) for i in range(60)])
        assert rising["macd"] > 0

    def test_bollinger_bands_are_two_deviations_either_side(self):
        b = ind.bollinger([1.0, 3.0] * 10)
        assert b["middle"] == 2.0
        assert math.isclose(b["upper"], 4.0) and math.isclose(b["lower"], 0.0)

    def test_atr_of_a_steady_range_is_that_range(self):
        closes = [100.0] * 30
        assert math.isclose(ind.atr([101.0] * 30, [99.0] * 30, closes), 2.0)


class Levels(unittest.TestCase):
    def test_a_repeated_floor_is_support_and_a_repeated_ceiling_resistance(self):
        # Oscillates between 90 and 110 and ends at 100.
        wave = [100 + 10 * math.sin(i / 4) for i in range(120)] + [100.0]
        highs = [c + 0.5 for c in wave]
        lows = [c - 0.5 for c in wave]
        levels = ind.swing_levels(highs, lows, 100.0)
        assert levels["support"] and levels["resistance"]
        assert all(s < 100 for s in levels["support"])
        assert all(r > 100 for r in levels["resistance"])
        assert abs(levels["support"][0] - 89.5) < 1.5
        assert abs(levels["resistance"][0] - 110.5) < 1.5


class Analyse(unittest.TestCase):
    def test_an_uptrend_reads_as_one(self):
        closes = [100 + i * 0.5 for i in range(300)]
        out = ind.analyse(bars_from(closes))
        assert out["price"] == closes[-1]
        assert out["moving_averages"]["sma_200"] < out["price"]
        assert any("above its 200-day" in s for s in out["signals"])
        assert out["rsi_14"] == 100.0
        assert out["returns_percent"]["1y"] > 0
        assert out["high_52w"] >= out["price"]

    def test_too_little_history_is_refused_with_the_reason(self):
        with self.assertRaises(ValueError):
            ind.analyse(bars_from([1.0] * 10))

    def test_bars_with_no_trades_are_left_out(self):
        data = {"chart": {"result": [{
            "timestamp": [1, 2, 3],
            "indicators": {"quote": [{"open": [1, None, 3], "high": [1, None, 3], "low": [1, None, 3],
                                      "close": [1, None, 3], "volume": [5, None, 7]}]},
        }]}}
        assert [b["close"] for b in daily_bars(data)] == [1, 3]


if __name__ == "__main__":
    unittest.main()
