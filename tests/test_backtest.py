import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest import BTParams, generate_signals, run_backtest, simulate, summarize  # noqa: E402


def synth(seed, drift=0.0005, n=600):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(drift, 0.018, n)))
    o = np.r_[c[0], c[:-1]] * (1 + r.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + abs(r.normal(0, 0.006, n)))
    lo = np.minimum(o, c) * (1 - abs(r.normal(0, 0.006, n)))
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": 1e5}, index=idx)


@pytest.mark.parametrize("seed", range(8))
def test_no_lookahead(seed):
    """Signals found on truncated history must equal full-history signals confirmed by then."""
    df = synth(seed, 0.001 if seed % 2 else -0.001)
    p = BTParams(trend_filter=seed % 3 == 0, regime_filter=seed % 4 == 0)
    full = generate_signals(df, p)
    for cut in (250, 380, 520):
        part = generate_signals(df.iloc[:cut], p)
        expected = [s for s in full if s["conf"] < cut - 1]
        assert len(part) == len(expected)
        for a, b in zip(part, expected):
            assert a["p1"] == b["p1"] and a["p2"] == b["p2"] and a["side"] == b["side"]
            assert a["target"] == pytest.approx(b["target"])


def test_signal_rules():
    p = BTParams()
    for seed in range(20):
        df = synth(seed, 0.001 if seed % 2 else -0.001)
        for s in generate_signals(df, p):
            if s["side"] == "Long":
                assert s["rsi2"] < s["rsi1"] and s["Y"] > s["W"]
                assert s["target"] == pytest.approx(s["Y"] + s["X"] - s["W"])
            else:
                assert s["rsi2"] > s["rsi1"] and s["Y"] < s["W"]
                assert s["target"] == pytest.approx(s["Y"] - (s["W"] - s["X"]))


def bars(rows):
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=1)


SIG = dict(side="Long", p1=0, p2=1, conf=1, W=90.0, X=110.0, Y=95.0, target=115.0,
           rsi1=45, rsi2=40, regime="Bull (40-80)", aligned=True, with_trend=True)


def test_target_exit():
    df = bars([[100, 101, 99, 100]] * 2 + [[100, 104, 99, 103], [103, 116, 102, 115]])
    t = simulate(df, [SIG], BTParams(cost_pct=0), "X")[0]
    assert t["Exit Reason"] == "Target" and t["Exit"] == 115 and t["Return %"] == 15.0


def test_stop_beats_target_same_bar():
    df = bars([[100, 101, 99, 100]] * 2 + [[100, 120, 90, 94]])
    t = simulate(df, [SIG], BTParams(cost_pct=0, stop_mode="close"), "X")[0]
    assert t["Exit Reason"] == "Stop" and t["Exit"] == 94


def test_gap_through_stop_fills_at_open():
    df = bars([[100, 101, 99, 100]] * 3 + [[92, 93, 91, 92]])
    t = simulate(df, [SIG], BTParams(cost_pct=0, stop_mode="intrabar"), "X")[0]
    assert t["Exit Reason"] == "Stop" and t["Exit"] == 92


def test_short_and_costs():
    sig = dict(SIG, side="Short", Y=105.0, target=85.0)
    df = bars([[100, 101, 99, 100]] * 2 + [[100, 101, 84, 86]])
    t = simulate(df, [sig], BTParams(cost_pct=0.1), "X")[0]
    assert t["Exit Reason"] == "Target" and t["Return %"] == pytest.approx(14.9)


def test_time_exit_and_open():
    df = bars([[100, 101, 99, 100]] * 10)
    t = simulate(df, [SIG], BTParams(cost_pct=0, max_hold=3), "X")[0]
    assert t["Exit Reason"] == "Time" and t["Bars Held"] == 3
    t = simulate(df, [SIG], BTParams(cost_pct=0, max_hold=50), "X")[0]
    assert t["Exit Reason"] == "Open"


def test_universe_run_and_summary():
    data = {f"S{i}": synth(i, 0.001 if i % 2 else -0.001) for i in range(30)}
    trades = run_backtest(data, BTParams())
    assert len(trades) > 20
    s = summarize(trades)
    assert 0 <= s["Win rate %"] <= 100
    closed = trades[trades["Exit Reason"] != "Open"]
    assert s["Sum of returns %"] == pytest.approx(closed["Return %"].sum(), abs=0.1)
    # one open position per symbol: no overlapping trades
    for _, g in trades.groupby("Symbol"):
        g = g.sort_values("Entry Date")
        assert (g["Entry Date"].values[1:] > g["Exit Date"].values[:-1]).all()
