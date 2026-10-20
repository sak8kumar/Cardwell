"""
Cardwell RSI reversal backtester
================================

Walk-forward, no look-ahead:
  * An RSI pivot at bar i is only "known" at bar i + pivot_right (confirmation bar).
  * The price extreme at the second pivot uses at most pivot_right bars after it,
    so it never peeks past the confirmation bar.
  * Regime / EMA filters are evaluated with data up to the confirmation bar.
  * Entry = OPEN of the bar after confirmation.

Exit rules (whichever comes first):
  * Stop   : price beyond Y (second-pivot price). Mode "close" = a close beyond Y,
             mode "intrabar" = a touch of Y.
  * Target : Cardwell projection (positive: Y + (X - W); negative: Y - (W - X)).
  * Time   : max_hold bars, exit at close.
  If stop and target both trigger on the same bar, the STOP is assumed (conservative).
  Gaps through stop / target fill at the open.

One open position per symbol at a time. Costs are deducted per round trip.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from cardwell_screener import pivots, rsi_regime, rsi_wilder


@dataclass
class BTParams:
    rsi: int = 14
    pivot_left: int = 5
    pivot_right: int = 2
    price_window: int = 2
    min_gap: int = 5
    max_gap: int = 60
    min_rsi_diff: float = 2.0
    min_price_diff: float = 0.5
    ema: int = 50
    regime_lookback: int = 60
    trend_filter: bool = False      # long only above EMA, short only below
    regime_filter: bool = False     # long only in bull range, short only in bear range
    direction: str = "both"         # "long", "short", "both"
    stop_mode: str = "close"        # "close" or "intrabar"
    max_hold: int = 40
    cost_pct: float = 0.10          # round-trip cost, % of entry
    min_reward_risk: float = 0.0    # skip trades whose (target-entry)/(entry-stop) is below this


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def generate_signals(df: pd.DataFrame, p: BTParams) -> list:
    n = len(df)
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    rsi = rsi_wilder(df["Close"], p.rsi)
    ema = df["Close"].ewm(span=p.ema, adjust=False).mean().values
    pw = min(p.price_window, p.pivot_right)
    sigs = []

    def ext(arr, i, fn):
        return fn(arr[max(0, i - pw): i + pw + 1])

    for kind in ("low", "high"):
        if kind == "low" and p.direction == "short":
            continue
        if kind == "high" and p.direction == "long":
            continue
        pv = pivots(rsi, p.pivot_left, p.pivot_right, kind)
        for j in range(1, len(pv)):
            p2 = pv[j]
            c = p2 + p.pivot_right            # confirmation bar
            if c >= n - 1:
                continue
            r2 = rsi.iloc[p2]
            best = None
            for p1 in reversed(pv[:j]):
                gap = p2 - p1
                if gap < p.min_gap:
                    continue
                if gap > p.max_gap:
                    break
                r1 = rsi.iloc[p1]
                if kind == "low":
                    W, Y = ext(L, p1, np.min), ext(L, p2, np.min)
                    ok = (r1 - r2) >= p.min_rsi_diff and Y > W * (1 + p.min_price_diff / 100)
                else:
                    W, Y = ext(H, p1, np.max), ext(H, p2, np.max)
                    ok = (r2 - r1) >= p.min_rsi_diff and Y < W * (1 - p.min_price_diff / 100)
                if ok and (best is None or abs(r2 - r1) > best[3]):
                    best = (p1, W, Y, abs(r2 - r1), r1)
            if best is None:
                continue
            p1, W, Y, _, r1 = best

            # still valid at confirmation
            between = C[p2 + 1: c + 1]
            if kind == "low":
                if len(between) and between.min() <= Y:
                    continue
                X = H[p1: p2 + 1].max()
                target = Y + (X - W)
                side = "Long"
            else:
                if len(between) and between.max() >= Y:
                    continue
                X = L[p1: p2 + 1].min()
                target = Y - (W - X)
                side = "Short"

            regime = rsi_regime(rsi.iloc[: c + 1], p.regime_lookback)
            with_trend = C[c] > ema[c] if side == "Long" else C[c] < ema[c]
            aligned = regime.startswith("Bull") if side == "Long" else regime.startswith("Bear")
            if p.trend_filter and not with_trend:
                continue
            if p.regime_filter and not aligned:
                continue

            sigs.append(dict(side=side, p1=p1, p2=p2, conf=c, W=float(W), X=float(X),
                             Y=float(Y), target=float(target), rsi1=float(r1), rsi2=float(r2),
                             regime=regime, aligned=aligned, with_trend=bool(with_trend)))
    sigs.sort(key=lambda s: s["conf"])
    return sigs


# ----------------------------------------------------------------------------
# Trade simulation
# ----------------------------------------------------------------------------
def simulate(df: pd.DataFrame, sigs: list, p: BTParams, symbol: str = "") -> list:
    n = len(df)
    O, H, L, C = (df[k].values for k in ("Open", "High", "Low", "Close"))
    idx = df.index
    trades, busy_until = [], -1

    for s in sigs:
        e = s["conf"] + 1
        if s["conf"] <= busy_until or e >= n:
            continue
        long = s["side"] == "Long"
        entry, stop, tgt = O[e], s["Y"], s["target"]

        # setup already broken / target already reached at the open -> skip
        if long and (entry <= stop or entry >= tgt):
            continue
        if not long and (entry >= stop or entry <= tgt):
            continue
        risk = abs(entry - stop)
        rr = abs(tgt - entry) / risk if risk > 0 else np.nan
        if p.min_reward_risk and (np.isnan(rr) or rr < p.min_reward_risk):
            continue

        last = min(e + p.max_hold - 1, n - 1)
        exit_i, exit_px, reason = last, C[last], "Time"
        for k in range(e, last + 1):
            # stop first (conservative)
            if long:
                if p.stop_mode == "intrabar" and L[k] <= stop:
                    exit_i, exit_px, reason = k, min(O[k], stop) if k > e else stop, "Stop"
                    break
                if p.stop_mode == "close" and C[k] < stop:
                    exit_i, exit_px, reason = k, C[k], "Stop"
                    break
                if H[k] >= tgt:
                    exit_i, exit_px, reason = k, max(O[k], tgt) if k > e else tgt, "Target"
                    break
            else:
                if p.stop_mode == "intrabar" and H[k] >= stop:
                    exit_i, exit_px, reason = k, max(O[k], stop) if k > e else stop, "Stop"
                    break
                if p.stop_mode == "close" and C[k] > stop:
                    exit_i, exit_px, reason = k, C[k], "Stop"
                    break
                if L[k] <= tgt:
                    exit_i, exit_px, reason = k, min(O[k], tgt) if k > e else tgt, "Target"
                    break
        else:
            if last == n - 1 and (last - e + 1) < p.max_hold:
                reason = "Open"

        gross = (exit_px / entry - 1) * (1 if long else -1)
        ret = gross - p.cost_pct / 100
        trades.append(dict(
            Symbol=symbol, Side=s["side"],
            **{"Pivot1 Date": idx[s["p1"]].date(), "Pivot2 Date": idx[s["p2"]].date(),
               "Signal Date": idx[s["conf"]].date(), "Entry Date": idx[e].date(),
               "Exit Date": idx[exit_i].date()},
            Entry=round(float(entry), 2), Stop=round(stop, 2), Target=round(tgt, 2),
            Exit=round(float(exit_px), 2), **{"Exit Reason": reason},
            **{"Return %": round(ret * 100, 2), "R Multiple": round(gross * entry / risk, 2) if risk else np.nan,
               "Reward:Risk": round(rr, 2) if not np.isnan(rr) else np.nan,
               "Bars Held": int(exit_i - e + 1), "RSI P1": round(s["rsi1"], 1),
               "RSI P2": round(s["rsi2"], 1), "Regime": s["regime"],
               "Regime Aligned": "Yes" if s["aligned"] else "No",
               "With Trend": "Yes" if s["with_trend"] else "No"},
            _p1=s["p1"], _p2=s["p2"], _e=e, _x=exit_i, _W=s["W"], _X=s["X"],
        ))
        busy_until = exit_i
    return trades


def run_backtest(data: dict, p: BTParams) -> pd.DataFrame:
    rows = []
    for sym, df in data.items():
        if len(df) < max(p.rsi, p.ema) + 20:
            continue
        rows += simulate(df, generate_signals(df, p), p, sym)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Entry Date").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def summarize(trades: pd.DataFrame) -> dict:
    t = trades[trades["Exit Reason"] != "Open"] if not trades.empty else trades
    if t.empty:
        return {}
    r = t["Return %"]
    wins, losses = r[r > 0], r[r <= 0]
    eq = equity_curve(t)
    return {
        "Trades": len(t),
        "Win rate %": round(100 * len(wins) / len(t), 1),
        "Avg return %": round(r.mean(), 2),
        "Median return %": round(r.median(), 2),
        "Avg win %": round(wins.mean(), 2) if len(wins) else 0.0,
        "Avg loss %": round(losses.mean(), 2) if len(losses) else 0.0,
        "Profit factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else np.inf,
        "Expectancy (R)": round(t["R Multiple"].mean(), 2),
        "Target hit %": round(100 * (t["Exit Reason"] == "Target").mean(), 1),
        "Stopped %": round(100 * (t["Exit Reason"] == "Stop").mean(), 1),
        "Timed out %": round(100 * (t["Exit Reason"] == "Time").mean(), 1),
        "Avg bars held": round(t["Bars Held"].mean(), 1),
        "Sum of returns %": round(r.sum(), 1),
        "Max drawdown (pts)": round(float((eq["Cumulative %"].cummax() - eq["Cumulative %"]).max()), 1),
        "Open trades": int((trades["Exit Reason"] == "Open").sum()),
    }


def equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """Cumulative sum of per-trade % returns by exit date (equal size per trade)."""
    t = trades[trades["Exit Reason"] != "Open"].sort_values("Exit Date")
    out = t.groupby("Exit Date")["Return %"].sum().cumsum().rename("Cumulative %").reset_index()
    return out


def breakdown(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    t = trades[trades["Exit Reason"] != "Open"]
    if t.empty:
        return pd.DataFrame()
    g = t.groupby(by)["Return %"]
    out = pd.DataFrame({
        "Trades": g.size(),
        "Win rate %": g.apply(lambda x: round(100 * (x > 0).mean(), 1)),
        "Avg return %": g.mean().round(2),
        "Sum %": g.sum().round(1),
        "Target hit %": t.groupby(by)["Exit Reason"].apply(lambda x: round(100 * (x == "Target").mean(), 1)),
    })
    return out.reset_index()


def params_dict(p: BTParams) -> dict:
    return asdict(p)
