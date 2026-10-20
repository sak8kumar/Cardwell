"""
Streamlit front end — Cardwell RSI reversal screener & backtester
Run:  streamlit run app.py
"""
import io
import types
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import cardwell_screener as cs
from backtest import BTParams, breakdown, equity_curve, run_backtest, summarize

st.set_page_config(page_title="Cardwell Reversal Lab", page_icon="📈", layout="wide")

SAMPLE = ("RELIANCE HDFCBANK ICICIBANK INFY TCS SBIN AXISBANK KOTAKBANK LT ITC "
          "HINDUNILVR BHARTIARTL BAJFINANCE MARUTI SUNPHARMA TATASTEEL TITAN ULTRACEMCO "
          "ASIANPAINT HCLTECH NTPC POWERGRID ONGC COALINDIA JSWSTEEL")

# ----------------------------------------------------------------------------
# Data (cached)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_universe_nse():
    return cs.load_universe(None)


@st.cache_data(ttl=3600, show_spinner=False)
def get_data(symbols: tuple, period: str, interval: str, fallback_ns: bool):
    data = cs.download(list(symbols), period, interval, ".BO")
    missing = [s for s in symbols if s not in data]
    if missing and fallback_ns:
        data.update(cs.download(missing, period, interval, ".NS"))
        missing = [s for s in symbols if s not in data]
    return data, missing


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("📈 Cardwell Reversal Lab")
st.sidebar.caption("F&O universe · BSE prices (Yahoo .BO)")

src = st.sidebar.radio("Universe", ["Sample (25 large caps)", "Full NSE F&O list", "Custom"])
if src == "Custom":
    up = st.sidebar.file_uploader("Symbols file: one per line, or NSE's fo_mktlots.csv", type=["txt", "csv"])
    txt = st.sidebar.text_area("…or paste symbols", "RELIANCE, TATASTEEL, SBIN")
    raw = up.read().decode("utf-8", "ignore") if up else txt
    first = raw.splitlines()[0].upper() if raw.strip() else ""
    if "SYMBOL" in first:  # NSE fo_mktlots.csv format
        df_fo = pd.read_csv(io.StringIO(raw))
        df_fo.columns = [c.strip().upper() for c in df_fo.columns]
        cand = df_fo["SYMBOL"].astype(str).str.strip().str.upper()
    else:
        cand = [s.strip().upper() for s in raw.replace(",", "\n").split()]
    symbols = sorted({s for s in cand if s and s != "SYMBOL" and s not in cs.INDEX_SYMBOLS})
elif src == "Full NSE F&O list":
    try:
        symbols = get_universe_nse()
    except SystemExit as e:
        st.sidebar.error("NSE blocked the automatic download. Open nseindia.com/content/fo/fo_mktlots.csv "
                         "in your browser, save it, then choose Universe → Custom and upload it. "
                         "Using the 25-stock sample for now.")
        symbols = SAMPLE.split()
else:
    symbols = SAMPLE.split()
st.sidebar.write(f"**{len(symbols)}** symbols")

c1, c2 = st.sidebar.columns(2)
interval = c1.selectbox("Timeframe", ["1d", "1wk"], format_func={"1d": "Daily", "1wk": "Weekly"}.get)
period = c2.selectbox("History", ["2y", "5y", "10y", "max"], index=1)
fallback_ns = st.sidebar.checkbox("Use NSE price if BSE ticker fails", True)

with st.sidebar.expander("Signal rules", expanded=False):
    rsi_len = st.number_input("RSI length", 2, 50, 14)
    pl = st.number_input("Pivot left bars", 1, 20, 5)
    pr = st.number_input("Pivot right bars (confirmation lag)", 1, 10, 2)
    gmin, gmax = st.slider("Bars between pivots", 2, 150, (5, 60))
    min_rsi = st.number_input("Min RSI difference (pts)", 0.0, 20.0, 2.0, 0.5)
    min_px = st.number_input("Min price difference (%)", 0.0, 10.0, 0.5, 0.1)
    ema_len = st.number_input("Trend EMA", 5, 300, 50)
    reg_lb = st.number_input("Regime lookback (bars)", 20, 300, 60)

with st.sidebar.expander("Trade rules (backtest)", expanded=True):
    direction = st.selectbox("Direction", ["both", "long", "short"],
                             format_func={"both": "Long + Short", "long": "Long only (positive reversals)",
                                          "short": "Short only (negative reversals)"}.get)
    stop_mode = st.selectbox("Stop at Y triggers on", ["close", "intrabar"],
                             format_func={"close": "Close beyond Y", "intrabar": "Touch of Y"}.get)
    max_hold = st.number_input("Max holding bars", 1, 250, 40)
    cost = st.number_input("Round-trip cost (%)", 0.0, 2.0, 0.10, 0.05)
    min_rr = st.number_input("Min reward:risk at entry (0 = off)", 0.0, 10.0, 0.0, 0.25)
    trend_f = st.checkbox("Trend filter (long > EMA, short < EMA)")
    regime_f = st.checkbox("Regime filter (long in bull range, short in bear range)")

params = BTParams(rsi=rsi_len, pivot_left=pl, pivot_right=pr, price_window=2, min_gap=gmin,
                  max_gap=gmax, min_rsi_diff=min_rsi, min_price_diff=min_px, ema=ema_len,
                  regime_lookback=reg_lb, trend_filter=trend_f, regime_filter=regime_f,
                  direction=direction, stop_mode=stop_mode, max_hold=max_hold,
                  cost_pct=cost, min_reward_risk=min_rr)


def load():
    with st.spinner(f"Downloading {len(symbols)} symbols ({interval}, {period})…"):
        data, missing = get_data(tuple(symbols), period, interval, fallback_ns)
    if missing:
        st.warning(f"{len(missing)} not loaded: {', '.join(missing[:30])}{'…' if len(missing) > 30 else ''}")
    if not data:
        st.error("No price data loaded. Check symbols or internet access.")
        st.stop()
    return data


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def trade_chart(df, tr, rsi_len, pad=40):
    lo = max(0, tr["_p1"] - pad)
    hi = min(len(df), tr["_x"] + pad // 2)
    d = df.iloc[lo:hi]
    rsi = cs.rsi_wilder(df["Close"], rsi_len).iloc[lo:hi]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close,
                                 name="Price", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(width=1.5)), 2, 1)
    long = tr["Side"] == "Long"
    ix = df.index
    pts = [(tr["_p1"], tr["_W"], "W"), (tr["_p2"], tr["Stop"], "Y")]
    fig.add_trace(go.Scatter(x=[ix[i] for i, _, _ in pts], y=[v for _, v, _ in pts], mode="markers+text+lines",
                             text=[t for *_, t in pts], textposition="bottom center" if long else "top center",
                             marker=dict(size=9), line=dict(dash="dot"), name="Price pivots"), 1, 1)
    full_rsi = cs.rsi_wilder(df["Close"], rsi_len)
    fig.add_trace(go.Scatter(x=[ix[tr["_p1"]], ix[tr["_p2"]]],
                             y=[full_rsi.iloc[tr["_p1"]], full_rsi.iloc[tr["_p2"]]],
                             mode="markers+lines", line=dict(dash="dot"), name="RSI pivots"), 2, 1)
    fig.add_trace(go.Scatter(x=[ix[tr["_e"]]], y=[tr["Entry"]], mode="markers", name="Entry",
                             marker=dict(symbol="triangle-up" if long else "triangle-down", size=14)), 1, 1)
    fig.add_trace(go.Scatter(x=[ix[tr["_x"]]], y=[tr["Exit"]], mode="markers", name=f"Exit ({tr['Exit Reason']})",
                             marker=dict(symbol="x", size=12)), 1, 1)
    x0, x1 = ix[tr["_e"]], ix[tr["_x"]]
    fig.add_shape(type="line", x0=x0, x1=x1, y0=tr["Target"], y1=tr["Target"], line=dict(color="green", dash="dash"), row=1, col=1)
    fig.add_shape(type="line", x0=x0, x1=x1, y0=tr["Stop"], y1=tr["Stop"], line=dict(color="red", dash="dash"), row=1, col=1)
    for lvl in (40, 60, 80, 20):
        fig.add_hline(y=lvl, line=dict(width=0.6, dash="dot", color="grey"), row=2, col=1)
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.04))
    return fig


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_bt, tab_scr, tab_about = st.tabs(["🧪 Backtest", "🔎 Today's screener", "ℹ️ Method"])

with tab_bt:
    st.subheader("Backtest")
    if st.button("Run backtest", type="primary"):
        data = load()
        with st.spinner("Simulating trades…"):
            trades = run_backtest(data, params)
        st.session_state["bt"] = (trades, data, params)

    if "bt" in st.session_state:
        trades, data, p_used = st.session_state["bt"]
        if trades.empty:
            st.info("No trades with these settings. Loosen filters or extend history.")
        else:
            s = summarize(trades)
            k = st.columns(6)
            for col, key in zip(k, ["Trades", "Win rate %", "Avg return %", "Profit factor",
                                    "Expectancy (R)", "Max drawdown (pts)"]):
                col.metric(key, s.get(key))
            k2 = st.columns(6)
            for col, key in zip(k2, ["Target hit %", "Stopped %", "Timed out %", "Avg win %",
                                     "Avg loss %", "Avg bars held"]):
                col.metric(key, s.get(key))

            left, right = st.columns([3, 2])
            eq = equity_curve(trades)
            fig = go.Figure(go.Scatter(x=eq["Exit Date"], y=eq["Cumulative %"], fill="tozeroy", name="Equity"))
            fig.update_layout(title="Cumulative return (sum of trade %, equal size per trade)",
                              height=360, margin=dict(l=10, r=10, t=40, b=10))
            left.plotly_chart(fig, width="stretch")
            closed = trades[trades["Exit Reason"] != "Open"]
            h = go.Figure(go.Histogram(x=closed["Return %"], nbinsx=40))
            h.update_layout(title="Trade return distribution (%)", height=360, margin=dict(l=10, r=10, t=40, b=10))
            right.plotly_chart(h, width="stretch")

            st.markdown("##### Breakdowns")
            b = st.columns(4)
            yr = closed.assign(Year=pd.to_datetime(closed["Exit Date"]).dt.year)
            b[0].caption("By side"); b[0].dataframe(breakdown(trades, "Side"), hide_index=True)
            b[1].caption("Regime aligned"); b[1].dataframe(breakdown(trades, "Regime Aligned"), hide_index=True)
            b[2].caption("With EMA trend"); b[2].dataframe(breakdown(trades, "With Trend"), hide_index=True)
            b[3].caption("By exit reason"); b[3].dataframe(breakdown(trades, "Exit Reason"), hide_index=True)
            c1, c2 = st.columns(2)
            c1.caption("By year"); c1.dataframe(breakdown(yr, "Year"), hide_index=True, width="stretch")
            c2.caption("By symbol (sorted by sum %)")
            c2.dataframe(breakdown(trades, "Symbol").sort_values("Sum %", ascending=False),
                         hide_index=True, width="stretch", height=300)

            st.markdown("##### Trade log")
            show = trades[[c for c in trades.columns if not c.startswith("_")]]
            st.dataframe(show, hide_index=True, width="stretch", height=320)
            st.download_button("Download trade log (CSV)", show.to_csv(index=False).encode(),
                               f"cardwell_backtest_{interval}_{datetime.now():%Y%m%d}.csv", "text/csv")

            st.markdown("##### Inspect a trade")
            labels = [f"{r['Symbol']} · {r['Side']} · {r['Entry Date']} → {r['Exit Date']} · "
                      f"{r['Exit Reason']} · {r['Return %']:+.2f}%" for _, r in trades.iterrows()]
            pick = st.selectbox("Trade", range(len(trades)), format_func=lambda i: labels[i],
                                index=len(trades) - 1)
            tr = trades.iloc[pick]
            st.plotly_chart(trade_chart(data[tr["Symbol"]], tr, p_used.rsi), width="stretch")

with tab_scr:
    st.subheader("Live signals (most recent bars)")
    recent = st.slider("Second pivot within last N bars", 1, 30, 10)
    only_valid = st.checkbox("Only signals still valid (price not beyond Y)", True)
    if st.button("Scan now", type="primary"):
        data = load()
        a = types.SimpleNamespace(rsi=params.rsi, pivot_left=params.pivot_left, pivot_right=params.pivot_right,
                                  price_window=params.price_window, min_gap=params.min_gap, max_gap=params.max_gap,
                                  min_rsi_diff=params.min_rsi_diff, min_price_diff=params.min_price_diff,
                                  recent=recent, ema=params.ema, regime_lookback=params.regime_lookback,
                                  trend_filter=params.trend_filter, only_valid=only_valid,
                                  interval=interval, period=period, fallback_ns=fallback_ns)
        res = [{"Symbol": s, **r} for s, df in data.items() for r in cs.scan(df, a)]
        if params.direction == "long":
            res = [r for r in res if r["Signal"] == "Positive Reversal"]
        elif params.direction == "short":
            res = [r for r in res if r["Signal"] == "Negative Reversal"]
        if params.regime_filter:
            res = [r for r in res if r["Regime Aligned"] == "Yes"]
        st.session_state["scan"] = (res, a)

    if "scan" in st.session_state:
        res, a = st.session_state["scan"]
        if not res:
            st.info("No current signals.")
        else:
            df = pd.DataFrame(res).sort_values(["Regime Aligned", "Bars Since P2"], ascending=[False, True])
            m = st.columns(3)
            m[0].metric("Signals", len(df))
            m[1].metric("Positive", int((df.Signal == "Positive Reversal").sum()))
            m[2].metric("Negative", int((df.Signal == "Negative Reversal").sum()))
            st.dataframe(df, hide_index=True, width="stretch")
            buf = io.BytesIO()
            cs.write_excel(res, [], buf, a)
            st.download_button("Download Excel", buf.getvalue(),
                               f"cardwell_reversals_{interval}_{datetime.now():%Y%m%d}.xlsx")

with tab_about:
    st.markdown("""
**Andrew Cardwell's reversals** (often called *reverse divergence*) are trend-**continuation** signals.

| Signal | RSI | Price | Implies | Target |
|---|---|---|---|---|
| Positive reversal | Lower low | Higher low | Uptrend continues | Y + (X − W) |
| Negative reversal | Higher high | Lower high | Downtrend continues | Y − (W − X) |

W = price at first RSI pivot, Y = price at second pivot, X = extreme between them.
**RSI range regime:** bull markets tend to hold RSI in 40–80, bear markets in 20–60.

**Backtest assumptions (no look-ahead)**
- A pivot is only usable once `pivot right` bars have closed after it; entry is the next bar's open.
- Stop = Y (close beyond, or touch — your choice). Target = Cardwell projection. Otherwise exit after max holding bars.
- If stop and target hit on the same bar, the stop is assumed. Gaps fill at the open.
- One open trade per symbol. Costs deducted per round trip. Equity curve is a simple sum of per-trade % (equal size), not a portfolio simulation.

**Limitations**
- Survivorship bias: the universe is *today's* F&O list, so stocks that were dropped or delisted are missing — results will look better than reality.
- Yahoo data can have gaps and unadjusted corporate actions on some BSE tickers.
- Short trades assume you can short via futures; F&O lot sizes and margin are not modelled.

Not investment advice.
""")
