# Cardwell Reversal Screener & Backtester

This tool screens and backtests **Andrew Cardwell's RSI reversals** (also called "reverse divergence") on the Indian F&O stock universe, using **BSE prices** from Yahoo Finance (`SYMBOL.BO`).

| Signal | RSI | Price | Cardwell's reading | Target |
|---|---|---|---|---|
| Positive reversal | Lower low | Higher low | Uptrend continuation | Y + (X − W) |
| Negative reversal | Higher high | Lower high | Downtrend continuation | Y − (W − X) |

- W = price at the first RSI pivot
- Y = price at the second RSI pivot
- X = the price extreme between the two pivots

Each signal is tagged with the RSI range regime: bull markets tend to hold RSI in 40–80, bear markets in 20–60.

> Stock F&O trades on NSE; BSE's derivatives volume is in index options. This tool therefore takes the F&O *universe* from NSE and uses *prices* from BSE.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py            # web front end: backtest + live screener
python cardwell_screener.py     # command-line screener, saves to Excel
```

## Front end (`app.py`)

- **Backtest tab:**
  - KPIs: win rate, profit factor, expectancy in R, target / stop / timeout split, drawdown
  - Equity curve and return histogram
  - Breakdowns by side, regime alignment, EMA trend, exit reason, year and symbol
  - Full trade log, downloadable as CSV
  - Candlestick + RSI chart for any trade, with pivots, target and stop marked
- **Screener tab:** live signals from the latest bars, downloadable as Excel.
- **Sidebar:** universe (sample of 25, full NSE F&O list, or custom/uploaded), daily or weekly timeframe, history length, signal rules and trade rules.

## Backtest rules: no look-ahead

- A pivot can only be used once `pivot_right` bars have closed after it. Entry is at the next bar's open.
- **Stop** is Y, triggered either by a close beyond Y or by a touch of Y (your choice). **Target** is the Cardwell projection. Otherwise the trade exits after `max_hold` bars.
- If the stop and target both trigger on the same bar, the stop is assumed. Gaps fill at the open.
- Only one open trade per symbol at a time. Costs are deducted per round trip.
- `tests/test_backtest.py` checks that signals generated from truncated history match signals generated from full history.

## Known limitations

- **Survivorship bias:** the universe is today's F&O list, so results will look better than reality.
- The equity curve is a sum of per-trade % returns at equal size per trade. It is not a capital-constrained portfolio simulation, and F&O lot sizes and margin are not modelled.
- Yahoo BSE data can have gaps. Symbols that fail to load are listed in the app. You can add ticker overrides in `YAHOO_BSE_OVERRIDES` in `cardwell_screener.py`.
- NSE sometimes blocks the automated download of the F&O list. If that happens, use a custom symbols file.

## Use on iPad / phone

1. On [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and choose **Create app → Deploy from GitHub**. Pick this repo, branch `main`, and file `app.py`.
2. Open the app link in Safari and use **Share → Add to Home Screen**.
3. If "Full NSE F&O list" fails because NSE blocks cloud servers:
   - open `https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv` in Safari and save it to Files;
   - in the app, choose **Universe → Custom** and upload that file.

## Deploy (optional)

[Streamlit Community Cloud](https://streamlit.io/cloud) can host the app for free directly from this repo, with `app.py` as the entry point.

## Tests

```bash
pytest -q tests
```

*For research only. Not investment advice.*
