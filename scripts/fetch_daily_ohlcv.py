"""Fetch multi-year daily OHLCV for TW tickers via modern yfinance -> a CSV for
the parent app's `seed_2330.py --from-csv-daily` (yfinance_1d). Longer history
than the 1y forecast data so the year-line (MA240) draws as a real line.

CSV columns: datetime, ticker, adjclose, close, high, low, open, volume.

Usage:
  uv run python scripts/fetch_daily_ohlcv.py --period 3y --out daily_ohlcv.csv \
      --tickers 2330.TW 0050.TW 6770.TW 5483.TWO 2409.TW 2337.TW
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import yfinance as yf


def fetch_one(ticker: str, period: str):
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):   # single-ticker can still be MultiIndex
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime"
    adj = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    out = pd.DataFrame({
        "datetime": pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "adjclose": adj, "close": df["Close"], "high": df["High"],
        "low": df["Low"], "open": df["Open"], "volume": df["Volume"],
    })
    return out.dropna(subset=["close"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--period", default="3y")
    p.add_argument("--out", default="daily_ohlcv.csv")
    args = p.parse_args()

    frames = []
    for t in args.tickers:
        try:
            df = fetch_one(t, args.period)
        except Exception as e:  # noqa: BLE001
            print("FAILED", t, e)
            continue
        if df is None or df.empty:
            print("EMPTY", t)
            continue
        frames.append(df)
        print("fetched", t, len(df), "rows",
              df["datetime"].iloc[0], "..", df["datetime"].iloc[-1])
    if not frames:
        print("no data")
        return 1
    pd.concat(frames).to_csv(args.out, index=False)
    print("wrote", args.out, "with", sum(len(f) for f in frames), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
