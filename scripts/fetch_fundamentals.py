"""Fetch a fundamentals snapshot for TW tickers via modern yfinance -> CSV for
the parent app's `seed_2330.py --fundamentals`. Runs in this submodule env (has
a modern yfinance); the container's pinned yfinance==0.1.87 can't fetch.

CSV columns match yfinance_fundamentals: ticker, pe_ratio, eps, forward_pe,
dividend_yield, market_cap. Missing fields (ETFs lack EPS; loss-makers lack P/E)
are written empty -> the academic screener omits that component.

Usage:
  uv run python scripts/fetch_fundamentals.py --tickers 2330.TW 0050.TW 6770.TW \
      5483.TWO 2409.TW 2337.TW --out fundamentals.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

import yfinance as yf

FIELDS = ["ticker", "pe_ratio", "eps", "forward_pe", "dividend_yield", "market_cap"]


def normalize_dividend_yield(v):
    """yfinance dividendYield is inconsistent across versions: sometimes a
    fraction (0.013), sometimes a percent (1.31). Store a fraction. A real yield
    as a fraction is always < 0.5 (a 50% yield is absurd), so treat any value
    >= 0.5 as percent form and divide by 100."""
    if v is None:
        return None
    return v / 100.0 if v >= 0.5 else v


def fetch_one(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker,
        "pe_ratio": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": normalize_dividend_yield(info.get("dividendYield")),
        "market_cap": info.get("marketCap"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--out", default="fundamentals.csv")
    args = parser.parse_args()

    rows = []
    for t in args.tickers:
        try:
            row = fetch_one(t)
            rows.append(row)
            print("fetched", t, {k: row[k] for k in FIELDS[1:]})
        except Exception as e:  # noqa: BLE001 — 誠實記錄抓取失敗,不中斷其它標的
            print("FAILED", t, e)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out, "with", len(rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
