"""Tests for forecast date alignment fixes.

1. Forecast dates must be future TRADING days (no weekends).
2. Stale cached data (last date too far behind 'now') must be invalidated so
   predict() does not anchor the forecast at an old date.
"""

import numpy as np
import pandas as pd

from currency_predictor.prediction.predictor import CurrencyPredictor


class TestFutureTradingDays:
    def test_skips_weekends(self):
        # 2026-06-05 is a Friday → next trading days are Mon 06-08, Tue 06-09, ...
        last = pd.Timestamp("2026-06-05")
        dates = CurrencyPredictor._future_trading_days(last, 5)
        assert len(dates) == 5
        assert all(d.weekday() < 5 for d in dates)  # no Sat/Sun
        assert dates[0] == pd.Timestamp("2026-06-08")  # skips the weekend
        assert all(d > last for d in dates)
        assert dates == sorted(dates)

    def test_count_matches(self):
        dates = CurrencyPredictor._future_trading_days(pd.Timestamp("2026-01-01"), 15)
        assert len(dates) == 15
        assert all(d.weekday() < 5 for d in dates)


class TestStaleCacheInvalidation:
    def _predictor(self):
        return CurrencyPredictor(model_name="patchtst_sklearn", data_storage_path="/tmp")

    def test_stale_data_invalidated(self):
        """Data ending ~60 days ago must be flagged stale (return False) before
        any network spot-check."""
        idx = pd.date_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=60),
                            periods=300, freq="B")
        df = pd.DataFrame({"Close": np.linspace(100, 120, len(idx))}, index=idx)
        p = self._predictor()
        assert p._validate_cached_data("TEST", df) is False

    def test_too_few_rows_skipped(self):
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        p = self._predictor()
        assert p._validate_cached_data("TEST", df) is True  # <10 rows → skip
