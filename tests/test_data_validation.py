"""
測試資料抽樣驗證功能

測試 CurrencyPredictor._validate_cached_data() 和
YahooFinanceCollector.get_spot_check_data()
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from currency_predictor.prediction.predictor import CurrencyPredictor
from currency_predictor.data.collectors import YahooFinanceCollector


class TestValidateCachedData:
    """CurrencyPredictor._validate_cached_data() 測試"""

    @pytest.fixture
    def predictor(self):
        with patch.object(CurrencyPredictor, '_create_model', return_value=MagicMock()):
            return CurrencyPredictor(
                model_name="patchtst_sklearn",
                model_params={"seq_len": 50, "pred_len": 5},
            )

    @pytest.fixture
    def sample_data(self):
        """建立一個具有 Close 欄位的範例 DataFrame(近期日期,通過 staleness 檢查)"""
        dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=100, freq="B")
        data = pd.DataFrame({
            "Close": np.random.uniform(30, 33, size=100),
            "Open": np.random.uniform(30, 33, size=100),
            "High": np.random.uniform(31, 34, size=100),
            "Low": np.random.uniform(29, 32, size=100),
        }, index=dates)
        return data

    def test_small_data_skips_validation(self, predictor):
        """資料少於 10 筆時應直接通過"""
        small_data = pd.DataFrame(
            {"Close": [31.0, 31.5, 32.0]},
            index=pd.date_range("2025-01-01", periods=3, freq="B"),
        )
        assert predictor._validate_cached_data("USDTWD=X", small_data) is True

    def test_no_close_column_passes(self, predictor):
        """沒有 Close 欄位時應直接通過"""
        data = pd.DataFrame(
            {"Open": range(20)},
            index=pd.date_range("2025-01-01", periods=20, freq="B"),
        )
        assert predictor._validate_cached_data("USDTWD=X", data) is True

    def test_matching_data_passes(self, predictor, sample_data):
        """快取資料與下載資料匹配時應通過"""
        def fake_spot_check(symbol, date):
            # 回傳與快取完全相同的 Close 值
            d = pd.Timestamp(date).normalize()
            return pd.DataFrame(
                {"Close": [sample_data.loc[date, "Close"]]},
                index=[d],
            )

        predictor.data_collector.get_spot_check_data = fake_spot_check
        assert predictor._validate_cached_data("USDTWD=X", sample_data) is True

    def test_mismatched_data_fails(self, predictor, sample_data):
        """快取資料與下載資料不匹配時應失敗"""
        def fake_spot_check(symbol, date):
            d = pd.Timestamp(date).normalize()
            # 回傳差異超過 0.01 的值
            return pd.DataFrame(
                {"Close": [sample_data.loc[date, "Close"] + 1.0]},
                index=[d],
            )

        predictor.data_collector.get_spot_check_data = fake_spot_check
        assert predictor._validate_cached_data("USDTWD=X", sample_data) is False

    def test_network_failure_skips_sample(self, predictor, sample_data):
        """網路問題（回傳 None）時應跳過該 sample"""
        predictor.data_collector.get_spot_check_data = lambda s, d: None
        # 所有 spot check 都回 None → 驗證通過（無法驗證視為安全）
        assert predictor._validate_cached_data("USDTWD=X", sample_data) is True

    def test_exception_in_spot_check_skips(self, predictor, sample_data):
        """spot check 拋出異常時應跳過該 sample"""
        def raise_error(symbol, date):
            raise ConnectionError("network error")

        predictor.data_collector.get_spot_check_data = raise_error
        assert predictor._validate_cached_data("USDTWD=X", sample_data) is True

    def test_stale_data_invalidated_before_sampling(self, predictor):
        """過期快取(最後日期落後太多)應在抽樣前就判定失效,不需網路。"""
        old = pd.date_range(
            end=pd.Timestamp.now().normalize() - pd.Timedelta(days=60),
            periods=50, freq="B",
        )
        data = pd.DataFrame({"Close": np.linspace(30, 33, len(old))}, index=old)

        def boom(symbol, date):  # 若有呼叫網路就讓測試失敗
            raise AssertionError("staleness should short-circuit before spot-check")

        predictor.data_collector.get_spot_check_data = boom
        assert predictor._validate_cached_data("USDTWD=X", data) is False


class TestGetSpotCheckData:
    """YahooFinanceCollector.get_spot_check_data() 測試"""

    @patch("currency_predictor.data.collectors.yf.Ticker")
    def test_returns_data_for_valid_date(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [31.5], "Open": [31.3], "High": [31.8], "Low": [31.2]},
            index=pd.DatetimeIndex([datetime(2025, 6, 15)]),
        )
        mock_ticker_cls.return_value = mock_ticker

        collector = YahooFinanceCollector()
        result = collector.get_spot_check_data("USDTWD=X", datetime(2025, 6, 15))

        assert result is not None
        assert len(result) == 1
        mock_ticker.history.assert_called_once()

    @patch("currency_predictor.data.collectors.yf.Ticker")
    def test_returns_none_for_empty_data(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        collector = YahooFinanceCollector()
        result = collector.get_spot_check_data("INVALID", datetime(2025, 6, 15))

        assert result is None

    @patch("currency_predictor.data.collectors.yf.Ticker")
    def test_returns_none_on_exception(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("API error")

        collector = YahooFinanceCollector()
        result = collector.get_spot_check_data("USDTWD=X", datetime(2025, 6, 15))

        assert result is None
