"""Tests for the log-return target transform (Part D)."""

import numpy as np
import pandas as pd
import pytest

from currency_predictor.data_processor import DataProcessor
from currency_predictor.config.settings import ModelTrainingConfig


class TestLogReturnHelpers:
    """to_log_returns / from_log_returns pure helpers."""

    def test_to_log_returns_first_is_nan(self):
        close = pd.Series([100.0, 110.0, 121.0])
        r = DataProcessor.to_log_returns(close)
        assert np.isnan(r.iloc[0])
        # ln(110/100), ln(121/110)
        assert r.iloc[1] == pytest.approx(np.log(1.1))
        assert r.iloc[2] == pytest.approx(np.log(1.1))

    def test_from_log_returns_formula(self):
        last_price = 100.0
        returns = np.array([np.log(1.1), np.log(1.1)])
        prices = DataProcessor.from_log_returns(last_price, returns)
        assert prices == pytest.approx([110.0, 121.0])

    def test_round_trip_reconstructs_prices(self):
        prices = pd.Series([50.0, 52.0, 48.0, 49.5, 55.0])
        returns = DataProcessor.to_log_returns(prices).iloc[1:].to_numpy()
        reconstructed = DataProcessor.from_log_returns(prices.iloc[0], returns)
        assert reconstructed == pytest.approx(prices.iloc[1:].to_numpy())

    def test_from_log_returns_length_matches(self):
        out = DataProcessor.from_log_returns(10.0, np.zeros(7))
        assert len(out) == 7
        # zero returns → flat at last price
        assert np.all(out == pytest.approx(10.0))


class TestTargetTransformConfig:
    """ModelTrainingConfig.target_transform validation."""

    def test_default_is_price(self):
        cfg = ModelTrainingConfig()
        assert cfg.target_transform == "price"

    def test_accepts_log_return(self):
        cfg = ModelTrainingConfig(target_transform="log_return")
        assert cfg.target_transform == "log_return"

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            ModelTrainingConfig(target_transform="returns")
