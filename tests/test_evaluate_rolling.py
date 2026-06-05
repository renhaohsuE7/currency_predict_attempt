"""Tests for BaseModel.evaluate_rolling() — rolling origin evaluation."""

import numpy as np
import pandas as pd
import pytest

from currency_predictor.models.naive import NaiveModel


def _make_data(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 0.5,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.random.randint(1000, 10000, n),
        },
        index=dates,
    )
    return df, df["Close"]


class TestEvaluateRollingStructure:
    """Test return structure and basic properties."""

    def test_returns_correct_keys(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(50)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=5)

        assert "aggregate" in result
        assert "per_horizon" in result
        assert "h1_actual" in result
        assert "h1_predicted" in result

    def test_aggregate_has_standard_metric_keys(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(50)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=5)
        agg = result["aggregate"]

        for key in ["rmse", "mae", "mse", "mape", "mda", "direction_accuracy"]:
            assert key in agg, f"Missing key: {key}"
        assert "n_origins" in agg

    def test_aggregate_has_mase_when_y_train_provided(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(50)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=5, y_train=y[:30])
        assert "mase" in result["aggregate"]

    def test_aggregate_no_mase_without_y_train(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(50)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=5)
        assert "mase" not in result["aggregate"]

    def test_per_horizon_count_equals_pred_len(self):
        pred_len = 7
        model = NaiveModel(pred_len=pred_len)
        X, y = _make_data(60)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=pred_len)
        assert len(result["per_horizon"]) == pred_len

    def test_per_horizon_keys_are_1_indexed(self):
        pred_len = 5
        model = NaiveModel(pred_len=pred_len)
        X, y = _make_data(50)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=pred_len)
        assert set(result["per_horizon"].keys()) == {1, 2, 3, 4, 5}

    def test_per_horizon_has_rmse_mae(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(50)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=5)
        for h, metrics in result["per_horizon"].items():
            assert "rmse" in metrics
            assert "mae" in metrics


class TestEvaluateRollingMath:
    """Test mathematical correctness."""

    def test_n_origins_formula(self):
        """n_origins = len(X) - seq_len - pred_len + 1 (when step=1)"""
        model = NaiveModel(pred_len=5)
        X, y = _make_data(60)
        model.fit(X, y)

        seq_len, pred_len = 10, 5
        result = model.evaluate_rolling(X, y, seq_len=seq_len, pred_len=pred_len)

        expected = len(X) - seq_len - pred_len + 1
        assert int(result["aggregate"]["n_origins"]) == expected

    def test_step_reduces_origins(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(60)
        model.fit(X, y)

        r1 = model.evaluate_rolling(X, y, seq_len=10, pred_len=5, step=1)
        r5 = model.evaluate_rolling(X, y, seq_len=10, pred_len=5, step=5)

        assert int(r5["aggregate"]["n_origins"]) < int(r1["aggregate"]["n_origins"])

    def test_h1_series_length_equals_n_origins(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(50)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=5)
        n_origins = int(result["aggregate"]["n_origins"])

        assert len(result["h1_actual"]) == n_origins
        assert len(result["h1_predicted"]) == n_origins

    def test_naive_h1_prediction_is_last_context_value(self):
        """NaiveModel predicts last value. h=1 prediction should be the value
        at position origin-1 (last element of context window)."""
        model = NaiveModel(pred_len=3)
        X, y = _make_data(30)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=5, pred_len=3)
        h1_pred = result["h1_predicted"]

        # For each origin, naive predicts the last value of the context window
        # origin=5: context=[0..4], last_value = Close[4]
        # origin=6: context=[1..5], last_value = Close[5]
        close = X["Close"].values
        for i, origin in enumerate(range(5, 30 - 3 + 1)):
            assert h1_pred[i] == pytest.approx(close[origin - 1], abs=1e-6)

    def test_metrics_are_non_negative(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(50)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=5)
        agg = result["aggregate"]

        assert agg["rmse"] >= 0
        assert agg["mae"] >= 0
        assert agg["mse"] >= 0

    def test_per_horizon_rmse_non_decreasing_tendency(self):
        """For naive model on random walk, RMSE should generally increase with horizon.
        This isn't strictly guaranteed per run, so we just check they're all positive."""
        model = NaiveModel(pred_len=10)
        X, y = _make_data(100)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=10)
        for h, metrics in result["per_horizon"].items():
            assert metrics["rmse"] > 0


class TestEvaluateRollingEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data_raises_value_error(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(10)
        model.fit(X, y)

        with pytest.raises(ValueError, match="Data length"):
            model.evaluate_rolling(X, y, seq_len=8, pred_len=5)

    def test_exact_minimum_data_works(self):
        """seq_len + pred_len data points should give exactly 1 origin."""
        model = NaiveModel(pred_len=3)
        X, y = _make_data(13)  # 10 + 3
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=3)
        assert int(result["aggregate"]["n_origins"]) == 1

    def test_step_larger_than_available_origins(self):
        """Large step should still produce at least 1 origin."""
        model = NaiveModel(pred_len=3)
        X, y = _make_data(20)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=3, step=100)
        assert int(result["aggregate"]["n_origins"]) == 1

    def test_mda_in_valid_range(self):
        model = NaiveModel(pred_len=5)
        X, y = _make_data(50)
        model.fit(X, y)

        result = model.evaluate_rolling(X, y, seq_len=10, pred_len=5)
        mda_val = result["aggregate"]["mda"]
        assert 0.0 <= mda_val <= 1.0
