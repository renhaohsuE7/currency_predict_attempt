"""Tests for BacktestRunner with mock model."""

from unittest.mock import MagicMock, patch
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from currency_predictor.backtesting.runner import BacktestRunner
from currency_predictor.backtesting.result import BacktestResult, FoldResult
from currency_predictor.backtesting.splitter import FoldSpec


def _make_processed_df(n_rows: int = 300) -> pd.DataFrame:
    """Create a synthetic processed DataFrame for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n_rows)
    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close - np.random.rand(n_rows),
            "High": close + np.random.rand(n_rows),
            "Low": close - np.random.rand(n_rows) - 0.5,
            "Close": close,
            "Volume": np.random.randint(1000, 10000, n_rows),
            "SMA_5": close,
            "SMA_10": close,
            "Returns": np.concatenate([[0], np.diff(close) / close[:-1]]),
            "Close_lag_1": np.concatenate([[close[0]], close[:-1]]),
        },
        index=dates,
    )
    return df


def _make_mock_model():
    """Create a mock model that returns predictions."""
    model = MagicMock()
    model.is_fitted = False

    def fit_side_effect(X, y, **kwargs):
        model.is_fitted = True
        return model

    model.fit.side_effect = fit_side_effect

    def predict_side_effect(X, horizon=None):
        n = horizon or len(X)
        return np.random.randn(n) * 0.5 + 100

    model.predict.side_effect = predict_side_effect
    return model


class TestBacktestRunnerFoldExecution:
    """Test single fold execution logic."""

    def test_run_fold_returns_fold_result(self):
        """_run_fold should return a FoldResult."""
        runner = BacktestRunner(config={"model_params": {}})
        processed = _make_processed_df(100)
        fold = FoldSpec(fold_index=0, train_start=0, train_end=70, test_start=70, test_end=100)

        with patch(
            "currency_predictor.backtesting.runner.ModelFactory"
        ) as mock_factory:
            mock_factory.create_model.return_value = _make_mock_model()

            result = runner._run_fold(
                model_name="patchtst_sklearn",
                model_params={},
                processed_data=processed,
                fold=fold,
                target_column="Close",
            )

        assert isinstance(result, FoldResult)
        assert result.fold_index == 0
        assert result.train_size == 70
        assert "rmse" in result.prediction_metrics
        assert "sharpe_ratio" in result.financial_metrics

    def test_fresh_model_per_fold(self):
        """Each fold should create a new model via ModelFactory."""
        runner = BacktestRunner(config={"model_params": {}})
        processed = _make_processed_df(200)
        folds = [
            FoldSpec(0, 0, 100, 100, 130),
            FoldSpec(1, 30, 130, 130, 160),
        ]

        with patch(
            "currency_predictor.backtesting.runner.ModelFactory"
        ) as mock_factory:
            mock_factory.create_model.return_value = _make_mock_model()

            for fold in folds:
                runner._run_fold("patchtst_sklearn", {}, processed, fold, "Close")

            assert mock_factory.create_model.call_count == 2


class TestBacktestRunnerSingleBacktest:
    """Test _run_single_backtest aggregation."""

    def test_result_structure(self):
        """Should produce BacktestResult with correct fields."""
        runner = BacktestRunner(config={"model_params": {}})
        processed = _make_processed_df(200)
        folds = [
            FoldSpec(0, 0, 100, 100, 120),
            FoldSpec(1, 20, 120, 120, 140),
            FoldSpec(2, 40, 140, 140, 160),
        ]

        with patch(
            "currency_predictor.backtesting.runner.ModelFactory"
        ) as mock_factory:
            mock_factory.create_model.return_value = _make_mock_model()

            result = runner._run_single_backtest(
                symbol="USDTWD=X",
                model_name="patchtst_sklearn",
                processed_data=processed,
                folds=folds,
                strategy="rolling",
            )

        assert isinstance(result, BacktestResult)
        assert result.symbol == "USDTWD=X"
        assert result.model_name == "patchtst_sklearn"
        assert result.n_folds == 3
        assert len(result.fold_results) == 3
        assert "rmse" in result.avg_prediction_metrics
        assert "sharpe_ratio" in result.avg_financial_metrics
        assert len(result.overall_equity_curve) > 0

    def test_fold_count_matches(self):
        """Number of fold results should match number of folds."""
        runner = BacktestRunner(config={"model_params": {}})
        processed = _make_processed_df(300)
        folds = [FoldSpec(i, i * 30, i * 30 + 100, i * 30 + 100, i * 30 + 120) for i in range(5)]

        with patch(
            "currency_predictor.backtesting.runner.ModelFactory"
        ) as mock_factory:
            mock_factory.create_model.return_value = _make_mock_model()

            result = runner._run_single_backtest(
                "TEST", "patchtst_sklearn", processed, folds, "rolling"
            )

        assert result.n_folds == 5


class TestChainEquityCurves:
    """Test equity curve chaining."""

    def test_chain_single_fold(self):
        """Single fold should return its equity curve scaled to 10000."""
        fold = MagicMock()
        fold.equity_curve = np.array([1000, 1100, 1050, 1150])
        result = BacktestRunner._chain_equity_curves([fold])
        assert result[0] == pytest.approx(10000)
        assert len(result) == 4

    def test_chain_continuity(self):
        """Second fold should start where first ended."""
        f1 = MagicMock()
        f1.equity_curve = np.array([10000, 11000])
        f2 = MagicMock()
        f2.equity_curve = np.array([10000, 10500])

        result = BacktestRunner._chain_equity_curves([f1, f2])
        # f1 ends at 11000, f2 should start at 11000
        assert result[2] == pytest.approx(11000)  # start of f2
        assert len(result) == 4

    def test_chain_empty(self):
        result = BacktestRunner._chain_equity_curves([])
        assert len(result) == 1
        assert result[0] == 10000.0


class TestBacktestRunnerRollingEvaluation:
    """Test rolling evaluation integration in _run_fold."""

    def test_rolling_fold_has_per_horizon_metrics(self):
        """When test set is large enough, _run_fold should use rolling evaluation."""
        from currency_predictor.models.naive import NaiveModel

        runner = BacktestRunner(config={"model_params": {}})
        processed = _make_processed_df(200)
        # Large test window to trigger rolling path
        fold = FoldSpec(fold_index=0, train_start=0, train_end=100, test_start=100, test_end=200)

        with patch(
            "currency_predictor.backtesting.runner.ModelFactory"
        ) as mock_factory:
            mock_factory.create_model.return_value = NaiveModel(pred_len=5)

            result = runner._run_fold(
                model_name="naive",
                model_params={"seq_len": 10, "pred_len": 5},
                processed_data=processed,
                fold=fold,
                target_column="Close",
            )

        assert isinstance(result, FoldResult)
        assert result.per_horizon_metrics, "per_horizon_metrics should be populated"
        assert result.n_origins > 0

    def test_rolling_fold_per_horizon_keys(self):
        """Per-horizon metrics should have 1..pred_len keys."""
        from currency_predictor.models.naive import NaiveModel

        runner = BacktestRunner(config={"model_params": {}})
        processed = _make_processed_df(150)
        fold = FoldSpec(fold_index=0, train_start=0, train_end=80, test_start=80, test_end=150)

        pred_len = 7
        with patch(
            "currency_predictor.backtesting.runner.ModelFactory"
        ) as mock_factory:
            mock_factory.create_model.return_value = NaiveModel(pred_len=pred_len)

            result = runner._run_fold(
                model_name="naive",
                model_params={"seq_len": 10, "pred_len": pred_len},
                processed_data=processed,
                fold=fold,
                target_column="Close",
            )

        assert set(result.per_horizon_metrics.keys()) == set(range(1, pred_len + 1))

    def test_fallback_to_single_shot_when_test_too_small(self):
        """When test window < seq_len + pred_len, should fallback."""
        runner = BacktestRunner(config={"model_params": {}})
        processed = _make_processed_df(100)
        # Small test window: 10 rows < seq_len(64) + pred_len(15) = 79
        fold = FoldSpec(fold_index=0, train_start=0, train_end=90, test_start=90, test_end=100)

        with patch(
            "currency_predictor.backtesting.runner.ModelFactory"
        ) as mock_factory:
            mock_factory.create_model.return_value = _make_mock_model()

            result = runner._run_fold(
                model_name="patchtst_sklearn",
                model_params={},
                processed_data=processed,
                fold=fold,
                target_column="Close",
            )

        assert result.per_horizon_metrics == {}
        assert result.n_origins == 0

    def test_aggregate_per_horizon_across_folds(self):
        """BacktestResult should aggregate per-horizon metrics across folds."""
        from currency_predictor.models.naive import NaiveModel

        pred_len = 5
        runner = BacktestRunner(
            config={"model_params": {"seq_len": 10, "pred_len": pred_len}}
        )
        processed = _make_processed_df(300)
        folds = [
            FoldSpec(0, 0, 100, 100, 200),
            FoldSpec(1, 50, 150, 150, 250),
        ]

        with patch(
            "currency_predictor.backtesting.runner.ModelFactory"
        ) as mock_factory:
            mock_factory.create_model.return_value = NaiveModel(pred_len=pred_len)

            result = runner._run_single_backtest(
                symbol="TEST",
                model_name="naive",
                processed_data=processed,
                folds=folds,
                strategy="rolling",
            )

        assert result.avg_per_horizon_metrics, "Should have aggregated per-horizon metrics"
        assert len(result.avg_per_horizon_metrics) == pred_len
