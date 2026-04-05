"""
離線端到端測試 (Layer 1)

使用已提交的 fixture CSV 測試完整 pipeline：
DataProcessor → PatchTSTSklearn.fit → predict → evaluate → Visualizer → Formatter

標記 @pytest.mark.slow — 訓練真實 sklearn 模型（約數秒）。
執行：uv run pytest -m slow -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from currency_predictor.data_processor import DataProcessor
from currency_predictor.models.patchtst.sklearn.model import PatchTSTSklearn
from currency_predictor.visualization.visualizer import CurrencyVisualizer
from currency_predictor.reporting.formatter import ResultFormatter
from currency_predictor.backtesting.runner import BacktestRunner
from currency_predictor.backtesting.result import BacktestResult
from currency_predictor.backtesting.splitter import WalkForwardSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_process(fixture_path, lags=None):
    """Load fixture CSV and run full DataProcessor pipeline."""
    if lags is None:
        lags = [1, 2, 3]
    df = pd.read_csv(fixture_path, index_col="Date", parse_dates=True)
    processor = DataProcessor()
    cleaned = processor.clean_data(df)
    with_indicators = processor.create_technical_indicators(cleaned)
    return processor.create_lagged_features(with_indicators, lags=lags)


def _split_features_target(processed, target_col="Close", test_ratio=0.2):
    """Split processed DataFrame into train/test features and target."""
    feature_cols = [c for c in processed.columns if c != target_col]
    X = processed[feature_cols]
    y = processed[target_col]
    split = int(len(X) * (1 - test_ratio))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ---------------------------------------------------------------------------
# Phase 1: Data Processing
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestOfflineE2EDataProcessing:
    """Test data processing pipeline with fixture data."""

    def test_fixture_loads_and_has_expected_shape(self, e2e_fixture_path):
        """Fixture CSV loads correctly with OHLCV columns and 250+ rows."""
        df = pd.read_csv(e2e_fixture_path, index_col="Date", parse_dates=True)
        assert len(df) >= 250
        for col in ["Open", "High", "Low", "Close"]:
            assert col in df.columns

    def test_data_processor_pipeline_preserves_rows(self, e2e_fixture_path):
        """clean → indicators → lag features preserves 90%+ rows."""
        raw = pd.read_csv(e2e_fixture_path, index_col="Date", parse_dates=True)
        processed = _load_and_process(e2e_fixture_path)
        assert len(processed) >= len(raw) * 0.90


# ---------------------------------------------------------------------------
# Phase 2: Train → Predict → Evaluate
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestOfflineE2ETrainPredict:
    """Test full train → predict → evaluate cycle with fixture data."""

    def test_sklearn_fit_predict_round_trip(self, e2e_fixture_path, fast_sklearn_params):
        """Full load → process → fit → predict produces valid output."""
        processed = _load_and_process(e2e_fixture_path)
        X_train, X_test, y_train, y_test = _split_features_target(processed)

        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X_train, y_train)
        assert model.is_fitted is True

        preds = model.predict(X_test)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == fast_sklearn_params["pred_len"]
        assert np.all(np.isfinite(preds))

    def test_predictions_are_reasonable_range(self, e2e_fixture_path, fast_sklearn_params):
        """Predictions within 50% of the last known Close value."""
        processed = _load_and_process(e2e_fixture_path)
        X_train, X_test, y_train, y_test = _split_features_target(processed)

        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        last_close = y_test.iloc[-1]
        for p in preds:
            assert abs(p - last_close) / abs(last_close) < 0.5

    def test_evaluate_returns_valid_metrics(self, e2e_fixture_path, fast_sklearn_params):
        """model.evaluate() returns non-negative RMSE, MAE, MSE."""
        processed = _load_and_process(e2e_fixture_path)
        X_train, X_test, y_train, y_test = _split_features_target(processed)

        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["mse"] >= 0


# ---------------------------------------------------------------------------
# Phase 3: Visualization + Report
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestOfflineE2EVisualizationReport:
    """Test chart generation and report formatting with pipeline output."""

    def test_prediction_chart_creates_file(
        self, e2e_fixture_path, fast_sklearn_params, e2e_output_dir
    ):
        """plot_prediction_results produces a non-empty PNG file."""
        import matplotlib.pyplot as plt

        processed = _load_and_process(e2e_fixture_path)
        X_train, X_test, y_train, y_test = _split_features_target(processed)

        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Build Series for visualizer
        pred_dates = y_test.index[-len(preds):]
        predicted_series = pd.Series(preds, index=pred_dates)
        actual_series = y_test.iloc[-len(preds):]

        figures_dir = e2e_output_dir / "results" / "figures"
        chart_path = str(figures_dir / "e2e_offline_chart.png")

        viz = CurrencyVisualizer(output_dir=str(figures_dir))
        fig = viz.plot_prediction_results(
            actual=actual_series,
            predicted=predicted_series,
            symbol="USDTWD",
            save_path=chart_path,
        )
        plt.close(fig)

        assert Path(chart_path).exists()
        assert Path(chart_path).stat().st_size > 0

    def test_report_generation_produces_content(self):
        """ResultFormatter.generate_report() produces non-empty Markdown."""
        formatter = ResultFormatter(use_logger=False)
        results = {
            "success": True,
            "pipeline_status": {
                "data_collection": True,
                "model_training": True,
                "prediction": True,
                "results_saved": True,
            },
            "predictions": [
                {
                    "symbol": "USDTWD=X",
                    "predictions": [30.5, 30.6, 30.7],
                    "last_known_value": 30.4,
                }
            ],
        }
        report = formatter.generate_report(results)
        assert len(report) > 100
        assert "USDTWD" in report


# ---------------------------------------------------------------------------
# Phase 4: Backtesting with Real Model
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestOfflineE2EBacktest:
    """Test BacktestRunner with real PatchTSTSklearn on fixture data."""

    # Smaller seq_len to fit within test_size=60 fold windows
    _backtest_params = {
        "seq_len": 20,
        "pred_len": 5,
        "patch_len": 5,
        "stride": 5,
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
    }

    def test_backtest_sklearn_on_fixture_data(self, e2e_fixture_path):
        """BacktestRunner produces valid BacktestResult with real model."""
        processed = _load_and_process(e2e_fixture_path)

        runner = BacktestRunner(
            config={"model_params": self._backtest_params},
        )
        splitter = WalkForwardSplitter(
            strategy="rolling",
            initial_train_size=100,
            test_size=60,
            step_size=60,
        )
        folds = splitter.split(len(processed))

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
        assert result.n_folds >= 2
        assert np.all(np.isfinite(list(result.avg_prediction_metrics.values())))
        assert np.all(np.isfinite(list(result.avg_financial_metrics.values())))
        assert len(result.overall_equity_curve) > 0
        assert result.total_training_time > 0

    def test_backtest_fold_metrics_are_consistent(self, e2e_fixture_path):
        """Each fold in backtest result has valid metrics and prices."""
        processed = _load_and_process(e2e_fixture_path)

        runner = BacktestRunner(
            config={"model_params": self._backtest_params},
        )
        splitter = WalkForwardSplitter(
            strategy="rolling",
            initial_train_size=100,
            test_size=60,
            step_size=60,
        )
        folds = splitter.split(len(processed))

        result = runner._run_single_backtest(
            symbol="USDTWD=X",
            model_name="patchtst_sklearn",
            processed_data=processed,
            folds=folds,
            strategy="rolling",
        )

        for fr in result.fold_results:
            assert fr.train_size > 0
            assert fr.test_size > 0
            assert "rmse" in fr.prediction_metrics
            assert "sharpe_ratio" in fr.financial_metrics
            assert np.all(np.isfinite(fr.actual_prices))
            assert np.all(np.isfinite(fr.predicted_prices))
