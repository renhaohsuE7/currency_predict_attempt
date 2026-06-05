"""
Tests for ModelComparer CSV export methods.
"""

import numpy as np
import pandas as pd
import pytest

from currency_predictor.prediction.comparer import ModelComparer


@pytest.fixture
def sample_comparison_results():
    """Minimal comparison_results dict matching comparer.compare() output."""
    dates = pd.date_range("2026-03-25", periods=5, freq="D").tolist()
    actual = np.array([31.0, 31.1, 31.2, 31.3, 31.4])
    return {
        "symbols_results": {
            "USDTWD=X": {
                "actual": actual,
                "models": {
                    "naive": {
                        "predictions": actual + 0.01,
                        "prediction_dates": dates,
                        "train_metrics": {"mse": 0.01, "mae": 0.08, "rmse": 0.10, "mape": 0.25},
                        "test_metrics": {
                            "mse": 0.005,
                            "mae": 0.06,
                            "rmse": 0.07,
                            "mape": 0.20,
                            "mase": 0.58,
                            "mda": 0.0,
                        },
                    },
                    "patchtst_sklearn": {
                        "predictions": actual - 0.02,
                        "prediction_dates": dates,
                        "train_metrics": {"mse": 0.02, "mae": 0.10, "rmse": 0.14, "mape": 0.30},
                        "test_metrics": {
                            "mse": 0.04,
                            "mae": 0.18,
                            "rmse": 0.21,
                            "mape": 0.57,
                            "mase": 1.72,
                            "mda": 42.86,
                        },
                    },
                },
                "best_model": "naive",
            }
        },
        "overall_ranking": [("naive", 0.07), ("patchtst_sklearn", 0.21)],
        "model_names": ["naive", "patchtst_sklearn"],
        "prediction_horizon": 5,
    }


@pytest.fixture
def comparer():
    """ModelComparer instance (models don't matter for CSV tests)."""
    return ModelComparer(
        model_names=["naive", "patchtst_sklearn"],
        config={"data_collection": {"period": "6mo", "interval": "1d"}},
        output_dir="/tmp/test_comparer",
    )


class TestSavePredictionsCsv:
    def test_creates_csv(self, comparer, sample_comparison_results, tmp_path):
        comparer.save_predictions_csv(sample_comparison_results, tmp_path)
        csv_path = tmp_path / "predictions_USDTWD.csv"
        assert csv_path.exists()

    def test_csv_columns(self, comparer, sample_comparison_results, tmp_path):
        comparer.save_predictions_csv(sample_comparison_results, tmp_path)
        df = pd.read_csv(tmp_path / "predictions_USDTWD.csv")
        assert "Date" in df.columns
        assert "Actual" in df.columns
        assert "naive" in df.columns
        assert "patchtst_sklearn" in df.columns

    def test_csv_row_count(self, comparer, sample_comparison_results, tmp_path):
        comparer.save_predictions_csv(sample_comparison_results, tmp_path)
        df = pd.read_csv(tmp_path / "predictions_USDTWD.csv")
        assert len(df) == 5

    def test_skips_error_models(self, comparer, sample_comparison_results, tmp_path):
        results = sample_comparison_results
        results["symbols_results"]["USDTWD=X"]["models"]["patchtst_sklearn"]["error"] = "fail"
        comparer.save_predictions_csv(results, tmp_path)
        df = pd.read_csv(tmp_path / "predictions_USDTWD.csv")
        assert "patchtst_sklearn" not in df.columns
        assert "naive" in df.columns

    def test_no_actual(self, comparer, sample_comparison_results, tmp_path):
        del sample_comparison_results["symbols_results"]["USDTWD=X"]["actual"]
        comparer.save_predictions_csv(sample_comparison_results, tmp_path)
        df = pd.read_csv(tmp_path / "predictions_USDTWD.csv")
        assert "Actual" not in df.columns

    def test_empty_results(self, comparer, tmp_path):
        comparer.save_predictions_csv({"symbols_results": {}}, tmp_path)
        assert not list(tmp_path.glob("predictions_*.csv"))


class TestSaveMetricsCsv:
    def test_creates_csv(self, comparer, sample_comparison_results, tmp_path):
        comparer.save_metrics_csv(sample_comparison_results, tmp_path)
        assert (tmp_path / "metrics.csv").exists()

    def test_csv_columns(self, comparer, sample_comparison_results, tmp_path):
        comparer.save_metrics_csv(sample_comparison_results, tmp_path)
        df = pd.read_csv(tmp_path / "metrics.csv")
        expected = {"Symbol", "Model", "Dataset", "MSE", "MAE", "RMSE", "MAPE", "MASE", "MDA"}
        assert expected == set(df.columns)

    def test_row_count(self, comparer, sample_comparison_results, tmp_path):
        """2 models x 2 datasets (train + test) = 4 rows."""
        comparer.save_metrics_csv(sample_comparison_results, tmp_path)
        df = pd.read_csv(tmp_path / "metrics.csv")
        assert len(df) == 4

    def test_train_has_no_mase(self, comparer, sample_comparison_results, tmp_path):
        comparer.save_metrics_csv(sample_comparison_results, tmp_path)
        df = pd.read_csv(tmp_path / "metrics.csv")
        train_rows = df[df["Dataset"] == "train"]
        assert train_rows["MASE"].isna().all()

    def test_test_has_mase(self, comparer, sample_comparison_results, tmp_path):
        comparer.save_metrics_csv(sample_comparison_results, tmp_path)
        df = pd.read_csv(tmp_path / "metrics.csv")
        test_rows = df[df["Dataset"] == "test"]
        assert test_rows["MASE"].notna().all()

    def test_skips_error_models(self, comparer, sample_comparison_results, tmp_path):
        results = sample_comparison_results
        results["symbols_results"]["USDTWD=X"]["models"]["patchtst_sklearn"]["error"] = "fail"
        comparer.save_metrics_csv(results, tmp_path)
        df = pd.read_csv(tmp_path / "metrics.csv")
        assert set(df["Model"].unique()) == {"naive"}

    def test_empty_results(self, comparer, tmp_path):
        comparer.save_metrics_csv({"symbols_results": {}}, tmp_path)
        assert not (tmp_path / "metrics.csv").exists()
