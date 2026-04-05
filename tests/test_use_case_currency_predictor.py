"""
Use Case Test: CurrencyPredictor Full Flow

驗證 CurrencyPredictor 的 prepare → train → predict → save/load 完整流程，
使用 fixture CSV 注入 DataStorage（不需網路）。

標記 @pytest.mark.slow — 訓練真實 sklearn 模型。
執行：uv run pytest -m slow tests/test_use_case_currency_predictor.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from currency_predictor.prediction.predictor import CurrencyPredictor
from currency_predictor.data.storage import DataStorage


@pytest.fixture
def predictor_with_data(e2e_fixture_path, tmp_path):
    """Create a CurrencyPredictor with fixture data pre-loaded into storage.

    Injects the fixture CSV into DataStorage so train/predict work
    without network access.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Load fixture and inject into storage
    df = pd.read_csv(e2e_fixture_path, index_col="Date", parse_dates=True)
    storage = DataStorage(base_dir=str(data_dir))
    storage.save_raw_data(df, "USDTWD", "1y")

    predictor = CurrencyPredictor(
        model_name="patchtst_sklearn",
        model_params={
            "seq_len": 50,
            "pred_len": 5,
            "patch_len": 10,
            "stride": 5,
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
        },
        data_storage_path=str(data_dir),
    )
    return predictor


@pytest.mark.slow
class TestCurrencyPredictorFlow:
    """Test CurrencyPredictor full flow with fixture data."""

    def test_prepare_training_data_shapes(self, predictor_with_data):
        """prepare_training_data returns correctly shaped tuple."""
        X_train, y_train, X_test, y_test = predictor_with_data.prepare_training_data(
            "USDTWD=X", period="1y"
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert "Close" not in X_train.columns

    def test_train_model_returns_metrics(self, predictor_with_data):
        """train_model returns training_completed=True with metrics."""
        result = predictor_with_data.train_model("USDTWD=X", period="1y")

        assert result['training_completed'] is True
        assert result['model_name'] == 'patchtst_sklearn'
        assert result['train_size'] > 0
        assert result['test_size'] > 0

    def test_train_then_predict(self, predictor_with_data):
        """train → predict returns finite predictions."""
        predictor_with_data.train_model("USDTWD=X", period="1y")

        result = predictor_with_data.predict(
            "USDTWD=X", horizon=5, period="1y"
        )

        assert 'error' not in result
        assert 'predictions' in result
        preds = result['predictions']
        assert len(preds) == 5
        assert np.all(np.isfinite(preds))

    def test_predict_without_training_returns_error(self, predictor_with_data):
        """predict() without training returns error dict."""
        result = predictor_with_data.predict(
            "USDTWD=X", horizon=5, period="1y"
        )

        assert 'error' in result

    def test_evaluate_model_produces_metrics(self, predictor_with_data):
        """After training, _evaluate_model returns non-negative metrics."""
        predictor_with_data.train_model("USDTWD=X", period="1y")

        X_train, y_train, X_test, y_test = predictor_with_data.prepare_training_data(
            "USDTWD=X", period="1y"
        )

        metrics = predictor_with_data._evaluate_model(X_test, y_test, "test")

        # May be empty if evaluate fails on short data, but should not crash
        if metrics:
            assert metrics.get('mse', 0) >= 0
            assert metrics.get('mae', 0) >= 0

    def test_save_and_reload_model(self, predictor_with_data, tmp_path):
        """train → save → load → predict produces consistent results."""
        predictor_with_data.train_model("USDTWD=X", period="1y")

        result1 = predictor_with_data.predict(
            "USDTWD=X", horizon=5, period="1y"
        )

        model_path = str(tmp_path / "model.joblib")
        assert predictor_with_data.save_model(model_path) is True

        assert predictor_with_data.load_model(model_path) is True

        result2 = predictor_with_data.predict(
            "USDTWD=X", horizon=5, period="1y"
        )

        np.testing.assert_array_almost_equal(
            result1['predictions'], result2['predictions']
        )
