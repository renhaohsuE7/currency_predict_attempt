"""
Use Case Test: Save → Load → Re-predict

驗證 PatchTSTSklearn 模型的完整 save/load/predict cycle，
使用真實 fixture 資料訓練。

標記 @pytest.mark.slow — 訓練真實 sklearn 模型。
執行：uv run pytest -m slow tests/test_use_case_save_load_predict.py -v
"""

import pytest
import numpy as np
import pandas as pd

from currency_predictor.models.patchtst.sklearn.model import PatchTSTSklearn


def _split_xy(processed, target_col="Close"):
    """Split processed DataFrame into features and target."""
    feature_cols = [c for c in processed.columns if c != target_col]
    return processed[feature_cols], processed[target_col]


@pytest.mark.slow
class TestSaveLoadPredictCycle:
    """Test full save → load → re-predict cycle with fixture data."""

    def test_save_load_predict_consistency(
        self, processed_fixture_data, fast_sklearn_params, tmp_path
    ):
        """fit → predict → save → load → predict → results match."""
        X, y = _split_xy(processed_fixture_data)

        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X, y)
        preds_before = model.predict(X)

        # Save and load
        model_path = str(tmp_path / "model.joblib")
        assert model.save_model(model_path) is True

        model2 = PatchTSTSklearn(**fast_sklearn_params)
        assert model2.load_model(model_path) is True

        preds_after = model2.predict(X)
        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_load_model_predict_with_extra_columns(
        self, processed_fixture_data, fast_sklearn_params, tmp_path
    ):
        """save → load → predict(X_with_extra_col) auto-filters columns."""
        X, y = _split_xy(processed_fixture_data)

        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X, y)

        model_path = str(tmp_path / "model.joblib")
        model.save_model(model_path)

        model2 = PatchTSTSklearn(**fast_sklearn_params)
        model2.load_model(model_path)

        # Add extra column (e.g. Close that was excluded from X)
        X_extra = pd.concat([X, y.to_frame("Close")], axis=1)
        preds = model2.predict(X_extra)
        assert len(preds) == fast_sklearn_params["pred_len"]
        assert np.all(np.isfinite(preds))

    def test_save_load_preserves_training_history(
        self, processed_fixture_data, fast_sklearn_params, tmp_path
    ):
        """save → load → training_history matches."""
        X, y = _split_xy(processed_fixture_data)

        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X, y)
        history_before = model.training_history.copy()

        model_path = str(tmp_path / "model.joblib")
        model.save_model(model_path)

        model2 = PatchTSTSklearn(**fast_sklearn_params)
        model2.load_model(model_path)

        assert model2.training_history['train_loss'] == history_before['train_loss']

    def test_save_load_preserves_scaler(
        self, processed_fixture_data, fast_sklearn_params, tmp_path
    ):
        """save → load → scaler preserved."""
        X, y = _split_xy(processed_fixture_data)

        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X, y)
        scaler_mean_before = model.scaler.mean_.copy()

        model_path = str(tmp_path / "model.joblib")
        model.save_model(model_path)

        model2 = PatchTSTSklearn(**fast_sklearn_params)
        model2.load_model(model_path)

        np.testing.assert_array_almost_equal(
            model2.scaler.mean_, scaler_mean_before
        )

    def test_load_nonexistent_file_returns_false(self, fast_sklearn_params):
        """load_model with bad path returns False."""
        model = PatchTSTSklearn(**fast_sklearn_params)
        assert model.load_model("/nonexistent/path/model.joblib") is False
