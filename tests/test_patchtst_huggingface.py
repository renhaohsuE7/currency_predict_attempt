"""
測試 HuggingFace PatchTST

Integration tests for PatchTSTHuggingFace: fit → predict → save → load
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from currency_predictor.models.patchtst import PatchTSTHuggingFace
from currency_predictor.models.patchtst.config import PatchTSTConfig, TrainingConfig
from currency_predictor.models.base import BaseModel, ModelType


@pytest.fixture(scope="module")
def dummy_data():
    """Create dummy time series data (shared across all tests in module)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    return pd.DataFrame(
        {"Close": np.random.randn(500).cumsum() + 100},
        index=dates,
    )


@pytest.fixture(scope="module")
def trained_model(dummy_data):
    """Train a model once and reuse across tests."""
    model = PatchTSTHuggingFace(context_length=64, prediction_length=7)
    model.fit(dummy_data, dummy_data["Close"], num_epochs=2)
    return model


class TestPatchTSTHuggingFaceInit:
    """測試初始化"""

    def test_default_init(self):
        model = PatchTSTHuggingFace()
        assert model.model_name == "PatchTST_HuggingFace"
        assert model.model_type == ModelType.TRANSFORMER_BASED
        assert model.is_fitted is False

    def test_custom_params(self):
        model = PatchTSTHuggingFace(
            context_length=128,
            prediction_length=14,
            d_model=128,
            num_attention_heads=8,
        )
        assert model.context_length == 128
        assert model.prediction_length == 14

    def test_isinstance_base_model(self):
        model = PatchTSTHuggingFace()
        assert isinstance(model, BaseModel)

    def test_device_is_set(self):
        model = PatchTSTHuggingFace()
        assert model.device is not None


class TestPatchTSTHuggingFaceFit:
    """測試訓練"""

    def test_fit_marks_fitted(self, trained_model):
        assert trained_model.is_fitted is True

    def test_fit_returns_self(self, dummy_data):
        model = PatchTSTHuggingFace(context_length=64, prediction_length=7)
        result = model.fit(dummy_data, dummy_data["Close"], num_epochs=1)
        assert result is model

    def test_fit_with_training_config(self, dummy_data):
        """測試接受 TrainingConfig"""
        from currency_predictor.models.patchtst.config import TrainingConfig
        tc = TrainingConfig(num_epochs=1, batch_size=32, learning_rate=1e-3)
        model = PatchTSTHuggingFace(context_length=64, prediction_length=7)
        model.fit(dummy_data, dummy_data["Close"], training_config=tc)
        assert model.is_fitted is True


class TestPatchTSTHuggingFacePredict:
    """測試預測"""

    def test_predict_shape(self, trained_model, dummy_data):
        preds = trained_model.predict(dummy_data)
        assert preds.shape == (7,)

    def test_predict_custom_horizon(self, trained_model, dummy_data):
        preds = trained_model.predict(dummy_data, horizon=3)
        assert preds.shape == (3,)

    def test_predict_values_reasonable(self, trained_model, dummy_data):
        preds = trained_model.predict(dummy_data)
        last_close = dummy_data["Close"].iloc[-1]
        # Predictions should be in a reasonable range (within 50% of last value)
        assert all(abs(p - last_close) / abs(last_close) < 0.5 for p in preds)

    def test_predict_not_fitted_raises(self, dummy_data):
        model = PatchTSTHuggingFace()
        with pytest.raises(ValueError, match="尚未訓練"):
            model.predict(dummy_data)

    def test_predict_insufficient_data_raises(self, trained_model):
        short_df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="小於 context_length"):
            trained_model.predict(short_df)


class TestPatchTSTHuggingFaceUncertainty:
    """測試不確定性預測"""

    def test_uncertainty_keys(self, trained_model, dummy_data):
        result = trained_model.predict_with_uncertainty(dummy_data)
        assert "predictions" in result
        assert "std" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "confidence_level" in result

    def test_uncertainty_shapes(self, trained_model, dummy_data):
        result = trained_model.predict_with_uncertainty(dummy_data)
        assert result["predictions"].shape == (7,)
        assert result["std"].shape == (7,)
        assert result["lower_bound"].shape == (7,)
        assert result["upper_bound"].shape == (7,)

    def test_uncertainty_bounds_order(self, trained_model, dummy_data):
        result = trained_model.predict_with_uncertainty(dummy_data)
        assert all(result["lower_bound"] <= result["predictions"])
        assert all(result["predictions"] <= result["upper_bound"])

    def test_uncertainty_not_fitted_raises(self, dummy_data):
        model = PatchTSTHuggingFace()
        with pytest.raises(ValueError, match="尚未訓練"):
            model.predict_with_uncertainty(dummy_data)


class TestPatchTSTHuggingFaceSaveLoad:
    """測試模型存取"""

    def test_save_load_cycle(self, trained_model, dummy_data, tmp_path):
        preds_before = trained_model.predict(dummy_data)

        # Save
        save_path = str(tmp_path / "hf_model")
        assert trained_model.save_model(save_path) is True

        # Load into new model
        model2 = PatchTSTHuggingFace(context_length=64, prediction_length=7)
        assert model2.load_model(save_path) is True
        assert model2.is_fitted is True

        # Predictions should match
        preds_after = model2.predict(dummy_data)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-4)

    def test_load_nonexistent_path(self):
        model = PatchTSTHuggingFace()
        result = model.load_model("/nonexistent/path/model")
        assert result is False


class TestPatchTSTHuggingFaceEvaluate:
    """測試評估"""

    def test_evaluate_returns_metrics(self, trained_model, dummy_data):
        metrics = trained_model.evaluate(dummy_data, dummy_data["Close"])
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0


class TestPatchTSTHuggingFaceModelInfo:
    """測試模型資訊"""

    def test_get_model_info(self, trained_model):
        info = trained_model.get_model_info()
        assert info["model_name"] == "PatchTST_HuggingFace"
        assert info["model_type"] == "transformer_based"
        assert info["implementation"] == "huggingface"
        assert info["is_fitted"] is True
        assert info["context_length"] == 64
        assert info["prediction_length"] == 7

    def test_get_model_info_has_pretrained_fields(self, trained_model):
        info = trained_model.get_model_info()
        assert "pretrained_model_name_or_path" in info
        assert "fine_tune_mode" in info
        assert info["fine_tune_mode"] == "from_scratch"

    def test_get_model_info_has_trainable_params(self, trained_model):
        info = trained_model.get_model_info()
        assert "trainable_parameters" in info
        assert info["trainable_parameters"] > 0


class TestPatchTSTPretrainedInit:
    """測試預訓練模型初始化參數"""

    def test_default_is_from_scratch(self):
        model = PatchTSTHuggingFace()
        assert model.pretrained_model_name_or_path is None
        assert model.fine_tune_mode == "from_scratch"

    def test_accepts_pretrained_params(self):
        model = PatchTSTHuggingFace(
            pretrained_model_name_or_path="ibm-granite/granite-timeseries-patchtst",
            fine_tune_mode="full",
        )
        assert model.pretrained_model_name_or_path == "ibm-granite/granite-timeseries-patchtst"
        assert model.fine_tune_mode == "full"
        assert model.is_fitted is False

    def test_config_passes_pretrained_params(self):
        config = PatchTSTConfig(
            pretrained_model_name_or_path="some-model",
            fine_tune_mode="linear_probe",
        )
        model = PatchTSTHuggingFace(config=config)
        assert model.pretrained_model_name_or_path == "some-model"
        assert model.fine_tune_mode == "linear_probe"

    def test_constructor_overrides_config(self):
        config = PatchTSTConfig(fine_tune_mode="linear_probe")
        model = PatchTSTHuggingFace(
            config=config,
            fine_tune_mode="full",
        )
        assert model.fine_tune_mode == "full"


class TestPatchTSTFineTuneDefaults:
    """測試 TrainingConfig.for_fine_tune_mode()"""

    def test_from_scratch_defaults(self):
        tc = TrainingConfig.for_fine_tune_mode("from_scratch")
        assert tc.num_epochs == 50
        assert tc.learning_rate == 1e-4
        assert tc.early_stopping_patience == 10

    def test_full_defaults(self):
        tc = TrainingConfig.for_fine_tune_mode("full")
        assert tc.num_epochs == 20
        assert tc.learning_rate == 1e-5
        assert tc.early_stopping_patience == 5

    def test_linear_probe_defaults(self):
        tc = TrainingConfig.for_fine_tune_mode("linear_probe")
        assert tc.num_epochs == 10
        assert tc.learning_rate == 1e-3
        assert tc.early_stopping_patience == 5

    def test_unknown_mode_uses_from_scratch(self):
        tc = TrainingConfig.for_fine_tune_mode("unknown")
        assert tc.num_epochs == 50  # from_scratch default


class TestPatchTSTConfigPretrained:
    """測試 PatchTSTConfig 預訓練相關欄位"""

    def test_is_pretrained_false_by_default(self):
        config = PatchTSTConfig()
        assert config.is_pretrained is False

    def test_is_pretrained_true_with_path(self):
        config = PatchTSTConfig(pretrained_model_name_or_path="some-model")
        assert config.is_pretrained is True

    def test_to_dict_includes_pretrained_fields(self):
        config = PatchTSTConfig(
            pretrained_model_name_or_path="test-model",
            fine_tune_mode="full",
        )
        d = config.to_dict()
        assert d["pretrained_model_name_or_path"] == "test-model"
        assert d["fine_tune_mode"] == "full"

    def test_from_scratch_backward_compat(self, dummy_data):
        """from_scratch 模式應完全向後相容"""
        model = PatchTSTHuggingFace(context_length=64, prediction_length=7)
        model.fit(dummy_data, dummy_data["Close"], num_epochs=1)
        preds = model.predict(dummy_data)
        assert preds.shape == (7,)
