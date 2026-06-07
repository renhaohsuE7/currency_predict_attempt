"""
測試 PyTorch Lightning PatchTST

Integration tests for PatchTSTLightningWrapper: fit → predict → save → load
Mirrors the structure of test_patchtst_huggingface.py.
"""

import pytest
import numpy as np
import pandas as pd

# Skip entire module if pytorch-lightning is not installed
pl = pytest.importorskip("pytorch_lightning")

from currency_predictor.models.patchtst.lightning import (  # noqa: E402
    PatchTSTLightningWrapper,
)
from currency_predictor.models.base import BaseModel, ModelType  # noqa: E402
from currency_predictor.models.factory import ModelFactory  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dummy_data():
    """Create dummy time series data (shared across all tests in module)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    return pd.DataFrame(
        {"Close": np.random.randn(500).cumsum() + 100},
        index=dates,
    )


def _make_small_model(**overrides):
    """Helper: create a lightweight model for fast testing."""
    defaults = dict(
        context_length=64,
        prediction_length=7,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_epochs=3,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping_patience=0,
    )
    defaults.update(overrides)
    return PatchTSTLightningWrapper(**defaults)


@pytest.fixture(scope="module")
def trained_model(dummy_data):
    """Train a model once and reuse across tests."""
    model = _make_small_model()
    model.fit(dummy_data, dummy_data["Close"], num_epochs=3)
    return model


# ---------------------------------------------------------------------------
# TestLightningInit
# ---------------------------------------------------------------------------


class TestPatchTSTLightningInit:
    """測試初始化"""

    def test_default_init(self):
        model = PatchTSTLightningWrapper(accelerator="cpu", devices=1)
        assert model.model_name == "PatchTST_Lightning"
        assert model.model_type == ModelType.TRANSFORMER_BASED
        assert model.is_fitted is False

    def test_custom_params(self):
        model = _make_small_model(context_length=128, prediction_length=14)
        assert model.config.context_length == 128
        assert model.config.prediction_length == 14

    def test_isinstance_base_model(self):
        model = PatchTSTLightningWrapper(accelerator="cpu", devices=1)
        assert isinstance(model, BaseModel)

    def test_device_is_set(self):
        model = PatchTSTLightningWrapper(accelerator="cpu", devices=1)
        assert model.device is not None


# ---------------------------------------------------------------------------
# TestLightningFit
# ---------------------------------------------------------------------------


class TestPatchTSTLightningFit:
    """測試訓練"""

    def test_fit_marks_fitted(self, trained_model):
        assert trained_model.is_fitted is True

    def test_fit_returns_self(self, dummy_data):
        model = _make_small_model(max_epochs=1)
        result = model.fit(dummy_data, dummy_data["Close"], num_epochs=1)
        assert result is model

    def test_training_history_populated(self, trained_model):
        history = trained_model.training_history
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0
        assert history["best_val_loss"] < float("inf")

    def test_fit_with_training_config(self, dummy_data):
        """測試接受 TrainingConfig"""
        from currency_predictor.models.patchtst.config import TrainingConfig

        tc = TrainingConfig(num_epochs=2, batch_size=32)
        model = _make_small_model(max_epochs=2)
        model.fit(dummy_data, dummy_data["Close"], training_config=tc)
        assert model.is_fitted is True


# ---------------------------------------------------------------------------
# TestLightningPredict
# ---------------------------------------------------------------------------


class TestPatchTSTLightningPredict:
    """測試預測"""

    def test_predict_shape(self, trained_model, dummy_data):
        preds = trained_model.predict(dummy_data)
        assert preds.shape == (7,)

    def test_predict_custom_horizon(self, trained_model, dummy_data):
        preds = trained_model.predict(dummy_data, horizon=3)
        assert preds.shape == (3,)

    def test_predict_values_finite(self, trained_model, dummy_data):
        preds = trained_model.predict(dummy_data)
        assert np.all(np.isfinite(preds))

    def test_predict_not_fitted_raises(self, dummy_data):
        model = _make_small_model()
        with pytest.raises(ValueError, match="尚未訓練"):
            model.predict(dummy_data)

    def test_predict_insufficient_data_raises(self, trained_model):
        short_df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="小於"):
            trained_model.predict(short_df)


# ---------------------------------------------------------------------------
# TestLightningUncertainty
# ---------------------------------------------------------------------------


class TestPatchTSTLightningUncertainty:
    """測試不確定性預測"""

    def test_uncertainty_keys(self, trained_model, dummy_data):
        result = trained_model.predict_with_uncertainty(dummy_data, n_samples=10)
        assert "predictions" in result
        assert "std" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "confidence_level" in result

    def test_uncertainty_shapes(self, trained_model, dummy_data):
        result = trained_model.predict_with_uncertainty(dummy_data, n_samples=10)
        assert result["predictions"].shape == (7,)
        assert result["std"].shape == (7,)
        assert result["lower_bound"].shape == (7,)
        assert result["upper_bound"].shape == (7,)

    def test_uncertainty_bounds_order(self, trained_model, dummy_data):
        result = trained_model.predict_with_uncertainty(dummy_data, n_samples=50)
        assert np.all(result["lower_bound"] <= result["upper_bound"])

    def test_uncertainty_not_fitted_raises(self, dummy_data):
        model = _make_small_model()
        with pytest.raises(ValueError, match="尚未訓練"):
            model.predict_with_uncertainty(dummy_data)


# ---------------------------------------------------------------------------
# TestLightningSaveLoad
# ---------------------------------------------------------------------------


class TestPatchTSTLightningSaveLoad:
    """測試模型存取"""

    def test_save_load_cycle(self, trained_model, dummy_data, tmp_path):
        preds_before = trained_model.predict(dummy_data)

        # Save
        save_path = str(tmp_path / "lightning_model")
        assert trained_model.save_model(save_path) is True

        # Load into new model
        model2 = _make_small_model()
        assert model2.load_model(save_path) is True
        assert model2.is_fitted is True

        # Predictions should match
        preds_after = model2.predict(dummy_data)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-4)

    def test_load_nonexistent_path(self):
        model = _make_small_model()
        result = model.load_model("/nonexistent/path/model")
        assert result is False


# ---------------------------------------------------------------------------
# TestLightningEvaluate
# ---------------------------------------------------------------------------


class TestPatchTSTLightningEvaluate:
    """測試評估"""

    def test_evaluate_returns_metrics(self, trained_model, dummy_data):
        metrics = trained_model.evaluate_single_shot(dummy_data, dummy_data["Close"])
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mda" in metrics
        assert "direction_accuracy" in metrics
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0


# ---------------------------------------------------------------------------
# TestLightningModelInfo
# ---------------------------------------------------------------------------


class TestPatchTSTLightningModelInfo:
    """測試模型資訊"""

    def test_get_model_info(self, trained_model):
        info = trained_model.get_model_info()
        assert info["model_name"] == "PatchTST_Lightning"
        assert info["model_type"] == "transformer_based"
        assert info["implementation"] == "lightning"
        assert info["is_fitted"] is True
        assert info["context_length"] == 64
        assert info["prediction_length"] == 7

    def test_get_model_info_has_trainable_params(self, trained_model):
        info = trained_model.get_model_info()
        assert "trainable_parameters" in info
        assert info["trainable_parameters"] > 0

    def test_get_model_info_unfitted(self):
        """Should NOT crash when lightning_model is None (before fit)."""
        model = _make_small_model()
        info = model.get_model_info()
        assert info["is_fitted"] is False
        assert "trainable_parameters" not in info


# ---------------------------------------------------------------------------
# TestLightningFactory
# ---------------------------------------------------------------------------


class TestPatchTSTLightningFactory:
    """測試 ModelFactory 整合"""

    def test_create_via_factory(self):
        model = ModelFactory.create_model(
            "patchtst_lightning",
            accelerator="cpu",
            devices=1,
        )
        assert isinstance(model, PatchTSTLightningWrapper)

    def test_factory_with_params(self):
        model = ModelFactory.create_model(
            "patchtst_lightning",
            context_length=128,
            prediction_length=14,
            accelerator="cpu",
            devices=1,
        )
        assert model.config.context_length == 128
        assert model.config.prediction_length == 14


# ---------------------------------------------------------------------------
# Multi-channel regression tests
#
# Guards a real bug: `use_multi_channel` was silently dropped (kwargs not
# forwarded into the config), and when the target "Close" channel was absent
# the wrapper silently predicted a different-scale channel. These tests fit on
# multi-column data whose channels have deliberately different scales so a
# wrong-channel prediction is detectable.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multichannel_data():
    """Multi-column data: Close ~100-scale, BigFeat ~100000-scale.

    The huge scale gap means a prediction on the wrong channel is easy to
    detect (BigFeat predictions would be ~100000, far above Close's ~100).
    """
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100.0 + rng.standard_normal(n).cumsum()
    big_feat = 100_000.0 + rng.standard_normal(n).cumsum() * 50.0
    small_feat = 1.0 + rng.standard_normal(n).cumsum() * 0.01
    return pd.DataFrame(
        {"Close": close, "BigFeat": big_feat, "SmallFeat": small_feat},
        index=dates,
    )


def _make_small_mc_model(**overrides):
    """Helper: tiny multi-channel Lightning model for fast testing."""
    defaults = dict(
        use_multi_channel=True,
        seq_len=32,
        pred_len=5,
        patch_len=8,
        stride=4,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_epochs=2,
        accelerator="cpu",
        devices=1,
    )
    defaults.update(overrides)
    return ModelFactory.create_model("patchtst_lightning", **defaults)


class TestPatchTSTLightningMultiChannel:
    """多 channel 回歸測試"""

    def test_use_multi_channel_flag_is_honored(self):
        """use_multi_channel=True 必須穿透 kwargs → config(防靜默丟棄 bug)。"""
        model = _make_small_mc_model()
        assert model.use_multi_channel is True
        assert model.config.use_multi_channel is True

    def test_multichannel_predicts_close_channel_scale(self, multichannel_data):
        """多 channel 預測必須落在 Close (~100) 尺度,而非 BigFeat (~100000)。"""
        model = _make_small_mc_model()
        model.fit(multichannel_data, num_epochs=2)
        preds = model.predict(multichannel_data)

        assert preds.shape == (5,)
        # 預測必須遠離 BigFeat 的 100000 尺度 — 抓「預測錯 channel」。
        assert np.all(np.abs(preds) < 10_000), (
            f"predictions {preds} look like the wrong channel "
            f"(BigFeat ~100000 scale), not Close ~100"
        )
        # 進一步:應落在 Close 最後值附近的合理區間。
        last_close = multichannel_data["Close"].iloc[-1]
        assert np.all(
            np.abs(preds - last_close) < 50
        ), f"predictions {preds} far from last Close {last_close}"

    def test_multichannel_missing_close_raises(self, multichannel_data):
        """多 channel 但無 Close 欄位 → 必須 fail loud(ValueError 含 'Close')。"""
        no_close = multichannel_data.drop(columns=["Close"])
        model = _make_small_mc_model()
        with pytest.raises(ValueError, match="Close"):
            model.fit(no_close, num_epochs=1)
