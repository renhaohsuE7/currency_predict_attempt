"""
預訓練模型整合測試

需要網路下載模型，執行較慢，以 @pytest.mark.slow 標記。
執行方式: uv run pytest -m slow
"""

import pytest
import numpy as np
import pandas as pd

from currency_predictor.models.patchtst import PatchTSTHuggingFace


# 標記整個模組為 slow
pytestmark = pytest.mark.slow

PRETRAINED_MODEL = "ibm-granite/granite-timeseries-patchtst"


@pytest.fixture(scope="module")
def long_dummy_data():
    """建立較長的 dummy data（預訓練模型 context_length=512 需要 >519 筆）"""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=600, freq="D")
    return pd.DataFrame(
        {"Close": np.random.randn(600).cumsum() + 100},
        index=dates,
    )


class TestPretrainedModelLoad:
    """測試從 Hub 載入預訓練模型"""

    def test_load_granite_model(self):
        model = PatchTSTHuggingFace(
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            fine_tune_mode="full",
            prediction_length=7,
        )
        model._setup_model(num_features=1)
        assert model.model is not None
        # 架構參數應來自預訓練模型
        assert model.d_model > 64  # Granite 用 128
        assert model.prediction_length == 7  # 被覆蓋

    def test_pretrained_updates_context_length(self):
        model = PatchTSTHuggingFace(
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            fine_tune_mode="full",
            prediction_length=7,
        )
        model._setup_model(num_features=1)
        # context_length 應從預訓練模型讀取（不是 constructor 的 64）
        assert model.context_length == model.model.config.context_length

    def test_invalid_model_name_raises(self):
        model = PatchTSTHuggingFace(
            pretrained_model_name_or_path="nonexistent/fake-model-xyz",
            fine_tune_mode="full",
        )
        with pytest.raises(RuntimeError, match="無法載入預訓練模型"):
            model._setup_model(num_features=1)


class TestPretrainedFineTune:
    """測試 fine-tune 端對端"""

    def test_full_finetune_fit_predict(self, long_dummy_data):
        model = PatchTSTHuggingFace(
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            fine_tune_mode="full",
            prediction_length=7,
        )
        model.fit(long_dummy_data, long_dummy_data["Close"], num_epochs=1)
        preds = model.predict(long_dummy_data)
        assert preds.shape == (7,)
        assert model.is_fitted is True

    def test_linear_probe_freezes_backbone(self):
        model = PatchTSTHuggingFace(
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            fine_tune_mode="linear_probe",
            prediction_length=7,
        )
        model._setup_model(num_features=1)

        # 大部分參數應被凍結
        frozen = sum(1 for p in model.model.parameters() if not p.requires_grad)
        trainable = sum(1 for p in model.model.parameters() if p.requires_grad)
        assert frozen > 0
        assert trainable > 0
        assert frozen > trainable  # backbone 比 head 大得多


class TestPretrainedSaveLoad:
    """測試預訓練模型的 save/load"""

    def test_save_load_preserves_pretrained_metadata(self, long_dummy_data, tmp_path):
        model = PatchTSTHuggingFace(
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            fine_tune_mode="full",
            prediction_length=7,
        )
        model.fit(long_dummy_data, long_dummy_data["Close"], num_epochs=1)
        preds_before = model.predict(long_dummy_data)

        # Save
        save_path = str(tmp_path / "pretrained_model")
        assert model.save_model(save_path) is True

        # Load
        model2 = PatchTSTHuggingFace(prediction_length=7)
        assert model2.load_model(save_path) is True
        assert model2.pretrained_model_name_or_path == PRETRAINED_MODEL
        assert model2.fine_tune_mode == "full"

        # Predictions should match
        preds_after = model2.predict(long_dummy_data)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-4)
