"""Regression test for the Lightning predict scaler fix.

Before the fix, PatchTSTLightningWrapper.predict fed RAW (unscaled) values to a
model trained on standardized data and never inverse-transformed the output, so
predictions collapsed to ~0 (RMSE ≈ the price level). This asserts predictions
land in the data's own scale.
"""

import numpy as np
import pandas as pd
import pytest

from currency_predictor.models.patchtst import PatchTSTLightningWrapper


@pytest.mark.skipif(PatchTSTLightningWrapper is None, reason="pytorch-lightning 未安裝")
def test_lightning_predict_in_data_scale(tmp_path):
    # 造一段水準 ~30 的序列(模擬 USDTWD),用固定種子可重現
    rng = np.random.RandomState(0)
    n = 120
    close = 30.0 + np.cumsum(rng.normal(0, 0.05, n))
    df = pd.DataFrame({"Close": close})

    model = PatchTSTLightningWrapper(
        seq_len=20, pred_len=3, patch_len=4, stride=2, max_epochs=2,
    )
    model.fit(df, df["Close"], num_epochs=2, output_dir=str(tmp_path / "lt"))

    pred = np.asarray(model.predict(df[["Close"]])).ravel()

    assert pred.shape[0] >= 1
    lo, hi = close.min() - 5.0, close.max() + 5.0
    # 修好 scaler 後預測應落在資料水準附近,而非崩到 ~0
    assert np.all((pred > lo) & (pred < hi)), f"預測 {pred} 不在資料水準 [{lo:.2f}, {hi:.2f}]"
    assert np.all(pred > 10.0), f"預測 {pred} 疑似仍在標準化空間(~0),scaler 未還原"


@pytest.mark.skipif(PatchTSTLightningWrapper is None, reason="pytorch-lightning 未安裝")
def test_revin_normalize_is_level_invariant():
    """RevIN 純函式驗證:同形狀、不同水位的兩個視窗,正規化後應幾乎相同
    (水位錨已被移除);且以最後值為中心 → 正規化後**最後一點≈0**。純 numpy。"""
    base = np.array([10, 11, 12, 11, 13, 12, 14, 13], dtype=np.float32).reshape(1, 8, 1)
    shifted = base + 1000.0   # 同形狀,水位 +1000
    n_base, _ = PatchTSTLightningWrapper._revin_normalize(base)
    n_shift, _ = PatchTSTLightningWrapper._revin_normalize(shifted)
    assert np.allclose(n_base, n_shift, atol=1e-4), "RevIN 後仍受水位影響,未移除 level"
    assert abs(float(n_base[0, -1, 0])) < 1e-4, "RevIN 未以視窗最後值為中心(最後一點應≈0)"


@pytest.mark.skipif(PatchTSTLightningWrapper is None, reason="pytorch-lightning 未安裝")
def test_lightning_predict_tracks_recent_level_not_training_level(tmp_path):
    """Plan B 核心驗證:在「低價位」訓練,對「高價位」視窗預測 → 預測應貼著
    高價位,而非被拉回訓練期價位(level-anchoring 的根治)。

    舊全域 scaler 會把高價位視窗 transform 成超出訓練分布的 z,模型輸出再 inverse
    回 ~訓練水準 → 預測崩回低價位(本測試會 fail)。RevIN 以視窗自身水準為中心,
    預測貼著輸入水位。
    """
    rng = np.random.RandomState(0)
    n = 160
    close = 30.0 + np.cumsum(rng.normal(0, 0.05, n))   # 訓練序列在 ~30 水準
    df = pd.DataFrame({"Close": close})

    model = PatchTSTLightningWrapper(
        seq_len=20, pred_len=3, patch_len=4, stride=2, max_epochs=3,
    )
    model.fit(df, df["Close"], num_epochs=3, output_dir=str(tmp_path / "lt"))

    # 推論視窗整體 +20 平移到 ~50 水準(形狀相同,只是水位高很多)
    recent_high = close[-20:] + 20.0
    pred = np.asarray(model.predict(pd.DataFrame({"Close": recent_high}))).ravel()

    assert pred.mean() > 45.0, f"預測均值 {pred.mean():.2f} 被錨回訓練價位(~30),RevIN 未生效"
    assert pred.mean() < 55.0, f"預測均值 {pred.mean():.2f} 偏離輸入水位(~50)過多"
