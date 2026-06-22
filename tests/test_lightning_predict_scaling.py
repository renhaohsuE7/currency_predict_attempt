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
