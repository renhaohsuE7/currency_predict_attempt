"""PatchTSTSklearn must produce a real multi-step trajectory, not a flat line.

Before the fix, fit trained a single-output GBR on the mean of the next pred_len
values and predict returned np.full(pred_len, value[0]) — a constant. This pins
the corrected behavior: predict returns pred_len distinct-capable values.
"""
import numpy as np
import pandas as pd

from currency_predictor.models.patchtst import PatchTSTSklearn


def _trending_df(n=120):
    # a clear upward trend so consecutive horizon steps genuinely differ
    close = 30.0 + 0.05 * np.arange(n) + 0.3 * np.sin(np.arange(n) / 5.0)
    return pd.DataFrame({"feat": close, "Close": close})


def test_sklearn_predict_is_multistep_not_constant():
    df = _trending_df()
    m = PatchTSTSklearn(seq_len=20, pred_len=5, patch_len=4, stride=2)
    m.fit(df[["feat"]], df["Close"])
    pred = np.asarray(m.predict(df[["feat", "Close"]]))
    assert pred.shape[0] == 5
    # the whole point of the fix: NOT a flat line
    assert not np.allclose(pred, pred[0]), f"predict still constant: {pred}"


def test_sklearn_predict_respects_horizon():
    df = _trending_df()
    m = PatchTSTSklearn(seq_len=20, pred_len=5, patch_len=4, stride=2)
    m.fit(df[["feat"]], df["Close"])
    pred = np.asarray(m.predict(df[["feat", "Close"]], horizon=3))
    assert pred.shape[0] == 3
