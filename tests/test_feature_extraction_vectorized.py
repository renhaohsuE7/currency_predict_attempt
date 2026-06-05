"""Characterization tests: vectorized _extract_features_from_data must match
the original per-window loop element-for-element.

Ground truth is reconstructed from the model's OWN per-window helpers
(_create_patches + _extract_patch_features), which are unchanged and still used
by predict().
"""

import numpy as np
import pandas as pd
import pytest

from currency_predictor.models.patchtst import PatchTST


def _loop_reference(model, X: pd.DataFrame, y: pd.Series):
    """Original loop implementation, using the model's unchanged helpers."""
    X_seq, y_seq = [], []
    for i in range(len(X) - model.seq_len - model.pred_len + 1):
        X_seq.append(X.iloc[i : i + model.seq_len].values)
        y_seq.append(y.iloc[i + model.seq_len : i + model.seq_len + model.pred_len].values)
    X_sequences = np.array(X_seq)
    feats = np.array(
        [model._extract_patch_features(model._create_patches(s)) for s in X_sequences]
    )
    return feats, np.array(y_seq)


def _data(n, n_feat, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {f"f{c}": rng.standard_normal(n) for c in range(n_feat)}
    )
    y = pd.Series(rng.standard_normal(n))
    return X, y


@pytest.mark.parametrize(
    "seq_len,pred_len,patch_len,stride,n_feat,n",
    [
        (10, 3, 5, 2, 3, 60),
        (16, 5, 4, 4, 1, 80),
        (20, 4, 6, 3, 5, 100),
        (32, 15, 8, 4, 7, 200),
        (12, 2, 3, 1, 2, 50),
    ],
)
def test_vectorized_matches_loop(seq_len, pred_len, patch_len, stride, n_feat, n):
    model = PatchTST(
        seq_len=seq_len, pred_len=pred_len, patch_len=patch_len, stride=stride
    )
    X, y = _data(n, n_feat, seed=seq_len + n_feat)

    feats_new, tgts_new = model._extract_features_from_data(X, y)
    feats_old, tgts_old = _loop_reference(model, X, y)

    assert feats_new.shape == feats_old.shape, (feats_new.shape, feats_old.shape)
    assert tgts_new.shape == tgts_old.shape
    np.testing.assert_allclose(feats_new, feats_old, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(tgts_new, tgts_old, rtol=1e-12, atol=1e-12)


def test_insufficient_data_raises():
    model = PatchTST(seq_len=20, pred_len=5, patch_len=5, stride=2)
    X, y = _data(20, 3, seed=1)  # 20 < seq_len + pred_len = 25
    with pytest.raises(ValueError, match="資料不足"):
        model._extract_features_from_data(X, y)


def test_trained_model_predicts_after_vectorized_extract():
    """End-to-end sanity: fit (which uses the vectorized path) then predict."""
    model = PatchTST(seq_len=16, pred_len=4, patch_len=4, stride=4)
    X, y = _data(120, 3, seed=7)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 4
    assert np.all(np.isfinite(preds))
