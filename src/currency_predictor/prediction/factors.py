"""Factor targets and Stage-1 factor models for cascade forecasting.

Factor targets are FUTURE quantities (shifted), so a model trained to predict
them uses only past context — no look-ahead within the target itself.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold


def _log_returns(close: pd.Series) -> np.ndarray:
    return np.asarray(np.log(close.astype(float)).diff(), dtype=float)


def _require_no_nan(arr: np.ndarray, name: str) -> None:
    if np.isnan(np.asarray(arr, dtype=float)).any():
        raise ValueError(f"{name} 含 NaN;呼叫端必須先移除 factor target 尾端的 NaN 列")


def realized_volatility_target(close: pd.Series, horizon: int) -> pd.Series:
    """Future realized volatility at each t = std of log-returns over (t, t+horizon].

    The last `horizon` rows are NaN (no complete future window).
    std uses population std (ddof=0).
    """
    r = _log_returns(close)
    n = len(r)
    out = np.full(n, np.nan)
    for t in range(n - horizon):
        out[t] = float(np.std(r[t + 1 : t + 1 + horizon]))
    return pd.Series(out, index=close.index, name="vol_target")


def direction_target(close: pd.Series, horizon: int) -> pd.Series:
    """Future direction label at each t: 1.0 if cumulative log-return over
    (t, t+horizon] > 0 else 0.0. Last `horizon` rows are NaN.
    """
    r = _log_returns(close)
    n = len(r)
    out = np.full(n, np.nan)
    for t in range(n - horizon):
        out[t] = 1.0 if float(np.sum(r[t + 1 : t + 1 + horizon])) > 0 else 0.0
    return pd.Series(out, index=close.index, name="dir_target")


class FactorModel:
    """Stage-1 factor predictors: a regressor for realized volatility and a
    classifier for direction (returns P(up))."""

    def __init__(self, random_state: int = 42) -> None:
        self.vol_model = HistGradientBoostingRegressor(random_state=random_state)
        self.dir_model = HistGradientBoostingClassifier(random_state=random_state)
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, vol_y: pd.Series, dir_y: pd.Series) -> "FactorModel":
        """Fit volatility regressor and direction classifier on full data."""
        _require_no_nan(np.asarray(vol_y, dtype=float), "vol_y")
        _require_no_nan(np.asarray(dir_y, dtype=float), "dir_y")
        self.vol_model.fit(X.to_numpy(), np.asarray(vol_y, dtype=float))
        self.dir_model.fit(X.to_numpy(), np.asarray(dir_y, dtype=int))
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (vol_pred, dir_prob_up) arrays for input X."""
        vol_pred = np.asarray(self.vol_model.predict(X.to_numpy()), dtype=float)
        proba = self.dir_model.predict_proba(X.to_numpy())
        classes = list(self.dir_model.classes_)
        up_idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
        dir_pred = np.asarray(proba[:, up_idx], dtype=float)
        return vol_pred, dir_pred

    def crossfit_predict(
        self, X: pd.DataFrame, vol_y: pd.Series, dir_y: pd.Series, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Out-of-fold predictions to avoid factor leakage in Stage-2 training.

        Single-class direction folds are handled gracefully (held-out rows get the
        train fold's base rate) rather than crashing.
        """
        Xn = X.to_numpy()
        voly = np.asarray(vol_y, dtype=float)
        diry = np.asarray(dir_y, dtype=int)
        _require_no_nan(np.asarray(vol_y, dtype=float), "vol_y")
        _require_no_nan(np.asarray(dir_y, dtype=float), "dir_y")
        n = len(Xn)
        vol_oof = np.full(n, np.nan)
        dir_oof = np.full(n, np.nan)
        for tr, te in KFold(n_splits=k, shuffle=False).split(Xn):
            reg = HistGradientBoostingRegressor(random_state=self.random_state)
            reg.fit(Xn[tr], voly[tr])
            vol_oof[te] = reg.predict(Xn[te])
            if len(np.unique(diry[tr])) < 2:
                # single-class train fold → base rate (graceful, no crash)
                dir_oof[te] = float(diry[tr].mean())
            else:
                clf = HistGradientBoostingClassifier(random_state=self.random_state)
                clf.fit(Xn[tr], diry[tr])
                classes = list(clf.classes_)
                up_idx = classes.index(1) if 1 in classes else len(classes) - 1
                dir_oof[te] = clf.predict_proba(Xn[te])[:, up_idx]
        return vol_oof, dir_oof
