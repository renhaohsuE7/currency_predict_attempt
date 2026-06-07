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
from sklearn.model_selection import KFold, cross_val_predict


def _log_returns(close: pd.Series) -> np.ndarray:
    return np.asarray(np.log(close.astype(float)).diff(), dtype=float)


def realized_volatility_target(close: pd.Series, horizon: int) -> pd.Series:
    """Future realized volatility at each t = std of log-returns over (t, t+horizon].

    The last `horizon` rows are NaN (no complete future window).
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
        self,
        X: pd.DataFrame,
        vol_y: pd.Series,
        dir_y: pd.Series,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Out-of-fold predictions to avoid factor leakage in Stage-2 training."""
        kf = KFold(n_splits=k, shuffle=False)
        Xn = X.to_numpy()
        vol_oof = cross_val_predict(
            HistGradientBoostingRegressor(random_state=self.random_state),
            Xn,
            np.asarray(vol_y, dtype=float),
            cv=kf,
        )
        dir_oof = cross_val_predict(
            HistGradientBoostingClassifier(random_state=self.random_state),
            Xn,
            np.asarray(dir_y, dtype=int),
            cv=kf,
            method="predict_proba",
        )
        classes = sorted(set(int(v) for v in dir_y))
        up_idx = classes.index(1) if 1 in classes else dir_oof.shape[1] - 1
        return np.asarray(vol_oof, dtype=float), np.asarray(
            dir_oof[:, up_idx], dtype=float
        )
