"""Cascade factor-augmented forecasting.

Stage-1 predicts future realized volatility and direction (leak-free via
cross-fitting); these are injected as extra feature channels; Stage-2 (any
PatchTST backend) forecasts price/return from the augmented features.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..models.patchtst.config import TrainingConfig
from .factors import FactorModel, direction_target, realized_volatility_target
from .predictor import CurrencyPredictor

logger = logging.getLogger(__name__)

_MULTI_CHANNEL_BACKENDS = {"patchtst_huggingface", "patchtst_lightning"}


class CascadePredictor:
    """Two-stage cascade: factors (vol, dir) -> price/return."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cascade_cfg = config.get("cascade", {})
        self.backend = self.cascade_cfg.get("stage2_backend", "patchtst_sklearn")
        self.k = int(self.cascade_cfg.get("crossfit_folds", 5))
        self.model_params = dict(config.get("model_params", {}))
        self.horizon = int(self.model_params.get("pred_len", 15))
        self.target_transform = config.get("model_training", {}).get(
            "target_transform", "log_return"
        )
        if self.backend in _MULTI_CHANNEL_BACKENDS:
            self.model_params["use_multi_channel"] = True
        self.predictor = CurrencyPredictor(
            model_name=self.backend,
            model_params=self.model_params,
            data_storage_path=config.get("data_storage_path", "data"),
            capm_config=config.get("capm", {}),
            target_transform=self.target_transform,
        )
        self.factor_model = FactorModel()
        self._test_days = config.get("model_training", {}).get("test_days")
        self._period = "2y"

    def _augment(self, processed: pd.DataFrame, training: bool) -> pd.DataFrame:
        close = processed["Close"]
        vol_t = realized_volatility_target(close, self.horizon)
        dir_t = direction_target(close, self.horizon)
        feat_cols = [
            c
            for c in processed.select_dtypes(include=[np.number]).columns
            if c not in ("Close", "Factor_Vol", "Factor_Dir")
        ]
        Xf = processed[feat_cols]
        valid = vol_t.notna() & dir_t.notna()

        vol_series = pd.Series(np.nan, index=processed.index)
        dir_series = pd.Series(np.nan, index=processed.index)
        if training:
            vol_oof, dir_oof = self.factor_model.crossfit_predict(
                Xf[valid], vol_t[valid], dir_t[valid], k=self.k
            )
            vol_series.loc[valid] = vol_oof
            dir_series.loc[valid] = dir_oof
            self.factor_model.fit(Xf[valid], vol_t[valid], dir_t[valid])
        need = vol_series.isna()
        if need.any():
            v_pred, d_pred = self.factor_model.predict(Xf[need])
            vol_series.loc[need] = v_pred
            dir_series.loc[need] = d_pred

        out = processed.copy()
        out["Factor_Vol"] = vol_series.values
        out["Factor_Dir"] = dir_series.values
        return out

    def fit(
        self, symbol: str, period: str = "2y", **train_kwargs: Any
    ) -> Dict[str, Any]:
        self._period = period
        processed = self.predictor.build_processed_data(symbol, period)
        augmented = self._augment(processed, training=True)

        target_col = "Close"
        feature_cols = [c for c in augmented.columns if c != target_col]
        X = augmented[feature_cols]
        if self.target_transform == "log_return":
            y = self.predictor.data_processor.to_log_returns(augmented[target_col])
            mask = y.notna()
            X, y = X[mask], y[mask]
        else:
            y = augmented[target_col]
        test_days = self._test_days or max(2 * self.horizon, 30)
        split = len(X) - min(test_days, len(X) // 2)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        self.predictor.model.fit(X_train, y_train, training_config=TrainingConfig())
        self.predictor._target_column = target_col
        return {"symbol": symbol, "n_train": int(len(X_train)), "trained": True}

    def predict(
        self,
        symbol: str,
        horizon: Optional[int] = None,
        period: Optional[str] = None,
    ) -> Dict[str, Any]:
        horizon = horizon or self.horizon
        processed = self.predictor.build_processed_data(symbol, period or self._period)
        augmented = self._augment(processed, training=False)
        last = augmented.iloc[-1]
        feature_cols = [c for c in augmented.columns if c != "Close"]
        preds = self.predictor.model.predict(augmented[feature_cols], horizon)
        preds = np.asarray(preds, dtype=float)
        if self.target_transform == "log_return":
            last_close = float(augmented["Close"].iloc[-1])
            preds = self.predictor.data_processor.from_log_returns(last_close, preds)
        return {
            "symbol": symbol,
            "price": preds,
            "vol": float(last["Factor_Vol"]),
            "dir": float(last["Factor_Dir"]),
        }
