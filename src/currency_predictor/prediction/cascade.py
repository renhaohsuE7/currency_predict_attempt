"""Cascade factor-augmented forecasting.

Stage-1 predicts future realized volatility and direction (leak-free via
cross-fitting); these are injected as extra feature channels; Stage-2 (any
PatchTST backend) forecasts price/return from the augmented features.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..models.factory import ModelFactory
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

    def _chrono_split(self, X: pd.DataFrame, y: pd.Series):
        """Chronological train/test split with a fail-loud window guard.

        Mirrors CurrencyPredictor.prepare_training_data: the test window must be
        >= seq_len + pred_len, otherwise the windowed model cannot be evaluated
        (see .claude/rules/fail-loud.md).
        """
        seq_len = int(
            self.model_params.get("seq_len")
            or self.model_params.get("context_length")
            or 64
        )
        required = seq_len + self.horizon
        test_days = self._test_days or max(2 * self.horizon, 30)
        effective = min(test_days, len(X) // 2)
        if effective < required:
            raise ValueError(
                f"cascade 測試視窗太小無法評估:effective_test={effective} "
                f"< seq_len + pred_len = {seq_len} + {self.horizon} = {required};"
                f"請將 test_days 設為 >= {required} 或加長資料。"
            )
        split = len(X) - effective
        return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]

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
        X_train, y_train, _, _ = self._chrono_split(X, y)
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

    def evaluate_factor_lift(self, symbol: str, period: str = "2y") -> Dict[str, Any]:
        """Honest factor-lift: cascade (with factors) vs same backend without
        factors, on the same chronological split, plus Stage-1 factor quality."""
        processed = self.predictor.build_processed_data(symbol, period)
        augmented = self._augment(processed, training=True)
        seq_len = int(self.model_params.get("seq_len", 32))
        pred_len = self.horizon

        def split_xy(frame: pd.DataFrame):
            y = self.predictor.data_processor.to_log_returns(frame["Close"])
            mask = y.notna()
            X = frame[[c for c in frame.columns if c != "Close"]][mask]
            return self._chrono_split(X, y[mask])

        def fit_eval(frame: pd.DataFrame) -> Dict[str, float]:
            Xtr, ytr, Xte, yte = split_xy(frame)
            m = ModelFactory.create_model(self.backend, **self.model_params)
            m.fit(Xtr, ytr, training_config=TrainingConfig())
            if len(Xte) >= seq_len + pred_len:
                rolling = m.evaluate_rolling(Xte, yte, seq_len, pred_len, y_train=ytr)
                return dict(rolling["aggregate"])
            return dict(m.evaluate_single_shot(Xte, yte, y_train=ytr))

        plain = augmented[
            [c for c in augmented.columns if c not in ("Factor_Vol", "Factor_Dir")]
        ]
        with_f = fit_eval(augmented)
        without_f = fit_eval(plain)

        vol_t = realized_volatility_target(processed["Close"], pred_len)
        dir_t = direction_target(processed["Close"], pred_len)
        valid = vol_t.notna() & dir_t.notna()
        vol_rmse = float(
            np.sqrt(np.mean((augmented["Factor_Vol"][valid] - vol_t[valid]) ** 2))
        )
        dir_acc = float(
            np.mean(
                (augmented["Factor_Dir"][valid] > 0.5).astype(float).values
                == dir_t[valid].values
            )
        )
        return {
            "with_factors": with_f,
            "without_factors": without_f,
            "vol_rmse": vol_rmse,
            "dir_accuracy": dir_acc,
        }
