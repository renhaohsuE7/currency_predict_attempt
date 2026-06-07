"""Naive baseline model (persistence).

A persistence forecaster that predicts the last observed ``Close`` value for the
entire horizon. It is the reference baseline for MASE — a real model is only
useful when its MASE is below 1.0 (i.e. it beats naive).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseModel, ModelType


class NaiveModel(BaseModel):
    """Persistence baseline: predict the last ``Close`` value, repeated."""

    def __init__(
        self,
        pred_len: int = 24,
        target_transform: str = "price",
        **kwargs: Any,
    ) -> None:
        """Initialise the naive model.

        Args:
            pred_len: Default forecast horizon (number of repeated values).
            target_transform: Target space of the forecast. ``"price"`` repeats
                the last ``Close`` value (price persistence); ``"log_return"``
                predicts zero return (the return-space equivalent of price
                persistence) so the baseline matches a return-space target.
            **kwargs: Ignored; accepted for factory-call compatibility.
        """
        super().__init__(model_name="NaiveModel", model_type=ModelType.SKLEARN_BASED)
        self.pred_len = pred_len
        self.target_transform = target_transform
        self.model_params = {
            "pred_len": pred_len,
            "target_transform": target_transform,
        }
        self.training_history: Dict[str, Any] = {"epochs": 0}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        training_config: Optional[Any] = None,
    ) -> "NaiveModel":
        """No-op fit — the naive model has no parameters to learn.

        Returns:
            ``self`` (marked as fitted).
        """
        self.is_fitted = True
        self.training_history = {"epochs": 0}
        return self

    def _last_value(self, X: pd.DataFrame) -> float:
        """Return the last ``Close`` value (or first numeric column as fallback).

        Raises:
            ValueError: When ``X`` has no usable numeric column.
        """
        if "Close" in X.columns:
            series = X["Close"]
        else:
            numeric = X.select_dtypes(include=[np.number])
            if numeric.shape[1] == 0:
                raise ValueError("輸入資料沒有可用的數值欄位")
            series = numeric.iloc[:, 0]
        return float(series.iloc[-1])

    def predict(self, X: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Predict the last observed value, repeated for the horizon.

        Args:
            X: Input feature data (uses the ``Close`` column).
            horizon: When greater than 1, overrides ``pred_len``.

        Returns:
            Array of repeated last-value predictions.

        Raises:
            RuntimeError: When the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未訓練 (model not fitted)")

        steps = horizon if horizon and horizon > 1 else self.pred_len
        if self.target_transform == "log_return":
            # Return-space persistence: zero return = price stays the same.
            return np.zeros(steps, dtype=float)
        return np.full(steps, self._last_value(X), dtype=float)

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        horizon: int = 1,
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """Return point predictions with zero uncertainty.

        A naive model has no notion of predictive variance, so the bounds
        coincide with the point predictions.
        """
        preds = self.predict(X, horizon=horizon)
        return {
            "predictions": preds,
            "std": np.zeros_like(preds),
            "upper_bound": preds.copy(),
            "lower_bound": preds.copy(),
        }

    def save_model(self, filepath: str) -> bool:
        """No real persistence needed for a parameterless model."""
        return True

    def load_model(self, filepath: str) -> bool:
        """Loading is a no-op; mark the model as ready to predict."""
        self.is_fitted = True
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata including ``pred_len`` and a description."""
        info = super().get_model_info()
        info["pred_len"] = self.pred_len
        info["target_transform"] = self.target_transform
        info["description"] = "Naive persistence baseline (repeats last Close)"
        return info
