"""
基礎模型介面

定義所有時間序列預測模型的標準介面
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """模型類型枚舉"""

    SKLEARN_BASED = "sklearn_based"
    TRANSFORMER_BASED = "transformer_based"
    DEEP_LEARNING = "deep_learning"


class BaseModel(ABC):
    """所有預測模型的基礎抽象類別"""

    def __init__(
        self,
        model_name: str = "BaseModel",
        model_type: ModelType = ModelType.SKLEARN_BASED,
    ):
        """
        初始化基礎模型

        Args:
            model_name: 模型名稱
            model_type: 模型類型
        """
        self.model_name = model_name
        self.model_type = model_type
        self.is_fitted = False
        self.model_params: Dict[str, Any] = {}
        logger.info(f"{model_name} 模型已初始化 (類型: {model_type.value})")

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        training_config: Optional[Any] = None,
    ) -> "BaseModel":
        """
        訓練模型

        Args:
            X: 訓練特徵資料
            y: 訓練目標資料
            validation_data: 驗證資料 (X_val, y_val)
            training_config: 訓練配置 (TrainingConfig)

        Returns:
            訓練完成的模型實例
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """
        進行預測

        Args:
            X: 輸入特徵資料
            horizon: 預測時間範圍

        Returns:
            預測結果數組
        """
        pass

    @abstractmethod
    def predict_with_uncertainty(
        self, X: pd.DataFrame, horizon: int = 1, confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        進行帶不確定性的預測

        Args:
            X: 輸入特徵資料
            horizon: 預測時間範圍
            confidence_level: 信賴區間水準

        Returns:
            包含預測值和不確定性區間的字典
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        取得模型資訊

        Returns:
            模型資訊字典
        """
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "is_fitted": self.is_fitted,
            "model_params": self.model_params,
        }

    def save_model(self, filepath: str) -> bool:
        """
        儲存模型

        子類別必須實作具體儲存邏輯。基底不可回傳 True 謊報成功
        （見 .claude/rules/fail-loud.md）。

        Args:
            filepath: 儲存路徑

        Returns:
            是否儲存成功

        Raises:
            NotImplementedError: 子類別未實作時
        """
        raise NotImplementedError(
            f"{self.model_name} 未實作 save_model();子類別需提供具體儲存邏輯。"
        )

    def load_model(self, filepath: str) -> bool:
        """
        載入模型

        子類別必須實作具體載入邏輯。基底不可回傳 True 謊報成功。

        Args:
            filepath: 模型檔案路徑

        Returns:
            是否載入成功

        Raises:
            NotImplementedError: 子類別未實作時
        """
        raise NotImplementedError(
            f"{self.model_name} 未實作 load_model();子類別需提供具體載入邏輯。"
        )

    def fit_panel(
        self,
        datasets: List[Tuple[pd.DataFrame, pd.Series]],
        training_config: Optional[Any] = None,
    ) -> "BaseModel":
        """Train a single global model on sequences pooled across many series.

        Each ``(X, y)`` in ``datasets`` is one symbol; sequences are extracted
        per symbol (never across symbol boundaries) and concatenated into one
        training set. Requires comparable targets across symbols (use
        log-return target). Default raises — only models that support panel
        training override this.

        Args:
            datasets: List of (features, target) pairs, one per symbol.
            training_config: Optional training configuration.

        Returns:
            The fitted model.

        Raises:
            NotImplementedError: If the model does not support panel training.
        """
        raise NotImplementedError(
            f"{self.model_name} 尚不支援 panel 訓練 (fit_panel)；目前僅 patchtst_sklearn 支援。"
        )

    def _aggregate_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute standard forecast metrics for an (actual, predicted) pair.

        Returns ``mse``/``mae``/``rmse``/``mape``/``mda``/``direction_accuracy``;
        ``mase`` is added only when ``y_train`` is provided.

        Args:
            actual: Actual values.
            predicted: Predicted values (same length as ``actual``).
            y_train: Training-set values, used to scale MASE. When ``None``,
                MASE is omitted from the result.

        Returns:
            Dictionary of metric name to value.
        """
        # Imported lazily to avoid a models -> prediction import cycle at load time.
        from ..prediction.metrics import mase as _mase, mda as _mda

        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        errors = actual - predicted
        mse = float(np.mean(errors**2))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))

        # MAPE — guard against division by zero
        nonzero = actual != 0
        if np.any(nonzero):
            mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100)
        else:
            mape = 0.0

        dir_acc = _mda(actual, predicted)

        metrics: Dict[str, float] = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "mda": dir_acc,
            "direction_accuracy": dir_acc,
        }

        if y_train is not None:
            metrics["mase"] = _mase(actual, predicted, np.asarray(y_train, dtype=float))

        return metrics

    def evaluate_single_shot(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_train: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Single-shot evaluation.

        Calls ``predict()`` once over ``X`` and compares the output with the
        last ``len(predictions)`` values of ``y_true``.

        Note: statistically limited (uses only ``pred_len`` points) and carries
        a temporal-alignment caveat — prefer :meth:`evaluate_rolling` when the
        test window is large enough.

        Args:
            X: Test feature data.
            y_true: Actual target values.
            y_train: Training-set targets, used to scale MASE (optional).

        Returns:
            Dictionary of evaluation metrics.

        Raises:
            ValueError: When the model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")

        predictions = np.asarray(self.predict(X), dtype=float)
        actual = np.asarray(y_true, dtype=float)[-len(predictions) :]

        y_train_arr = None if y_train is None else np.asarray(y_train, dtype=float)
        return self._aggregate_metrics(actual, predictions, y_train_arr)

    def evaluate_rolling(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        seq_len: int,
        pred_len: int,
        y_train: Optional[pd.Series] = None,
        step: int = 1,
    ) -> Dict[str, Any]:
        """Rolling-origin evaluation over the test set.

        Slides an origin across the test set: at each origin the previous
        ``seq_len`` values form the context fed to ``predict()``, and the next
        ``pred_len`` values are compared against the prediction. This produces
        many (predicted, actual) pairs for statistically robust metrics, plus
        per-horizon error breakdowns.

        Args:
            X: Test feature data.
            y_true: Actual target values aligned with ``X``.
            seq_len: Context window length fed to the model.
            pred_len: Forecast horizon evaluated at each origin.
            y_train: Training-set targets, used to scale MASE (optional).
            step: Number of steps the origin advances each iteration.

        Returns:
            Dict with ``aggregate`` (flattened metrics + ``n_origins``),
            ``per_horizon`` (1-indexed h -> {rmse, mae}), and ``h1_actual`` /
            ``h1_predicted`` arrays (one value per origin, for visualization).

        Raises:
            ValueError: When ``len(X) < seq_len + pred_len``.
        """
        n = len(X)
        if n < seq_len + pred_len:
            raise ValueError(
                f"Data length ({n}) < seq_len ({seq_len}) + pred_len ({pred_len})"
            )

        origins = list(range(seq_len, n - pred_len + 1, step))
        n_origins = len(origins)

        all_pred = np.zeros((n_origins, pred_len))
        all_actual = np.zeros((n_origins, pred_len))
        y_values = np.asarray(y_true, dtype=float)

        for i, origin in enumerate(origins):
            X_window = X.iloc[origin - seq_len : origin]
            preds = np.asarray(self.predict(X_window, horizon=pred_len), dtype=float)
            # Normalise to exactly pred_len values
            if len(preds) >= pred_len:
                preds = preds[:pred_len]
            else:
                preds = np.pad(preds, (0, pred_len - len(preds)), mode="edge")
            all_pred[i] = preds
            all_actual[i] = y_values[origin : origin + pred_len]

        # Per-horizon metrics (column-wise across origins)
        per_horizon: Dict[int, Dict[str, float]] = {}
        for h in range(pred_len):
            errors = all_actual[:, h] - all_pred[:, h]
            per_horizon[h + 1] = {
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
            }

        # Aggregate over all (predicted, actual) pairs
        y_train_arr = None if y_train is None else np.asarray(y_train, dtype=float)
        aggregate = self._aggregate_metrics(
            all_actual.flatten(), all_pred.flatten(), y_train_arr
        )
        aggregate["n_origins"] = n_origins

        return {
            "aggregate": aggregate,
            "per_horizon": per_horizon,
            "h1_actual": all_actual[:, 0],
            "h1_predicted": all_pred[:, 0],
        }


class TimeSeriesModel(BaseModel):
    """時間序列模型的特殊基礎類別"""

    def __init__(self, model_name: str = "TimeSeriesModel"):
        super().__init__(model_name)
        self.sequence_length: Optional[int] = None
        self.feature_columns: list[str] = []

    def prepare_sequences(
        self, data: pd.DataFrame, sequence_length: int, target_column: str = "Close"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        準備時間序列資料

        Args:
            data: 時間序列資料
            sequence_length: 序列長度
            target_column: 目標欄位名稱

        Returns:
            (X, y) 序列資料元組
        """
        self.sequence_length = sequence_length

        # 確保資料按時間順序排列
        data = data.sort_index()

        X, y = [], []

        for i in range(len(data) - sequence_length):
            # 輸入序列
            seq_x = data.iloc[i : i + sequence_length].values
            X.append(seq_x)

            # 目標值
            if target_column in data.columns:
                seq_y = data.iloc[i + sequence_length][target_column]
            else:
                seq_y = data.iloc[i + sequence_length, 0]  # 使用第一欄
            y.append(seq_y)

        return np.array(X), np.array(y)

    def validate_input_shape(self, X: pd.DataFrame) -> bool:
        """
        驗證輸入資料形狀

        Args:
            X: 輸入資料

        Returns:
            是否符合要求
        """
        if not self.is_fitted:
            logger.warning("模型尚未訓練")
            return False

        if self.sequence_length is None:
            logger.warning("序列長度未設定")
            return False

        if len(X) < self.sequence_length:
            logger.warning(
                f"輸入資料長度 {len(X)} 小於所需序列長度 {self.sequence_length}"
            )
            return False

        return True


class SklearnBasedModel(TimeSeriesModel):
    """
    基於 Sklearn 的時間序列模型基類
    """

    def __init__(self, model_name: str = "SklearnBasedModel", **kwargs):
        super().__init__(model_name)
        self.model_type = ModelType.SKLEARN_BASED
        self.scaler = None
        self.model = None


class TransformerBasedModel(TimeSeriesModel):
    """
    基於 Transformer 的時間序列模型基類
    """

    def __init__(self, model_name: str = "TransformerBasedModel", **kwargs):
        super().__init__(model_name)
        self.model_type = ModelType.TRANSFORMER_BASED
        self.tokenizer = None
        self.model = None
        self.device = None

    @abstractmethod
    def prepare_data_for_transformer(self, data) -> Dict[str, Any]:
        """
        為 Transformer 模型準備資料

        Args:
            data: 原始時間序列資料

        Returns:
            準備好的資料字典
        """
        pass

    @abstractmethod
    def setup_model(self, **model_kwargs):
        """
        設置 Transformer 模型

        Args:
            **model_kwargs: 模型配置參數
        """
        pass
