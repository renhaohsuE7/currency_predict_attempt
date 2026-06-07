"""
PatchTST PyTorch Lightning 封裝

提供與 sklearn 和 HuggingFace 版本一致的介面
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import joblib

from ...base import TransformerBasedModel, ModelType
from ..config import PatchTSTConfig, TrainingConfig

logger = logging.getLogger(__name__)

# 嘗試導入 Lightning
try:
    import pytorch_lightning as pl
    from .lightning_module import PatchTSTLightning, get_lightning_callbacks
    from .modules import PatchTSTModel

    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None  # type: ignore[assignment]
    logger.warning("pytorch_lightning 未安裝，PatchTSTLightningWrapper 將無法使用")


if HAS_LIGHTNING:

    class MetricsHistoryCallback(pl.Callback):  # type: ignore[name-defined]
        """累積每個 epoch 的 train/val loss 歷史"""

        def __init__(self):
            super().__init__()
            self.train_losses: list = []
            self.val_losses: list = []

        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics.get("train_loss_epoch")
            if train_loss is not None:
                self.train_losses.append(train_loss.item())

        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self.val_losses.append(val_loss.item())


class TimeSeriesDataset(Dataset):
    """
    時間序列資料集

    用於 PyTorch DataLoader
    """

    def __init__(
        self, past_values: torch.Tensor, future_values: Optional[torch.Tensor] = None
    ):
        self.past_values = past_values
        self.future_values = future_values

    def __len__(self) -> int:
        return len(self.past_values)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"past_values": self.past_values[idx]}
        if self.future_values is not None:
            item["future_values"] = self.future_values[idx]
        return item


class PatchTSTLightningWrapper(TransformerBasedModel):
    """
    PatchTST PyTorch Lightning 封裝

    提供與 PatchTSTSklearn 和 PatchTSTHuggingFace 一致的介面

    主要特點:
    - 完整控制訓練過程
    - 豐富的回調系統
    - 易於調試和擴展
    - 支援分散式訓練

    使用範例:
        ```python
        model = PatchTSTLightningWrapper(
            seq_len=64,
            pred_len=7,
            d_model=64,
            n_layers=2
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        ```
    """

    def __init__(
        self,
        # sklearn 風格參數
        seq_len: Optional[int] = None,
        pred_len: Optional[int] = None,
        patch_len: Optional[int] = None,
        stride: Optional[int] = None,
        # 配置物件
        config: Optional[PatchTSTConfig] = None,
        # 直接參數
        context_length: int = 64,
        prediction_length: int = 7,
        patch_length: int = 8,
        patch_stride: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        # 訓練參數
        max_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 10,
        # 其他
        random_state: int = 42,
        accelerator: str = "auto",
        devices: str = "auto",
        **kwargs,
    ):
        """
        初始化 PatchTST Lightning 封裝

        Args:
            seq_len: 輸入序列長度 (sklearn 風格別名)
            pred_len: 預測長度 (sklearn 風格別名)
            patch_len: Patch 大小 (sklearn 風格別名)
            stride: Patch 步長 (sklearn 風格別名)
            config: PatchTSTConfig 配置物件
            context_length: 輸入序列長度
            prediction_length: 預測長度
            patch_length: Patch 大小
            patch_stride: Patch 步長
            d_model: Transformer 隱藏層維度
            n_heads: 注意力頭數
            n_layers: Transformer 層數
            d_ff: 前饋網路維度
            dropout: Dropout 率
            max_epochs: 最大訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            weight_decay: 權重衰減
            early_stopping_patience: 早停耐心值
            random_state: 隨機種子
            accelerator: Lightning 加速器 ('auto', 'gpu', 'cpu')
            devices: 設備數量 ('auto', 1, 2, ...)
        """
        super().__init__(model_name="PatchTST_Lightning")

        if not HAS_LIGHTNING:
            raise ImportError(
                "pytorch_lightning 未安裝，請執行: pip install pytorch-lightning"
            )

        # 處理配置
        if config is not None:
            self.config = config
        else:
            self.config = PatchTSTConfig.from_sklearn_params(
                seq_len=seq_len,
                pred_len=pred_len,
                patch_len=patch_len,
                stride=stride,
                context_length=context_length,
                prediction_length=prediction_length,
                patch_length=patch_length,
                patch_stride=patch_stride,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout,
                random_state=random_state,
                **kwargs,
            )

        # 驗證配置
        self.config.validate()

        # 訓練配置
        self.training_config = TrainingConfig(
            num_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
        )

        # Lightning 配置
        self.accelerator = accelerator
        self.devices = devices
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 設置隨機種子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        pl.seed_everything(random_state)

        # 設備
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and accelerator != "cpu" else "cpu"
        )

        # 標準化器
        self.scaler = StandardScaler()

        # 多 channel 設定
        self.use_multi_channel = getattr(self.config, "use_multi_channel", False)
        self._target_channel_idx = 0
        self._feature_columns = None

        # 模型組件
        self.lightning_model: Optional[PatchTSTLightning] = None
        self.trainer: Optional[pl.Trainer] = None

        # 訓練歷史
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "best_val_loss": float("inf"),
        }

        # 模型參數
        self.model_params = self.config.to_dict()

        logger.info(
            f"PatchTST Lightning 初始化完成: "
            f"context_length={self.config.context_length}, "
            f"prediction_length={self.config.prediction_length}, "
            f"device={self.device}"
        )

    def _create_sequences(
        self, values: np.ndarray, context_length: int, prediction_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """創建訓練序列"""
        total_length = context_length + prediction_length
        num_sequences = len(values) - total_length + 1

        if num_sequences <= 0:
            raise ValueError(
                f"資料長度 ({len(values)}) 不足以創建序列 "
                f"(需要至少 {total_length} 筆)"
            )

        past_values = []
        future_values = []

        for i in range(num_sequences):
            past_values.append(values[i : i + context_length])
            future_values.append(
                values[i + context_length : i + context_length + prediction_length]
            )

        return np.array(past_values), np.array(future_values)

    def _to_numpy(self, data) -> np.ndarray:
        """將 pandas 或 numpy 資料統一轉為 numpy array"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.asarray(data)

    def _prepare_data(
        self, X, y=None, fit_scaler: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        準備訓練/預測資料

        支援輸入格式:
        - X shape (n_samples, context_length, n_features): 已切好的序列
        - X shape (n_timesteps, n_features) 或 (n_timesteps,): 原始時序，自動切片
        - X 可為 pandas DataFrame 或 numpy array
        """
        X_np = self._to_numpy(X)

        # Case 1: X 已是 3D (n_samples, context_length, n_features)
        if X_np.ndim == 3:
            past_values = X_np.astype(np.float32)
            if y is not None:
                y_np = self._to_numpy(y).astype(np.float32)
                # y shape: (n_samples,) or (n_samples, pred_len) or (n_samples, pred_len, 1)
                if y_np.ndim == 1:
                    future_values = y_np.reshape(-1, 1)
                elif y_np.ndim == 2:
                    future_values = y_np
                else:
                    future_values = y_np.squeeze(-1)
            else:
                raise ValueError("3D X 輸入時必須提供 y")

            return (torch.FloatTensor(past_values), torch.FloatTensor(future_values))

        # Case 2: X 是 2D 原始時序 (n_timesteps, n_features) 或 pandas DataFrame
        # 多 channel 模式：使用所有 numeric columns
        if self.use_multi_channel and isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                raise ValueError("資料中沒有數值欄位")
            target_values = X[numeric_cols].values  # (N, n_features)
            self._feature_columns = numeric_cols
            if "Close" not in numeric_cols:
                raise ValueError(
                    "multi_channel 模式找不到目標 channel 'Close':HF/Lightning "
                    "multi-channel 會預測 Close 價格 channel,不支援目標不在輸入欄位的設定"
                    "(例如 target_transform=log_return 且 X 不含 Close)。請改用 sklearn "
                    "或讓 Close 留在輸入欄位。"
                )
            self._target_channel_idx = numeric_cols.index("Close")
        elif y is not None:
            y_np = self._to_numpy(y)
            target_values = y_np.reshape(-1, 1)
        elif isinstance(X, pd.DataFrame) and "Close" in X.columns:
            target_values = X["Close"].values.reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("資料中沒有數值欄位")
            target_values = X[numeric_cols[0]].values.reshape(-1, 1)
        else:
            # numpy 2D: 用第一個特徵作為目標
            target_values = X_np[:, 0].reshape(-1, 1)

        target_values = target_values.astype(np.float32)

        # 標準化
        if fit_scaler:
            values_scaled = self.scaler.fit_transform(target_values)
        else:
            values_scaled = self.scaler.transform(target_values)

        # 創建序列
        past_values, future_values = self._create_sequences(
            values_scaled, self.config.context_length, self.config.prediction_length
        )

        return (torch.FloatTensor(past_values), torch.FloatTensor(future_values))

    def prepare_data_for_transformer(self, data: pd.DataFrame) -> Dict[str, Any]:
        """為 Transformer 準備資料 (實作基類抽象方法)"""
        past_values, future_values = self._prepare_data(data, fit_scaler=True)
        return {
            "past_values": past_values,
            "future_values": future_values,
            "scaler": self.scaler,
        }

    def setup_model(self, **model_kwargs):
        """設置模型 (實作基類抽象方法)"""
        num_features = model_kwargs.get("num_features", 1)
        self.lightning_model = PatchTSTLightning(
            config=self.config,
            num_features=num_features,
            learning_rate=self.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler=self.training_config.lr_scheduler,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        training_config: Optional["TrainingConfig"] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        output_dir: str = "./results/lightning_training",
        **kwargs,
    ) -> "PatchTSTLightningWrapper":
        """
        訓練模型

        Args:
            X: 訓練特徵資料
            y: 訓練目標資料
            validation_data: 驗證資料 (X_val, y_val)
            training_config: 訓練配置（優先順序: explicit kwarg > training_config > self defaults）
            num_epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            early_stopping_patience: 早停耐心值
            output_dir: 輸出目錄

        Returns:
            訓練完成的模型實例
        """
        logger.info("開始訓練 PatchTST Lightning 模型...")

        # 參數優先順序: explicit kwarg > training_config > self defaults
        effective = training_config or self.training_config
        num_epochs = num_epochs or effective.num_epochs
        batch_size = batch_size or effective.batch_size
        learning_rate = learning_rate or effective.learning_rate
        early_stopping_patience = (
            early_stopping_patience or effective.early_stopping_patience
        )

        try:
            # 準備訓練資料
            train_past, train_future = self._prepare_data(X, y, fit_scaler=True)

            # 準備驗證資料
            if validation_data is not None:
                X_val, y_val = validation_data
                val_past, val_future = self._prepare_data(
                    X_val, y_val, fit_scaler=False
                )
            else:
                # 自動分割 20% 作為驗證
                val_size = int(0.2 * len(train_past))
                if val_size > 0:
                    val_past = train_past[-val_size:]
                    val_future = train_future[-val_size:]
                    train_past = train_past[:-val_size]
                    train_future = train_future[:-val_size]
                else:
                    val_past = val_future = None

            # 創建資料集
            train_dataset = TimeSeriesDataset(train_past, train_future)
            val_dataset = (
                TimeSeriesDataset(val_past, val_future)
                if val_past is not None
                else None
            )

            # 創建 DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.device.type == "cuda" else False,
            )
            val_loader = (
                DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
                )
                if val_dataset
                else None
            )

            logger.info(
                f"訓練資料: {len(train_dataset)} 筆, "
                f"驗證資料: {len(val_dataset) if val_dataset else 0} 筆"
            )

            # 創建 Lightning 模型
            self.lightning_model = PatchTSTLightning(
                config=self.config,
                num_features=train_past.shape[-1],
                learning_rate=learning_rate,
                weight_decay=self.training_config.weight_decay,
            )

            # 設置回調
            self.training_config.early_stopping_patience = early_stopping_patience
            callbacks = get_lightning_callbacks(
                self.training_config, checkpoint_dir=f"{output_dir}/checkpoints"
            )

            # 新增 metrics history callback
            metrics_callback = MetricsHistoryCallback()
            callbacks.append(metrics_callback)

            # 記錄訓練時的 num_features
            self._num_features = train_past.shape[-1]

            # 創建訓練器
            self.trainer = pl.Trainer(
                max_epochs=num_epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                callbacks=callbacks,
                enable_progress_bar=True,
                enable_model_summary=True,
                gradient_clip_val=self.training_config.gradient_clip_val,
                accumulate_grad_batches=self.training_config.accumulate_grad_batches,
                default_root_dir=output_dir,
                log_every_n_steps=10,
            )

            # 開始訓練
            logger.info("開始 Lightning 訓練循環...")
            self.trainer.fit(
                self.lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

            # 更新訓練歷史（從 callback 取得完整歷史）
            self.training_history["train_loss"] = metrics_callback.train_losses
            self.training_history["val_loss"] = metrics_callback.val_losses
            self.training_history["best_val_loss"] = (
                min(metrics_callback.val_losses)
                if metrics_callback.val_losses
                else float("inf")
            )

            # 訓練後移到 CPU 並設為 eval，確保 predict 一致性
            self.lightning_model.cpu()
            self.lightning_model.eval()

            self.is_fitted = True

            logger.info(
                f"PatchTST Lightning 訓練完成! "
                f"Best val_loss: {self.training_history['best_val_loss']:.6f}"
            )

            return self

        except Exception as e:
            logger.error(f"PatchTST Lightning 訓練失敗: {str(e)}")
            raise

    def _prepare_input_tensor(self, X) -> torch.Tensor:
        """
        將輸入資料轉換為模型所需的張量格式

        支援:
        - 3D numpy (n_samples, context_length, n_features): 直接轉換
        - 2D numpy/DataFrame: 取最後 context_length 筆，作為單一樣本
        """
        X_np = self._to_numpy(X).astype(np.float32)

        if X_np.ndim == 3:
            # 已是批次輸入 (n_samples, context_length, n_features)
            return torch.FloatTensor(X_np)

        # 2D: (n_timesteps, n_features) 或 (n_timesteps,)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        if len(X_np) < self.config.context_length:
            raise ValueError(
                f"輸入資料長度 ({len(X_np)}) 小於 "
                f"context_length ({self.config.context_length})"
            )

        recent = X_np[-self.config.context_length :]
        return torch.FloatTensor(recent).unsqueeze(0)  # (1, context_length, n_features)

    def _extract_target_values(self, X) -> np.ndarray:
        """從 X 提取目標值 (用於 predict 的 scaling)"""
        if (
            self.use_multi_channel
            and isinstance(X, pd.DataFrame)
            and self._feature_columns is not None
        ):
            return X[self._feature_columns].values  # (N, n_features)
        elif isinstance(X, pd.DataFrame) and "Close" in X.columns:
            return X["Close"].values.reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("資料中沒有數值欄位")
            return X[numeric_cols[0]].values.reshape(-1, 1)
        else:
            X_np = self._to_numpy(X)
            if X_np.ndim == 1:
                return X_np.reshape(-1, 1)
            return X_np[:, 0].reshape(-1, 1)

    def predict(self, X, horizon: int = None, **kwargs) -> np.ndarray:
        """
        進行預測

        Args:
            X: 輸入資料，支援:
               - numpy array (n_samples, context_length, n_features)
               - numpy array (n_timesteps, n_features)
               - pandas DataFrame
            horizon: 預測範圍

        Returns:
            預測結果數組 (n_samples, horizon) 或 (horizon,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")

        horizon = horizon or self.config.prediction_length

        try:
            self.lightning_model.eval()
            X_np = self._to_numpy(X)

            # 3D 輸入：假設已經 scaled
            if X_np.ndim == 3:
                past_values = torch.FloatTensor(X_np.astype(np.float32))
                past_values = past_values.to(self.lightning_model.device)

                with torch.no_grad():
                    predictions = self.lightning_model(past_values).cpu().numpy()

                if predictions.ndim == 3:
                    predictions = predictions.mean(axis=-1)
                return predictions[:, :horizon]

            # 2D 輸入：提取目標值 → scale → 推論 → inverse_transform
            values = self._extract_target_values(X).astype(np.float32)

            if len(values) < self.config.context_length:
                raise ValueError(
                    f"輸入資料長度 ({len(values)}) 小於 "
                    f"context_length ({self.config.context_length})"
                )

            recent_values = values[-self.config.context_length :]
            recent_scaled = self.scaler.transform(recent_values)

            past_values = torch.FloatTensor(recent_scaled).unsqueeze(0)
            past_values = past_values.to(self.lightning_model.device)

            with torch.no_grad():
                predictions = self.lightning_model(past_values).cpu().numpy()

            # 若模型輸出 3D (batch, pred_len, n_features)
            if predictions.ndim == 3:
                if self.use_multi_channel and predictions.shape[-1] > 1:
                    # 多 channel: 取 target channel
                    predictions = predictions[
                        :, :, self._target_channel_idx
                    ]  # (batch, pred_len)
                else:
                    predictions = predictions.mean(axis=-1)  # (batch, pred_len)

            # Inverse transform
            if self.use_multi_channel and self.scaler.n_features_in_ > 1:
                # 多 channel scaler: 構建完整 feature 陣列
                n_feats = self.scaler.n_features_in_
                pred_flat = predictions.flatten()
                dummy = np.zeros((len(pred_flat), n_feats))
                dummy[:, self._target_channel_idx] = pred_flat
                rescaled = self.scaler.inverse_transform(dummy)
                pred_rescaled = rescaled[:, self._target_channel_idx]
            else:
                pred_2d = predictions.reshape(-1, 1)
                pred_rescaled = self.scaler.inverse_transform(pred_2d).flatten()

            # 截取 horizon
            pred_rescaled = pred_rescaled[:horizon]

            logger.info(
                f"PatchTST Lightning 預測完成，輸出 shape: {pred_rescaled.shape}"
            )
            return pred_rescaled

        except Exception as e:
            logger.error(f"PatchTST Lightning 預測失敗: {str(e)}")
            raise

    def predict_with_uncertainty(
        self,
        X,
        horizon: int = None,
        confidence_level: float = 0.95,
        n_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        帶不確定性的預測 (使用 MC Dropout)

        Args:
            X: 輸入資料，支援 numpy array 或 pandas DataFrame
            horizon: 預測範圍
            confidence_level: 信賴區間水準
            n_samples: Monte Carlo 樣本數

        Returns:
            包含 predictions, std, lower_bound, upper_bound 的字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")

        horizon = horizon or self.config.prediction_length

        try:
            # 準備 scaled 輸入
            values = self._extract_target_values(X).astype(np.float32)

            if len(values) < self.config.context_length:
                raise ValueError(
                    f"輸入資料長度 ({len(values)}) 小於 "
                    f"context_length ({self.config.context_length})"
                )

            recent_values = values[-self.config.context_length :]
            recent_scaled = self.scaler.transform(recent_values)
            past_values = torch.FloatTensor(recent_scaled).unsqueeze(0)
            past_values = past_values.to(self.lightning_model.device)

            # 啟用 dropout 進行 MC sampling
            self.lightning_model.train()  # 啟用 dropout

            all_predictions = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.lightning_model(past_values)
                    all_predictions.append(pred.cpu().numpy())

            self.lightning_model.eval()

            # all_predictions: list of (1, pred_len[, n_features]) → (n_mc, 1, ...)
            all_predictions = np.array(all_predictions)

            # 若模型輸出 4D (n_mc, batch, pred_len, n_features)，取特徵平均
            if all_predictions.ndim == 4:
                all_predictions = all_predictions.mean(axis=-1)  # (n_mc, 1, pred_len)

            # Squeeze batch dim: (n_mc, pred_len)
            all_predictions = all_predictions.squeeze(1)

            mean_pred = np.mean(all_predictions, axis=0)  # (pred_len,)
            std_pred = np.std(all_predictions, axis=0)  # (pred_len,)

            # Inverse transform mean, lower, upper
            from scipy import stats

            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha / 2)

            lower_scaled = mean_pred - z_score * std_pred
            upper_scaled = mean_pred + z_score * std_pred

            mean_rescaled = self.scaler.inverse_transform(
                mean_pred.reshape(-1, 1)
            ).flatten()
            lower_rescaled = self.scaler.inverse_transform(
                lower_scaled.reshape(-1, 1)
            ).flatten()
            upper_rescaled = self.scaler.inverse_transform(
                upper_scaled.reshape(-1, 1)
            ).flatten()
            # std: 只 scale 不 shift
            std_rescaled = std_pred.flatten() * self.scaler.scale_[0]

            result = {
                "predictions": mean_rescaled[:horizon],
                "std": std_rescaled[:horizon],
                "lower_bound": lower_rescaled[:horizon],
                "upper_bound": upper_rescaled[:horizon],
                "confidence_level": confidence_level,
            }

            logger.info(
                f"PatchTST Lightning 不確定性預測完成，"
                f"信賴區間: {confidence_level * 100}%"
            )
            return result

        except Exception as e:
            logger.error(f"PatchTST Lightning 不確定性預測失敗: {str(e)}")
            raise

    # evaluate() 已統一為 BaseModel.evaluate_single_shot()（含 MASE/MDA）

    def save_model(self, filepath: str) -> bool:
        """儲存模型"""
        try:
            save_path = Path(filepath)
            save_path.mkdir(parents=True, exist_ok=True)

            # 儲存 PyTorch 模型狀態
            torch.save(self.lightning_model.state_dict(), save_path / "model.pt")

            # 儲存元資料
            metadata = {
                "config": self.config.to_dict(),
                "training_config": {
                    "num_epochs": self.training_config.num_epochs,
                    "batch_size": self.training_config.batch_size,
                    "learning_rate": self.training_config.learning_rate,
                },
                "training_history": self.training_history,
                "is_fitted": self.is_fitted,
                "model_name": self.model_name,
                "implementation": "lightning",
                "num_features": getattr(self, "_num_features", 1),
            }

            with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # 儲存標準化器
            joblib.dump(self.scaler, save_path / "scaler.joblib")

            logger.info(f"PatchTST Lightning 模型已儲存至: {filepath}")
            return True

        except Exception as e:
            logger.error(f"模型儲存失敗: {str(e)}")
            return False

    def load_model(self, filepath: str) -> bool:
        """載入模型"""
        try:
            load_path = Path(filepath)

            # 載入元資料
            with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # 恢復配置
            config_dict = metadata["config"]
            self.config = PatchTSTConfig(
                **{
                    k: v
                    for k, v in config_dict.items()
                    if k in PatchTSTConfig.__dataclass_fields__
                }
            )

            self.training_history = metadata["training_history"]
            self.is_fitted = metadata["is_fitted"]

            # 創建模型（使用 metadata 中的 num_features）
            num_features = metadata.get("num_features", 1)
            self._num_features = num_features
            self.lightning_model = PatchTSTLightning(
                config=self.config,
                num_features=num_features,
                learning_rate=self.learning_rate,
            )

            # 載入狀態
            state_dict = torch.load(
                load_path / "model.pt", map_location="cpu", weights_only=True
            )
            self.lightning_model.load_state_dict(state_dict)
            self.lightning_model.cpu()
            self.lightning_model.eval()

            # 載入標準化器
            scaler_path = load_path / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            logger.info(f"PatchTST Lightning 模型已從 {filepath} 載入")
            return True

        except Exception as e:
            logger.error(f"模型載入失敗: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型資訊"""
        info = {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "implementation": "lightning",
            "is_fitted": self.is_fitted,
            "device": str(self.device),
            "context_length": self.config.context_length,
            "prediction_length": self.config.prediction_length,
            "config": self.config.to_dict(),
            "training_history": self.training_history,
        }

        if self.lightning_model is not None:
            info["num_parameters"] = self.lightning_model.model.get_num_parameters()
            info["trainable_parameters"] = (
                self.lightning_model.model.get_num_trainable_parameters()
            )

        return info
