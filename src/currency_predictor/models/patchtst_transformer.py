"""
基於 HuggingFace Transformers 的 PatchTST 模型實作

使用真正的 PatchTST Transformer 架構進行時間序列預測
保持與 sklearn 版本相同的介面，方便整合到現有 pipeline
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import joblib

from .base import TransformerBasedModel, ModelType

logger = logging.getLogger(__name__)

# 嘗試導入 HuggingFace transformers
try:
    from transformers import (
        PatchTSTConfig,
        PatchTSTForPrediction,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers 庫未安裝，PatchTSTTransformer 將無法使用")


class TimeSeriesDataset(Dataset):
    """
    時間序列資料集類別

    用於 PyTorch DataLoader，支援 HuggingFace Trainer
    """

    def __init__(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None
    ):
        """
        初始化時間序列資料集

        Args:
            past_values: 過去的時間序列值 (num_samples, context_length, num_channels)
            future_values: 未來的目標值 (num_samples, prediction_length, num_channels)
        """
        self.past_values = past_values
        self.future_values = future_values

    def __len__(self) -> int:
        return len(self.past_values)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {'past_values': self.past_values[idx]}

        if self.future_values is not None:
            item['future_values'] = self.future_values[idx]

        return item


class PatchTSTTransformer(TransformerBasedModel):
    """
    HuggingFace PatchTST Transformer 模型

    提供與 sklearn 版本 PatchTST 相同的介面，內部使用真正的 Transformer 架構。

    主要特點:
    - 支援 seq_len/pred_len 參數別名 (兼容 sklearn 版本)
    - 自動處理資料格式轉換
    - 支援 GPU 加速
    - 不確定性估計

    使用範例:
        ```python
        model = PatchTSTTransformer(
            seq_len=64,        # 或使用 context_length
            pred_len=7,        # 或使用 prediction_length
            d_model=64,
            num_hidden_layers=2
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test, horizon=7)
        ```
    """

    def __init__(
        self,
        # 兼容 sklearn 版本的參數名稱
        seq_len: Optional[int] = None,
        pred_len: Optional[int] = None,
        patch_len: Optional[int] = None,
        stride: Optional[int] = None,
        # HuggingFace 標準參數名稱
        context_length: int = 64,
        prediction_length: int = 7,
        patch_length: int = 8,
        patch_stride: int = 4,
        # 模型架構參數
        num_input_channels: int = 1,
        d_model: int = 64,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        # 其他參數
        num_parallel_samples: int = 100,
        scaling: str = 'std',
        loss: str = 'mse',
        random_state: int = 42,
        **kwargs
    ):
        """
        初始化 PatchTST Transformer 模型

        Args:
            seq_len: 輸入序列長度 (兼容 sklearn 版本，優先於 context_length)
            pred_len: 預測長度 (兼容 sklearn 版本，優先於 prediction_length)
            patch_len: Patch 大小 (兼容 sklearn 版本)
            stride: Patch 步長 (兼容 sklearn 版本)
            context_length: HuggingFace 標準輸入序列長度
            prediction_length: HuggingFace 標準預測長度
            patch_length: HuggingFace 標準 Patch 大小
            patch_stride: HuggingFace 標準 Patch 步長
            num_input_channels: 輸入通道數 (特徵數量)
            d_model: Transformer 隱藏層維度
            num_attention_heads: 注意力頭數量
            num_hidden_layers: Transformer 層數
            ffn_dim: 前饋網路維度
            dropout: Dropout 率
            attention_dropout: 注意力 Dropout 率
            num_parallel_samples: 並行採樣數量 (用於不確定性估計)
            scaling: 縮放方式 ('std', 'mean', None)
            loss: 損失函數 ('mse', 'nll')
            random_state: 隨機種子
        """
        super().__init__(model_name="PatchTST_Transformer")

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers 庫未安裝，請執行: pip install transformers torch"
            )

        # 參數映射 (sklearn 風格 → HuggingFace 風格)
        self.context_length = seq_len if seq_len is not None else context_length
        self.prediction_length = pred_len if pred_len is not None else prediction_length
        self.patch_length = patch_len if patch_len is not None else patch_length
        self.patch_stride = stride if stride is not None else patch_stride

        # 模型架構參數
        self.num_input_channels = num_input_channels
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_parallel_samples = num_parallel_samples
        self.scaling = scaling
        self.loss = loss
        self.random_state = random_state

        # 設置隨機種子
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # 初始化設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PatchTST Transformer 使用設備: {self.device}")

        # 初始化標準化器
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # 模型和配置
        self.config = None
        self.model = None
        self.trainer = None
        self._num_features = None

        # 訓練歷史
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'best_eval_loss': float('inf')
        }

        # 儲存模型參數
        self.model_params = {
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'patch_length': self.patch_length,
            'patch_stride': self.patch_stride,
            'd_model': self.d_model,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers
        }

        logger.info(
            f"PatchTST Transformer 初始化完成: "
            f"context_length={self.context_length}, "
            f"prediction_length={self.prediction_length}, "
            f"d_model={self.d_model}"
        )

    def _setup_model(self, num_features: int = 1):
        """
        設置 PatchTST 模型

        Args:
            num_features: 特徵數量
        """
        self._num_features = num_features

        # 驗證 patch 參數
        if self.patch_length > self.context_length:
            logger.warning(
                f"patch_length ({self.patch_length}) > context_length ({self.context_length})，"
                f"自動調整 patch_length"
            )
            self.patch_length = max(1, self.context_length // 4)
            self.patch_stride = max(1, self.patch_length // 2)

        # 創建模型配置
        config_params = {
            'num_input_channels': num_features,
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'patch_length': self.patch_length,
            'patch_stride': self.patch_stride,
            'd_model': self.d_model,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'ffn_dim': self.ffn_dim,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'num_parallel_samples': self.num_parallel_samples,
            'scaling': self.scaling,
            'loss': self.loss,
            'norm_type': 'batchnorm',
            'channel_attention': False,
            'pooling_type': None,
        }

        self.config = PatchTSTConfig(**config_params)
        self.model = PatchTSTForPrediction(self.config)
        self.model.to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"PatchTST 模型創建完成，參數數量: {num_params:,}")

    def _create_sequences(
        self,
        values: np.ndarray,
        context_length: int,
        prediction_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        從時間序列資料創建訓練序列

        Args:
            values: 時間序列值 (num_samples, num_features)
            context_length: 輸入序列長度
            prediction_length: 預測序列長度

        Returns:
            past_values: (num_sequences, context_length, num_features)
            future_values: (num_sequences, prediction_length, num_features)
        """
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
            past_values.append(values[i:i + context_length])
            future_values.append(
                values[i + context_length:i + context_length + prediction_length]
            )

        return np.array(past_values), np.array(future_values)

    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit_scaler: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        準備訓練/預測資料

        Args:
            X: 特徵資料
            y: 目標資料 (可選，用於時間序列預測時可能嵌入在 X 中)
            fit_scaler: 是否擬合標準化器

        Returns:
            past_values: 過去序列 (num_samples, context_length, num_channels)
            future_values: 未來序列 (num_samples, prediction_length, num_channels)
        """
        # 確定要使用的目標欄位
        if y is not None:
            # 使用提供的 y 作為主要目標
            target_values = y.values.reshape(-1, 1)
            num_features = 1
        elif 'Close' in X.columns:
            # 使用 Close 欄位
            target_values = X['Close'].values.reshape(-1, 1)
            num_features = 1
        else:
            # 使用第一個數值欄位
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("資料中沒有數值欄位")
            target_values = X[numeric_cols[0]].values.reshape(-1, 1)
            num_features = 1

        # 標準化
        if fit_scaler:
            values_scaled = self.scaler.fit_transform(target_values)
        else:
            values_scaled = self.scaler.transform(target_values)

        # 創建序列
        past_values, future_values = self._create_sequences(
            values_scaled,
            self.context_length,
            self.prediction_length
        )

        # 轉換為 PyTorch 張量
        past_tensor = torch.FloatTensor(past_values)
        future_tensor = torch.FloatTensor(future_values)

        logger.debug(
            f"資料準備完成: past_values={past_tensor.shape}, "
            f"future_values={future_tensor.shape}"
        )

        return past_tensor, future_tensor

    def prepare_data_for_transformer(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        為 Transformer 模型準備資料 (實作基類抽象方法)

        Args:
            data: 時間序列資料

        Returns:
            準備好的資料字典
        """
        past_values, future_values = self._prepare_data(data, fit_scaler=True)

        return {
            'past_values': past_values,
            'future_values': future_values,
            'scaler': self.scaler
        }

    def setup_model(self, **model_kwargs):
        """設置 Transformer 模型 (實作基類抽象方法)"""
        num_features = model_kwargs.get('num_features', 1)
        self._setup_model(num_features)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 10,
        output_dir: str = './results/transformer_training',
        **kwargs
    ) -> 'PatchTSTTransformer':
        """
        訓練 PatchTST Transformer 模型

        Args:
            X: 訓練特徵資料
            y: 訓練目標資料 (可選)
            validation_data: 驗證資料 (X_val, y_val)
            num_epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            early_stopping_patience: 早停耐心值
            output_dir: 輸出目錄

        Returns:
            訓練完成的模型實例
        """
        logger.info("開始訓練 PatchTST Transformer 模型...")

        try:
            # 準備訓練資料
            train_past, train_future = self._prepare_data(X, y, fit_scaler=True)

            # 設置模型 (如果尚未設置)
            if self.model is None:
                self._setup_model(num_features=train_past.shape[-1])

            # 準備驗證資料
            val_dataset = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_past, val_future = self._prepare_data(X_val, y_val, fit_scaler=False)
                val_dataset = TimeSeriesDataset(val_past, val_future)
            else:
                # 自動分割 20% 作為驗證
                val_size = int(0.2 * len(train_past))
                if val_size > 0:
                    val_past = train_past[-val_size:]
                    val_future = train_future[-val_size:]
                    train_past = train_past[:-val_size]
                    train_future = train_future[:-val_size]
                    val_dataset = TimeSeriesDataset(val_past, val_future)

            # 創建訓練資料集
            train_dataset = TimeSeriesDataset(train_past, train_future)

            logger.info(
                f"訓練資料: {len(train_dataset)} 筆, "
                f"驗證資料: {len(val_dataset) if val_dataset else 0} 筆"
            )

            # 設置訓練參數
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                warmup_ratio=0.1,
                logging_dir=f'{output_dir}/logs',
                logging_steps=50,
                eval_strategy="epoch" if val_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_loss" if val_dataset else None,
                greater_is_better=False,
                save_total_limit=2,
                remove_unused_columns=False,
                report_to="none",  # 關閉 wandb 等報告
                seed=self.random_state,
                label_names=["future_values"],  # 告訴 Trainer future_values 是標籤
            )

            # 設置回調
            callbacks = []
            if val_dataset and early_stopping_patience > 0:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=early_stopping_patience,
                        early_stopping_threshold=0.0001
                    )
                )

            # 創建訓練器
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=callbacks if callbacks else None,
            )

            # 開始訓練
            logger.info("開始 Transformer 訓練循環...")
            train_result = self.trainer.train()

            # 記錄訓練歷史
            for log in self.trainer.state.log_history:
                if 'loss' in log:
                    self.training_history['train_loss'].append(log['loss'])
                if 'eval_loss' in log:
                    self.training_history['eval_loss'].append(log['eval_loss'])

            if self.training_history['eval_loss']:
                self.training_history['best_eval_loss'] = min(
                    self.training_history['eval_loss']
                )

            self.is_fitted = True

            logger.info(
                f"PatchTST Transformer 訓練完成! "
                f"最終訓練損失: {train_result.training_loss:.6f}"
            )

            return self

        except Exception as e:
            logger.error(f"PatchTST Transformer 訓練失敗: {str(e)}")
            raise

    def predict(
        self,
        X: pd.DataFrame,
        horizon: int = None,
        **kwargs
    ) -> np.ndarray:
        """
        進行預測

        Args:
            X: 輸入特徵資料 (至少需要 context_length 筆)
            horizon: 預測範圍 (預設使用 prediction_length)

        Returns:
            預測結果數組 (horizon,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")

        horizon = horizon or self.prediction_length

        try:
            self.model.eval()

            # 準備輸入資料
            if 'Close' in X.columns:
                values = X['Close'].values.reshape(-1, 1)
            else:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                values = X[numeric_cols[0]].values.reshape(-1, 1)

            # 確保有足夠的資料
            if len(values) < self.context_length:
                raise ValueError(
                    f"輸入資料長度 ({len(values)}) 小於 context_length ({self.context_length})"
                )

            # 取最後 context_length 筆資料
            recent_values = values[-self.context_length:]

            # 標準化
            recent_scaled = self.scaler.transform(recent_values)

            # 轉換為張量
            past_values = torch.FloatTensor(recent_scaled).unsqueeze(0)  # (1, context_length, 1)
            past_values = past_values.to(self.device)

            # 進行預測
            with torch.no_grad():
                outputs = self.model(past_values=past_values)

                # 獲取預測結果
                # prediction_outputs shape: (batch, num_samples, prediction_length, channels)
                predictions = outputs.prediction_outputs

                # 取平均 (對並行樣本取平均)
                mean_pred = predictions.mean(dim=1)  # (batch, prediction_length, channels)

                # 只 squeeze 掉 batch 維度，保留 prediction_length 和 channels
                mean_pred = mean_pred.squeeze(0).cpu().numpy()  # (prediction_length, channels)

                # 確保是 2D 陣列用於 inverse_transform
                if mean_pred.ndim == 0:  # scalar
                    mean_pred = np.array([[mean_pred.item()]])
                elif mean_pred.ndim == 1:
                    mean_pred = mean_pred.reshape(-1, 1)

                # 反標準化
                predictions_rescaled = self.scaler.inverse_transform(mean_pred)
                predictions_final = predictions_rescaled.flatten()

                # 截取需要的 horizon
                predictions_final = predictions_final[:horizon]

            logger.info(f"PatchTST Transformer 預測完成，輸出 {len(predictions_final)} 個時間點")
            return predictions_final

        except Exception as e:
            logger.error(f"PatchTST Transformer 預測失敗: {str(e)}")
            raise

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        horizon: int = None,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        帶不確定性的預測

        Args:
            X: 輸入資料
            horizon: 預測範圍
            confidence_level: 信賴區間水準

        Returns:
            包含 predictions, std, lower_bound, upper_bound 的字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")

        horizon = horizon or self.prediction_length

        try:
            self.model.eval()

            # 準備輸入資料
            if 'Close' in X.columns:
                values = X['Close'].values.reshape(-1, 1)
            else:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                values = X[numeric_cols[0]].values.reshape(-1, 1)

            recent_values = values[-self.context_length:]
            recent_scaled = self.scaler.transform(recent_values)

            past_values = torch.FloatTensor(recent_scaled).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(past_values=past_values)
                predictions = outputs.prediction_outputs  # (1, num_samples, pred_len, channels)

                # 轉換為 numpy
                all_samples = predictions.squeeze(0).cpu().numpy()  # (num_samples, pred_len, channels)

                # 處理形狀
                if all_samples.ndim == 1:
                    all_samples = all_samples.reshape(-1, 1, 1)
                elif all_samples.ndim == 2:
                    all_samples = all_samples.reshape(all_samples.shape[0], -1, 1)

                # 計算統計量
                mean_pred = np.mean(all_samples, axis=0)  # (pred_len, channels)
                std_pred = np.std(all_samples, axis=0)    # (pred_len, channels)

                # 計算信賴區間
                from scipy import stats
                alpha = 1 - confidence_level
                z_score = stats.norm.ppf(1 - alpha / 2)

                lower_bound = mean_pred - z_score * std_pred
                upper_bound = mean_pred + z_score * std_pred

                # 確保是 2D 陣列用於 inverse_transform
                if mean_pred.ndim == 1:
                    mean_pred = mean_pred.reshape(-1, 1)
                    lower_bound = lower_bound.reshape(-1, 1)
                    upper_bound = upper_bound.reshape(-1, 1)
                    std_pred = std_pred.reshape(-1, 1)

                # 反標準化
                mean_rescaled = self.scaler.inverse_transform(mean_pred).flatten()
                lower_rescaled = self.scaler.inverse_transform(lower_bound).flatten()
                upper_rescaled = self.scaler.inverse_transform(upper_bound).flatten()
                std_rescaled = std_pred.flatten() * self.scaler.scale_[0]  # 縮放標準差

                # 截取需要的 horizon
                result = {
                    'predictions': mean_rescaled[:horizon],
                    'std': std_rescaled[:horizon],
                    'lower_bound': lower_rescaled[:horizon],
                    'upper_bound': upper_rescaled[:horizon],
                    'confidence_level': confidence_level
                }

            logger.info(f"PatchTST Transformer 不確定性預測完成，信賴區間: {confidence_level * 100}%")
            return result

        except Exception as e:
            logger.error(f"PatchTST Transformer 不確定性預測失敗: {str(e)}")
            raise

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series
    ) -> Dict[str, float]:
        """
        評估模型

        Args:
            X: 特徵資料
            y_true: 真實值

        Returns:
            評估指標字典
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        predictions = self.predict(X, horizon=len(y_true))

        # 對齊長度
        min_len = min(len(predictions), len(y_true))
        y_pred = predictions[:min_len]
        y_actual = y_true.values[-min_len:] if hasattr(y_true, 'values') else y_true[-min_len:]

        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mse)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }

    def save_model(self, filepath: str) -> bool:
        """
        儲存模型

        Args:
            filepath: 儲存路徑 (目錄)

        Returns:
            是否儲存成功
        """
        try:
            save_path = Path(filepath)
            save_path.mkdir(parents=True, exist_ok=True)

            # 儲存 HuggingFace 模型
            self.model.save_pretrained(save_path)

            # 儲存元資料
            metadata = {
                'context_length': self.context_length,
                'prediction_length': self.prediction_length,
                'patch_length': self.patch_length,
                'patch_stride': self.patch_stride,
                'd_model': self.d_model,
                'num_attention_heads': self.num_attention_heads,
                'num_hidden_layers': self.num_hidden_layers,
                'num_parallel_samples': self.num_parallel_samples,
                'training_history': self.training_history,
                'is_fitted': self.is_fitted,
                'model_type': self.model_type.value,
                'model_name': self.model_name
            }

            with open(save_path / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # 儲存標準化器
            joblib.dump(self.scaler, save_path / 'scaler.joblib')

            logger.info(f"PatchTST Transformer 模型已儲存至: {filepath}")
            return True

        except Exception as e:
            logger.error(f"模型儲存失敗: {str(e)}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        載入模型

        Args:
            filepath: 模型目錄路徑

        Returns:
            是否載入成功
        """
        try:
            load_path = Path(filepath)

            # 載入元資料
            with open(load_path / 'metadata.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 恢復參數
            self.context_length = metadata['context_length']
            self.prediction_length = metadata['prediction_length']
            self.patch_length = metadata['patch_length']
            self.patch_stride = metadata['patch_stride']
            self.d_model = metadata.get('d_model', 64)
            self.num_attention_heads = metadata.get('num_attention_heads', 4)
            self.num_hidden_layers = metadata.get('num_hidden_layers', 2)
            self.num_parallel_samples = metadata['num_parallel_samples']
            self.training_history = metadata['training_history']
            self.is_fitted = metadata['is_fitted']

            # 載入 HuggingFace 模型
            self.model = PatchTSTForPrediction.from_pretrained(load_path)
            self.model.to(self.device)

            # 載入標準化器
            scaler_path = load_path / 'scaler.joblib'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            logger.info(f"PatchTST Transformer 模型已從 {filepath} 載入")
            return True

        except Exception as e:
            logger.error(f"模型載入失敗: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型資訊"""
        info = {
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'is_fitted': self.is_fitted,
            'device': str(self.device),
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'patch_length': self.patch_length,
            'patch_stride': self.patch_stride,
            'd_model': self.d_model,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'num_parallel_samples': self.num_parallel_samples,
            'training_history': self.training_history
        }

        if self.model is not None:
            info['num_parameters'] = sum(p.numel() for p in self.model.parameters())

        return info
