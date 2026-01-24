"""
基於 Transformer 的 PatchTST 模型實作

使用 HuggingFace Transformers 實作真正的 PatchTST 模型
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
from transformers import (
    PatchTSTConfig, 
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .base import TransformerBasedModel, ModelType

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """時間序列資料集類別，用於 PyTorch DataLoader"""
    
    def __init__(self, 
                 past_values: torch.Tensor, 
                 future_values: torch.Tensor = None):
        """
        初始化時間序列資料集
        
        Args:
            past_values: 過去的時間序列值
            future_values: 未來的目標值（可選）
        """
        self.past_values = past_values
        self.future_values = future_values
    
    def __len__(self):
        return len(self.past_values)
    
    def __getitem__(self, idx):
        item = {
            'past_values': self.past_values[idx]
        }
        
        if self.future_values is not None:
            item['future_values'] = self.future_values[idx]
            
        return item


class PatchTSTTransformer(TransformerBasedModel):
    """
    真正的 PatchTST Transformer 模型
    
    使用 HuggingFace 的 PatchTST 實作
    """
    
    def __init__(self, 
                 context_length: int = 64,
                 prediction_length: int = 1,
                 patch_length: int = 16,
                 stride: int = 8,
                 num_parallel_samples: int = 100,
                 **kwargs):
        """
        初始化 PatchTST 模型
        
        Args:
            context_length: 輸入序列長度
            prediction_length: 預測序列長度
            patch_length: patch 大小
            stride: patch 步長
            num_parallel_samples: 並行樣本數量
        """
        super().__init__()
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.stride = stride
        self.num_parallel_samples = num_parallel_samples
        
        # 初始化設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用設備: {self.device}")
        
        # 初始化縮放器
        self.scaler = StandardScaler()
        
        # 模型配置
        self.config = None
        self.model = None
        self.trainer = None
        
        # 訓練歷史
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'best_eval_loss': float('inf')
        }
        
        logger.info(f"PatchTST Transformer 模型初始化完成")
    
    def setup_model(self, num_features: int = 1, **model_kwargs):
        """
        設置 PatchTST 模型配置
        
        Args:
            num_features: 特徵數量
            **model_kwargs: 額外的模型配置參數
        """
        try:
            # 創建模型配置
            config_params = {
                'context_length': self.context_length,
                'prediction_length': self.prediction_length,
                'patch_length': self.patch_length,
                'stride': self.stride,
                'num_input_channels': num_features,
                'num_parallel_samples': self.num_parallel_samples,
                'd_model': model_kwargs.get('d_model', 128),
                'num_attention_heads': model_kwargs.get('num_attention_heads', 8),
                'num_hidden_layers': model_kwargs.get('num_hidden_layers', 6),
                'ffn_dim': model_kwargs.get('ffn_dim', 512),
                'dropout': model_kwargs.get('dropout', 0.1),
                'attention_dropout': model_kwargs.get('attention_dropout', 0.1),
                'pooling_type': model_kwargs.get('pooling_type', 'mean'),
                'channel_attention': model_kwargs.get('channel_attention', False),
                'scaling': model_kwargs.get('scaling', 'std'),
                'use_positional_encoding': model_kwargs.get('use_positional_encoding', True)
            }
            
            self.config = PatchTSTConfig(**config_params)
            
            # 創建模型
            self.model = PatchTSTForPrediction(self.config)
            self.model.to(self.device)
            
            logger.info(f"PatchTST 模型配置完成: {config_params}")
            logger.info(f"模型參數數量: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"模型配置失敗: {str(e)}")
            raise
    
    def prepare_data_for_transformer(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        為 Transformer 模型準備資料
        
        Args:
            data: 時間序列資料
            
        Returns:
            準備好的資料字典
        """
        try:
            # 確保資料按時間順序排列
            data = data.sort_index()
            
            # 選擇數值欄位
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            values = data[numeric_columns].values
            
            # 標準化資料
            values_scaled = self.scaler.fit_transform(values)
            
            # 創建序列
            sequences = []
            targets = []
            
            for i in range(len(values_scaled) - self.context_length - self.prediction_length + 1):
                # 輸入序列
                seq = values_scaled[i:i + self.context_length]
                sequences.append(seq)
                
                # 目標序列
                target = values_scaled[i + self.context_length:i + self.context_length + self.prediction_length]
                targets.append(target)
            
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            # 轉換為 PyTorch 張量
            past_values = torch.FloatTensor(sequences).to(self.device)
            future_values = torch.FloatTensor(targets).to(self.device)
            
            logger.info(f"資料準備完成: 序列形狀 {past_values.shape}, 目標形狀 {future_values.shape}")
            
            return {
                'past_values': past_values,
                'future_values': future_values,
                'scaler': self.scaler,
                'feature_names': list(numeric_columns)
            }
            
        except Exception as e:
            logger.error(f"資料準備失敗: {str(e)}")
            raise
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None, 
            validation_split: float = 0.2,
            num_epochs: int = 50,
            batch_size: int = 32,
            learning_rate: float = 1e-4,
            early_stopping_patience: int = 10,
            **kwargs):
        """
        訓練 PatchTST 模型
        
        Args:
            X: 輸入特徵資料
            y: 目標資料（在時間序列預測中，通常從 X 中提取）
            validation_split: 驗證集比例
            num_epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            early_stopping_patience: 早停耐心值
        """
        try:
            logger.info("開始訓練 PatchTST 模型...")
            
            # 如果模型尚未設置，使用預設配置
            if self.model is None:
                num_features = X.shape[1] if len(X.shape) > 1 else 1
                self.setup_model(num_features=num_features)
            
            # 準備資料
            data_dict = self.prepare_data_for_transformer(X)
            past_values = data_dict['past_values']
            future_values = data_dict['future_values']
            
            # 分割訓練和驗證資料
            train_size = int((1 - validation_split) * len(past_values))
            
            train_past = past_values[:train_size]
            train_future = future_values[:train_size]
            val_past = past_values[train_size:]
            val_future = future_values[train_size:]
            
            # 創建資料集
            train_dataset = TimeSeriesDataset(train_past, train_future)
            val_dataset = TimeSeriesDataset(val_past, val_future)
            
            # 設置訓練參數
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                learning_rate=learning_rate,
                remove_unused_columns=False,
            )
            
            # 創建訓練器
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            )
            
            # 開始訓練
            train_result = self.trainer.train()
            
            # 保存訓練歷史
            self.training_history['train_loss'] = [log['train_loss'] for log in self.trainer.state.log_history if 'train_loss' in log]
            self.training_history['eval_loss'] = [log['eval_loss'] for log in self.trainer.state.log_history if 'eval_loss' in log]
            
            if self.training_history['eval_loss']:
                self.training_history['best_eval_loss'] = min(self.training_history['eval_loss'])
            
            self.is_fitted = True
            logger.info(f"模型訓練完成! 最佳驗證損失: {self.training_history['best_eval_loss']:.6f}")
            
            return self
            
        except Exception as e:
            logger.error(f"模型訓練失敗: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        進行預測
        
        Args:
            X: 輸入特徵資料
            
        Returns:
            預測結果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")
        
        try:
            self.model.eval()
            
            # 準備預測資料
            data_dict = self.prepare_data_for_transformer(X)
            past_values = data_dict['past_values']
            
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(past_values), 32):  # 批次預測
                    batch_past = past_values[i:i+32]
                    
                    # 進行預測
                    outputs = self.model(past_values=batch_past)
                    batch_pred = outputs.prediction_outputs.mean(dim=1)  # 平均並行樣本
                    
                    predictions.append(batch_pred.cpu().numpy())
            
            # 合併預測結果
            predictions = np.concatenate(predictions, axis=0)
            
            # 反標準化
            predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])
            predictions_rescaled = self.scaler.inverse_transform(predictions_reshaped)
            
            # 取第一個特徵作為主要預測結果（通常是收盤價）
            final_predictions = predictions_rescaled[:, 0] if predictions_rescaled.shape[1] > 1 else predictions_rescaled.flatten()
            
            logger.info(f"預測完成，結果數量: {len(final_predictions)}")
            return final_predictions
            
        except Exception as e:
            logger.error(f"預測失敗: {str(e)}")
            raise
    
    def predict_with_uncertainty(self, X: pd.DataFrame, horizon: int = 1, 
                               confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        帶不確定性的預測
        
        Args:
            X: 輸入資料
            horizon: 預測範圍
            confidence_level: 信賴區間水準
            
        Returns:
            包含預測值和不確定性的字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")
        
        try:
            self.model.eval()
            
            # 準備預測資料
            data_dict = self.prepare_data_for_transformer(X)
            past_values = data_dict['past_values']
            
            all_samples = []
            
            with torch.no_grad():
                for i in range(0, len(past_values), 32):
                    batch_past = past_values[i:i+32]
                    
                    # 獲取所有並行樣本
                    outputs = self.model(past_values=batch_past)
                    batch_samples = outputs.prediction_outputs  # [batch_size, num_samples, prediction_length, num_features]
                    
                    all_samples.append(batch_samples.cpu().numpy())
            
            # 合併所有樣本
            all_samples = np.concatenate(all_samples, axis=0)
            
            # 計算統計量
            mean_pred = np.mean(all_samples, axis=1)
            std_pred = np.std(all_samples, axis=1)
            
            # 計算信賴區間
            from scipy import stats
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred
            
            # 反標準化
            mean_pred_flat = mean_pred.reshape(-1, mean_pred.shape[-1])
            lower_bound_flat = lower_bound.reshape(-1, lower_bound.shape[-1])
            upper_bound_flat = upper_bound.reshape(-1, upper_bound.shape[-1])
            
            mean_rescaled = self.scaler.inverse_transform(mean_pred_flat)
            lower_rescaled = self.scaler.inverse_transform(lower_bound_flat)
            upper_rescaled = self.scaler.inverse_transform(upper_bound_flat)
            
            # 取第一個特徵
            result = {
                'mean': mean_rescaled[:, 0] if mean_rescaled.shape[1] > 1 else mean_rescaled.flatten(),
                'lower_bound': lower_rescaled[:, 0] if lower_rescaled.shape[1] > 1 else lower_rescaled.flatten(),
                'upper_bound': upper_rescaled[:, 0] if upper_rescaled.shape[1] > 1 else upper_rescaled.flatten(),
                'std': std_pred.mean(axis=-1) if len(std_pred.shape) > 2 else std_pred.flatten()
            }
            
            logger.info(f"不確定性預測完成，信賴區間: {confidence_level*100}%")
            return result
            
        except Exception as e:
            logger.error(f"不確定性預測失敗: {str(e)}")
            raise
    
    def save_model(self, filepath: str) -> bool:
        """
        儲存模型
        
        Args:
            filepath: 儲存路徑
            
        Returns:
            是否儲存成功
        """
        try:
            save_path = Path(filepath)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 儲存模型和配置
            self.model.save_pretrained(save_path)
            
            # 儲存縮放器和其他元資料
            metadata = {
                'context_length': self.context_length,
                'prediction_length': self.prediction_length,
                'patch_length': self.patch_length,
                'stride': self.stride,
                'num_parallel_samples': self.num_parallel_samples,
                'training_history': self.training_history,
                'is_fitted': self.is_fitted,
                'model_type': self.model_type.value
            }
            
            with open(save_path / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 儲存縮放器
            import joblib
            if self.scaler is not None:
                joblib.dump(self.scaler, save_path / 'scaler.joblib')
            
            logger.info(f"模型已儲存至: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"模型儲存失敗: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        載入模型
        
        Args:
            filepath: 模型路徑
            
        Returns:
            是否載入成功
        """
        try:
            load_path = Path(filepath)
            
            # 載入元資料
            with open(load_path / 'metadata.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 恢復模型配置
            self.context_length = metadata['context_length']
            self.prediction_length = metadata['prediction_length']
            self.patch_length = metadata['patch_length']
            self.stride = metadata['stride']
            self.num_parallel_samples = metadata['num_parallel_samples']
            self.training_history = metadata['training_history']
            self.is_fitted = metadata['is_fitted']
            
            # 載入模型
            self.model = PatchTSTForPrediction.from_pretrained(load_path)
            self.model.to(self.device)
            
            # 載入縮放器
            import joblib
            scaler_path = load_path / 'scaler.joblib'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            logger.info(f"模型已從 {filepath} 載入")
            return True
            
        except Exception as e:
            logger.error(f"模型載入失敗: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型資訊"""
        base_info = super().get_model_info()
        
        transformer_info = {
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'patch_length': self.patch_length,
            'stride': self.stride,
            'num_parallel_samples': self.num_parallel_samples,
            'device': str(self.device),
            'training_history': self.training_history
        }
        
        if self.config:
            transformer_info['config'] = self.config.to_dict()
        
        base_info.update(transformer_info)
        return base_info