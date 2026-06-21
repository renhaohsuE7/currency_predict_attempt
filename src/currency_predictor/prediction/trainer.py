"""
模型訓練器

擁有模型實例,負責「建立 → 訓練(含驗證分割)→ 評估 → 存/載」。
不認識 symbol / 資料來源(那是 DataManager 的事);上層 Facade 提供切好的
X/y,並在結果加上 symbol。
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..models.factory import ModelFactory

logger = logging.getLogger(__name__)

# 一個 validation split 至少要留給「訓練子集」的最少筆數,否則 PatchTST 切不出序列
# (保守估:seq_len=168 + pred_len=24 + 緩衝)。低於此就跳過驗證分割、全資料訓練。
MIN_RECORDS_FOR_VALIDATION_SPLIT = 200


class ModelTrainer:
    """持有並訓練/評估/持久化單一模型。"""

    def __init__(self, model_name: str = "patchtst_sklearn", model_params: Dict[str, Any] = None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = self._create_model()

    def _create_model(self):
        """依名稱建立模型;失敗則回退 sklearn 版。"""
        try:
            return ModelFactory.create_model(self.model_name, **self.model_params)
        except Exception as e:
            logger.error(f"創建模型失敗: {str(e)}")
            logger.info("回退到 sklearn 版本的 PatchTST")
            return ModelFactory.create_model("patchtst_sklearn", **self.model_params)

    @property
    def is_fitted(self) -> bool:
        return getattr(self.model, "is_fitted", False)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **train_kwargs,
    ) -> Dict[str, Any]:
        """訓練模型並回傳結果 dict(不含 symbol,由 Facade 補上)。"""
        try:
            logger.info(f"開始訓練 {self.model_name} 模型")

            # validation_split (float) → validation_data (tuple);PatchTST 吃後者。
            if 'validation_split' in train_kwargs:
                val_split = train_kwargs.pop('validation_split')

                if val_split > 0 and len(X_train) * (1 - val_split) >= MIN_RECORDS_FOR_VALIDATION_SPLIT:
                    split_idx = int(len(X_train) * (1 - val_split))
                    X_val = X_train.iloc[split_idx:]
                    y_val = y_train.iloc[split_idx:]
                    X_train_subset = X_train.iloc[:split_idx]
                    y_train_subset = y_train.iloc[:split_idx]

                    train_kwargs['validation_data'] = (X_val, y_val)
                    logger.info(f"使用驗證分割: {len(X_train_subset)} 訓練, {len(X_val)} 驗證")
                    self.model.fit(X_train_subset, y_train_subset, **train_kwargs)
                else:
                    logger.warning(f"訓練資料不足({len(X_train)})，跳過驗證分割")
                    self.model.fit(X_train, y_train, **train_kwargs)
            else:
                self.model.fit(X_train, y_train, **train_kwargs)

            train_metrics = self.evaluate(X_train, y_train, "訓練")
            test_metrics = self.evaluate(X_test, y_test, "測試")

            logger.info("模型訓練完成")
            return {
                'model_name': self.model_name,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'training_completed': True,
            }

        except Exception as e:
            logger.error(f"模型訓練失敗: {str(e)}")
            return {
                'model_name': self.model_name,
                'error': str(e),
                'training_completed': False,
            }

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series, dataset_name: str) -> Dict[str, float]:
        """評估模型性能(優先用模型自帶 evaluate,否則退回 MSE/MAE/RMSE)。"""
        try:
            if hasattr(self.model, 'evaluate'):
                metrics = self.model.evaluate(X, y_true)
            else:
                predictions = self.model.predict(X)
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                mse = mean_squared_error(y_true[-len(predictions):], predictions)
                mae = mean_absolute_error(y_true[-len(predictions):], predictions)
                metrics = {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse)}

            logger.info(f"{dataset_name}集評估結果: MSE={metrics.get('mse', 0):.6f}")
            return metrics

        except Exception as e:
            logger.error(f"模型評估失敗: {str(e)}")
            return {}

    def save(self, filepath: str) -> bool:
        """儲存模型。"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            success = self.model.save_model(filepath)
            if success:
                logger.info(f"模型已儲存至: {filepath}")
            return success
        except Exception as e:
            logger.error(f"儲存模型失敗: {str(e)}")
            return False

    def load(self, filepath: str) -> bool:
        """載入模型。"""
        try:
            success = self.model.load_model(filepath)
            if success:
                logger.info(f"模型已從 {filepath} 載入")
            return success
        except Exception as e:
            logger.error(f"載入模型失敗: {str(e)}")
            return False
