"""
PatchTST sklearn 實作

基於 sklearn GradientBoosting 的簡化版 PatchTST，
使用滑動窗口 (patching) 和統計特徵提取模擬 Transformer 的概念。

注意: 這不是真正的 Transformer 架構，但提供快速的訓練和推論。
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

from ...base import SklearnBasedModel
from ..config import PatchTSTConfig, TrainingConfig

logger = logging.getLogger(__name__)


class PatchTSTSklearn(SklearnBasedModel):
    """
    PatchTST 時間序列預測模型的 sklearn 實作

    使用滑動窗口 (模擬 patching) 和集成模型 (模擬 transformer 的表示學習)

    與 PatchTSTHuggingFace 和 PatchTSTLightning 共享相同的介面。
    """

    def __init__(
        self,
        # sklearn 風格參數 (優先)
        seq_len: Optional[int] = None,
        pred_len: Optional[int] = None,
        patch_len: Optional[int] = None,
        stride: Optional[int] = None,
        # 配置物件
        config: Optional[PatchTSTConfig] = None,
        # 直接參數
        context_length: int = 168,
        prediction_length: int = 24,
        patch_length: int = 12,
        patch_stride: int = 6,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
        **kwargs
    ):
        """
        初始化 PatchTST sklearn 模型

        Args:
            seq_len: 輸入序列長度 (sklearn 風格別名)
            pred_len: 預測序列長度 (sklearn 風格別名)
            patch_len: Patch 長度 (sklearn 風格別名)
            stride: Patch 步長 (sklearn 風格別名)
            config: PatchTSTConfig 配置物件
            context_length: 輸入序列長度
            prediction_length: 預測序列長度
            patch_length: Patch 長度
            patch_stride: Patch 步長
            n_estimators: 集成模型的估計器數量
            max_depth: 決策樹最大深度
            random_state: 隨機種子
        """
        super().__init__("PatchTST_Sklearn")

        # 處理配置
        if config is not None:
            self.config = config
        else:
            # 參數優先級: sklearn 風格 > 直接參數
            self.config = PatchTSTConfig.from_sklearn_params(
                seq_len=seq_len,
                pred_len=pred_len,
                patch_len=patch_len,
                stride=stride,
                context_length=context_length,
                prediction_length=prediction_length,
                patch_length=patch_length,
                patch_stride=patch_stride,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs
            )

        # 快捷屬性
        self.seq_len = self.config.context_length
        self.pred_len = self.config.prediction_length
        self.patch_len = self.config.patch_length
        self.stride = self.config.patch_stride
        self.n_estimators = self.config.n_estimators
        self.max_depth = self.config.max_depth
        self.random_state = self.config.random_state

        # 計算 patch 數量
        self.n_patches = self.config.num_patches

        # 模型組件
        self.scaler: StandardScaler = StandardScaler()
        self.target_scaler: StandardScaler = StandardScaler()

        # 集成預測模型（MultiOutputRegressor 支援真正多步預測）
        base_regressor = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.ensemble_model = MultiOutputRegressor(base_regressor)

        # 訓練歷史
        self.training_history: Dict[str, list[float]] = {
            'train_loss': [],
            'eval_loss': [],
        }

        # 儲存模型參數
        self.model_params = self.config.to_dict()

        logger.info(
            f"PatchTST sklearn 模型已初始化 - "
            f"seq_len: {self.seq_len}, pred_len: {self.pred_len}, "
            f"n_patches: {self.n_patches}"
        )

    def _create_patches(self, data: np.ndarray) -> np.ndarray:
        """
        創建 patches (滑動窗口)

        Args:
            data: 時間序列資料 [seq_len, n_features]

        Returns:
            patches: [n_patches, patch_len, n_features]
        """
        if len(data) < self.patch_len:
            raise ValueError(
                f"資料長度 {len(data)} 小於 patch 長度 {self.patch_len}"
            )

        patches: list[np.ndarray] = []
        for i in range(0, len(data) - self.patch_len + 1, self.stride):
            if len(patches) >= self.n_patches:
                break
            patch = data[i:i + self.patch_len]
            patches.append(patch)

        return np.array(patches)

    def _extract_patch_features(self, patches: np.ndarray) -> np.ndarray:
        """
        從 patches 中提取特徵 (模擬 transformer 的特徵提取)

        Args:
            patches: [n_patches, patch_len, n_features]

        Returns:
            features: [n_patches * n_features_per_patch]
        """
        features = []

        for patch in patches:
            # 統計特徵
            patch_features = [
                np.mean(patch, axis=0),      # 均值
                np.std(patch, axis=0),       # 標準差
                np.min(patch, axis=0),       # 最小值
                np.max(patch, axis=0),       # 最大值
                np.median(patch, axis=0),    # 中位數
            ]

            # 趨勢特徵 - 對每個特徵分別計算趨勢
            if len(patch) > 1 and patch.shape[1] > 0:
                trends = []
                for col in range(patch.shape[1]):
                    trend = np.polyfit(range(len(patch)), patch[:, col], 1)[0]
                    trends.append(trend)
                patch_features.append(trends)

            # 展平所有特徵
            flattened_features: list[float] = []
            for f in patch_features:
                if isinstance(f, (list, tuple)):
                    flattened_features.extend(f)
                else:
                    flattened_features.extend(f.flatten())
            features.extend(flattened_features)

        return np.array(features)

    def _extract_features_from_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        從原始資料提取 patch 特徵和目標值

        Args:
            X: 特徵資料（不含目標欄位）
            y: 目標資料

        Returns:
            (features, targets) numpy arrays
        """
        X_seq_list: list[np.ndarray] = []
        y_seq_list: list[np.ndarray] = []

        for i in range(len(X) - self.seq_len - self.pred_len + 1):
            input_seq = X.iloc[i:i + self.seq_len].values
            target_seq = y.iloc[
                i + self.seq_len:i + self.seq_len + self.pred_len
            ].values
            X_seq_list.append(input_seq)
            y_seq_list.append(target_seq)

        if not X_seq_list:
            raise ValueError("資料不足以創建訓練序列")

        X_sequences = np.array(X_seq_list)
        y_sequences = np.array(y_seq_list)

        features = np.array([
            self._extract_patch_features(self._create_patches(seq))
            for seq in X_sequences
        ])
        targets = y_sequences  # (N, pred_len) — 保留完整多步目標

        return features, targets

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        training_config: Optional[TrainingConfig] = None,
        **kwargs
    ) -> 'PatchTSTSklearn':
        """
        訓練模型

        Args:
            X: 訓練特徵資料
            y: 訓練目標資料
            validation_data: 驗證資料 (X_val, y_val)
            training_config: 訓練配置（可指定 validation_split 等）
            **kwargs: 額外參數

        Returns:
            訓練完成的模型
        """
        logger.info("開始訓練 PatchTST sklearn 模型")

        # 若有 training_config 且無 validation_data，自動切分
        if validation_data is None and training_config is not None:
            val_split = training_config.validation_split
            if val_split > 0:
                split_idx = int(len(X) * (1 - val_split))
                if split_idx > self.seq_len + self.pred_len:
                    validation_data = (X.iloc[split_idx:], y.iloc[split_idx:])
                    X = X.iloc[:split_idx]
                    y = y.iloc[:split_idx]

        # 記錄訓練時的欄位名，供 predict() 過濾用
        self._feature_columns: list[str] = list(X.columns)

        # 提取訓練特徵
        training_features, training_targets = self._extract_features_from_data(X, y)
        logger.info(f"準備了 {len(training_features)} 個訓練序列")

        # 標準化特徵
        training_features_scaled = self.scaler.fit_transform(training_features)

        # 標準化目標（2D: N × pred_len，逐 column 標準化）
        training_targets_scaled = self.target_scaler.fit_transform(training_targets)

        # 訓練集成模型
        self.ensemble_model.fit(training_features_scaled, training_targets_scaled)

        # 記錄訓練 loss
        train_pred = self.ensemble_model.predict(training_features_scaled)
        train_mse = float(np.mean((train_pred - training_targets_scaled) ** 2))
        self.training_history['train_loss'].append(train_mse)

        # 若有驗證資料，計算 eval loss
        if validation_data is not None:
            try:
                X_val, y_val = validation_data
                val_features, val_targets = self._extract_features_from_data(X_val, y_val)
                val_features_scaled = self.scaler.transform(val_features)
                val_targets_scaled = self.target_scaler.transform(val_targets)
                val_pred = self.ensemble_model.predict(val_features_scaled)
                eval_mse = float(np.mean((val_pred - val_targets_scaled) ** 2))
                self.training_history['eval_loss'].append(eval_mse)
            except (ValueError, Exception) as e:
                logger.warning(f"驗證資料評估失敗: {e}")

        self.is_fitted = True
        logger.info("PatchTST sklearn 模型訓練完成")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        horizon: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        進行預測

        Args:
            X: 輸入特徵資料
            horizon: 預測時間範圍 (使用模型內建的 pred_len)

        Returns:
            預測結果數組
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")

        # 過濾到訓練時使用的欄位（處理額外欄位或欄位順序不同的情況）
        if hasattr(self, '_feature_columns') and isinstance(X, pd.DataFrame):
            missing = set(self._feature_columns) - set(X.columns)
            if missing:
                raise ValueError(f"predict() 缺少訓練時使用的欄位: {missing}")
            X = X[self._feature_columns]

        if len(X) < self.seq_len:
            raise ValueError(
                f"輸入資料長度 {len(X)} 小於所需序列長度 {self.seq_len}"
            )

        # 取最後 seq_len 個數據點
        input_seq = X.iloc[-self.seq_len:].values

        # 創建 patches
        patches = self._create_patches(input_seq)

        # 提取特徵
        features = self._extract_patch_features(patches).reshape(1, -1)

        # 標準化特徵
        features_scaled = self.scaler.transform(features)

        # 預測（MultiOutputRegressor 輸出 shape: (1, pred_len)）
        prediction_scaled = self.ensemble_model.predict(features_scaled)

        # 反標準化
        prediction = self.target_scaler.inverse_transform(prediction_scaled)
        multi_step_prediction = prediction.flatten()

        return multi_step_prediction

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        horizon: Optional[int] = None,
        confidence_level: float = 0.95,
        **kwargs
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
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")

        # 基本預測
        predictions = self.predict(X, horizon)

        # 使用歷史誤差來估計不確定性
        # 在實際應用中，可以使用更複雜的方法

        # 估計預測標準差 (簡化假設)
        prediction_std = np.std(predictions) * 0.1  # 簡化的不確定性估計

        # 計算信賴區間
        z_score = 1.96  # 95% 信賴區間的 z 分數

        margin_of_error = z_score * prediction_std
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error

        return {
            'predictions': predictions,
            'std': np.full_like(predictions, prediction_std),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }

    def save_model(self, filepath: str) -> bool:
        """
        儲存模型

        Args:
            filepath: 儲存路徑

        Returns:
            是否儲存成功
        """
        try:
            model_data = {
                'ensemble_model': self.ensemble_model,
                'scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'config': self.config,
                'model_params': self.model_params,
                'is_fitted': self.is_fitted,
                'training_history': self.training_history,
                '_feature_columns': getattr(self, '_feature_columns', None),
            }

            joblib.dump(model_data, filepath)
            logger.info(f"模型已儲存至: {filepath}")
            return True

        except Exception as e:
            logger.error(f"模型儲存失敗: {str(e)}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        載入模型

        Args:
            filepath: 模型檔案路徑

        Returns:
            是否載入成功
        """
        try:
            if not Path(filepath).exists():
                logger.error(f"模型檔案不存在: {filepath}")
                return False

            model_data = joblib.load(filepath)

            self.ensemble_model = model_data['ensemble_model']
            self.scaler = model_data['scaler']
            self.target_scaler = model_data['target_scaler']
            self.config = model_data.get('config', PatchTSTConfig())
            self.model_params = model_data['model_params']
            self.is_fitted = model_data['is_fitted']
            self.training_history = model_data.get(
                'training_history', {'train_loss': [], 'eval_loss': []}
            )
            self._feature_columns = model_data.get('_feature_columns', None)

            # 恢復模型參數
            self.seq_len = self.config.context_length
            self.pred_len = self.config.prediction_length
            self.patch_len = self.config.patch_length
            self.stride = self.config.patch_stride
            self.n_patches = self.config.num_patches

            logger.info(f"模型已從 {filepath} 載入")
            return True

        except Exception as e:
            logger.error(f"模型載入失敗: {str(e)}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """
        取得特徵重要性

        Returns:
            特徵重要性字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")

        if hasattr(self.ensemble_model, 'feature_importances_'):
            importances = self.ensemble_model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importances))]

            return dict(zip(feature_names, importances))
        else:
            return {}

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series
    ) -> Dict[str, float]:
        """
        評估模型性能

        Args:
            X: 測試特徵資料
            y_true: 真實目標值

        Returns:
            評估指標字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")

        predictions = self.predict(X)

        # 計算評估指標
        y_actual = y_true[-len(predictions):]
        mse = mean_squared_error(y_actual, predictions)
        mae = mean_absolute_error(y_actual, predictions)
        rmse = np.sqrt(mse)

        # 計算 MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_actual - predictions) / y_actual)) * 100

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    def get_model_info(self) -> Dict[str, Any]:
        """取得模型資訊"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'implementation': 'sklearn',
            'is_fitted': self.is_fitted,
            'config': self.config.to_dict(),
            'n_patches': self.n_patches,
        }


# 向後兼容的別名
PatchTST = PatchTSTSklearn
