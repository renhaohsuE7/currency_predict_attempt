"""
PatchTST 時間序列預測模型

簡化版本的 PatchTST 實作，使用傳統機器學習方法模擬 patching 和 transformer 的概念
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path
from .base import TimeSeriesModel

logger = logging.getLogger(__name__)


class PatchTST(TimeSeriesModel):
    """
    PatchTST 時間序列預測模型的簡化實作
    
    使用滑動窗口（模擬 patching）和集成模型（模擬 transformer 的表示學習）
    """
    
    def __init__(
        self,
        seq_len: int = 168,          # 一週的小時數據
        pred_len: int = 24,          # 預測一天
        patch_len: int = 12,         # 半天的數據作為一個 patch
        stride: int = 6,             # patch 之間的步長
        n_estimators: int = 100,     # 隨機森林估計器數量
        max_depth: int = 10,         # 最大深度
        random_state: int = 42
    ):
        """
        初始化 PatchTST 模型
        
        Args:
            seq_len: 輸入序列長度
            pred_len: 預測序列長度
            patch_len: Patch 長度
            stride: Patch 步長
            n_estimators: 集成模型的估計器數量
            max_depth: 決策樹最大深度
            random_state: 隨機種子
        """
        super().__init__("PatchTST")
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # 計算 patch 數量
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        # 模型組件
        self.patch_models = []  # 每個 patch 對應一個子模型
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # 集成預測模型
        self.ensemble_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        # 儲存模型參數
        self.model_params = {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'patch_len': patch_len,
            'stride': stride,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
        
        logger.info(f"PatchTST 模型已初始化 - seq_len: {seq_len}, pred_len: {pred_len}, n_patches: {self.n_patches}")
    
    def _create_patches(self, data: np.ndarray) -> np.ndarray:
        """
        創建 patches（滑動窗口）
        
        Args:
            data: 時間序列資料 [seq_len, n_features]
            
        Returns:
            patches: [n_patches, patch_len, n_features]
        """
        if len(data) < self.patch_len:
            raise ValueError(f"資料長度 {len(data)} 小於 patch 長度 {self.patch_len}")
        
        patches = []
        for i in range(0, len(data) - self.patch_len + 1, self.stride):
            if len(patches) >= self.n_patches:
                break
            patch = data[i:i + self.patch_len]
            patches.append(patch)
        
        return np.array(patches)
    
    def _extract_patch_features(self, patches: np.ndarray) -> np.ndarray:
        """
        從 patches 中提取特徵（模擬 transformer 的特徵提取）
        
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
            flattened_features = []
            for f in patch_features:
                if isinstance(f, (list, tuple)):
                    flattened_features.extend(f)
                else:
                    flattened_features.extend(f.flatten())
            features.extend(flattened_features)
        
        return np.array(features)
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> 'PatchTST':
        """
        訓練模型
        
        Args:
            X: 訓練特徵資料
            y: 訓練目標資料
            validation_data: 驗證資料（暫未使用）
            
        Returns:
            訓練完成的模型
        """
        logger.info(f"開始訓練 PatchTST 模型")
        
        # 合併 X 和 y 用於序列準備
        full_data = pd.concat([X, y.to_frame()], axis=1)
        
        # 準備訓練序列
        X_sequences = []
        y_sequences = []
        
        for i in range(len(full_data) - self.seq_len - self.pred_len + 1):
            # 輸入序列
            input_seq = full_data.iloc[i:i + self.seq_len].values
            
            # 目標序列
            target_seq = full_data.iloc[i + self.seq_len:i + self.seq_len + self.pred_len, -1].values
            
            X_sequences.append(input_seq)
            y_sequences.append(target_seq)
        
        if not X_sequences:
            raise ValueError("資料不足以創建訓練序列")
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"準備了 {len(X_sequences)} 個訓練序列")
        
        # 提取特徵
        training_features = []
        
        for seq in X_sequences:
            # 創建 patches
            patches = self._create_patches(seq)
            
            # 提取特徵
            features = self._extract_patch_features(patches)
            training_features.append(features)
        
        training_features = np.array(training_features)
        
        # 標準化特徵
        training_features_scaled = self.scaler.fit_transform(training_features)
        
        # 為多步預測準備目標
        # 這裡我們簡化為預測序列的平均值
        training_targets = np.mean(y_sequences, axis=1)
        training_targets_scaled = self.target_scaler.fit_transform(training_targets.reshape(-1, 1)).flatten()
        
        # 訓練集成模型
        self.ensemble_model.fit(training_features_scaled, training_targets_scaled)
        
        self.is_fitted = True
        logger.info("PatchTST 模型訓練完成")
        
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        horizon: int = None
    ) -> np.ndarray:
        """
        進行預測
        
        Args:
            X: 輸入特徵資料
            horizon: 預測時間範圍（使用模型內建的 pred_len）
            
        Returns:
            預測結果數組
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")
        
        if len(X) < self.seq_len:
            raise ValueError(f"輸入資料長度 {len(X)} 小於所需序列長度 {self.seq_len}")
        
        # 取最後 seq_len 個數據點
        input_seq = X.iloc[-self.seq_len:].values
        
        # 創建 patches
        patches = self._create_patches(input_seq)
        
        # 提取特徵
        features = self._extract_patch_features(patches).reshape(1, -1)
        
        # 標準化特徵
        features_scaled = self.scaler.transform(features)
        
        # 預測
        prediction_scaled = self.ensemble_model.predict(features_scaled)
        
        # 反標準化
        prediction = self.target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
        
        # 生成多步預測（簡單重複預測值）
        # 實際應用中可以使用更複雜的策略
        multi_step_prediction = np.full(self.pred_len, prediction[0])
        
        return multi_step_prediction
    
    def predict_with_uncertainty(
        self, 
        X: pd.DataFrame, 
        horizon: int = None,
        confidence_level: float = 0.95
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
        
        # 估計預測標準差（簡化假設）
        prediction_std = np.std(predictions) * 0.1  # 簡化的不確定性估計
        
        # 計算信賴區間
        alpha = 1 - confidence_level
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
                'model_params': self.model_params,
                'is_fitted': self.is_fitted
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
            self.model_params = model_data['model_params']
            self.is_fitted = model_data['is_fitted']
            
            # 恢復模型參數
            for key, value in self.model_params.items():
                setattr(self, key, value)
            
            # 重新計算 n_patches
            self.n_patches = (self.seq_len - self.patch_len) // self.stride + 1
            
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
        mse = mean_squared_error(y_true[-len(predictions):], predictions)
        mae = mean_absolute_error(y_true[-len(predictions):], predictions)
        rmse = np.sqrt(mse)
        
        # 計算 MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true[-len(predictions):] - predictions) / y_true[-len(predictions):])) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }