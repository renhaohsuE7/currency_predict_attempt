"""
基礎模型介面

定義所有時間序列預測模型的標準介面
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
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
    
    def __init__(self, model_name: str = "BaseModel", model_type: ModelType = ModelType.SKLEARN_BASED):
        """
        初始化基礎模型
        
        Args:
            model_name: 模型名稱
            model_type: 模型類型
        """
        self.model_name = model_name
        self.model_type = model_type
        self.is_fitted = False
        self.model_params = {}
        logger.info(f"{model_name} 模型已初始化 (類型: {model_type.value})")
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        訓練模型

        統一的 fit 介面:三個 PatchTST 實作(sklearn / huggingface / lightning)
        皆採此簽名。`validation_data` 為共通的選用驗證集;各實作專屬的訓練超參數
        (如 num_epochs / batch_size / learning_rate)以 **kwargs 傳入,不適用的實作
        會忽略,因此上層(CurrencyPredictor.train_model)可用同一條呼叫路徑驅動任一實作。

        Args:
            X: 訓練特徵資料
            y: 訓練目標資料
            validation_data: 驗證資料 (X_val, y_val)
            **kwargs: 各實作專屬的訓練參數(不支援者忽略)

        Returns:
            訓練完成的模型實例
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        X: pd.DataFrame, 
        horizon: int = 1
    ) -> np.ndarray:
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
        self, 
        X: pd.DataFrame, 
        horizon: int = 1,
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
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        取得模型資訊
        
        Returns:
            模型資訊字典
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'is_fitted': self.is_fitted,
            'model_params': self.model_params
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
            # 子類別需要實作具體的儲存邏輯
            logger.info(f"模型儲存至: {filepath}")
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
            # 子類別需要實作具體的載入邏輯
            logger.info(f"模型從 {filepath} 載入")
            return True
        except Exception as e:
            logger.error(f"模型載入失敗: {str(e)}")
            return False


class TimeSeriesModel(BaseModel):
    """時間序列模型的特殊基礎類別"""
    
    def __init__(self, model_name: str = "TimeSeriesModel"):
        super().__init__(model_name)
        self.sequence_length = None
        self.feature_columns = []
    
    def prepare_sequences(
        self, 
        data: pd.DataFrame, 
        sequence_length: int,
        target_column: str = 'Close'
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
            seq_x = data.iloc[i:i+sequence_length].values
            X.append(seq_x)
            
            # 目標值
            if target_column in data.columns:
                seq_y = data.iloc[i+sequence_length][target_column]
            else:
                seq_y = data.iloc[i+sequence_length, 0]  # 使用第一欄
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
            logger.warning(f"輸入資料長度 {len(X)} 小於所需序列長度 {self.sequence_length}")
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