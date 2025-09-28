"""
貨幣預測執行器

提供貨幣匯率預測的核心功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path

from ..data.collectors import YahooFinanceCollector
from ..data.storage import DataStorage
from ..models.patchtst import PatchTST
from ..data_processor import DataProcessor

logger = logging.getLogger(__name__)


class CurrencyPredictor:
    """
    貨幣預測執行器
    
    整合資料收集、處理和模型預測功能
    """
    
    def __init__(
        self,
        model_name: str = "PatchTST",
        model_params: Optional[Dict[str, Any]] = None,
        data_storage_path: str = "data"
    ):
        """
        初始化預測器
        
        Args:
            model_name: 模型名稱
            model_params: 模型參數
            data_storage_path: 資料儲存路徑
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        
        # 初始化組件
        self.data_collector = YahooFinanceCollector()
        self.data_storage = DataStorage(base_dir=data_storage_path)
        self.data_processor = DataProcessor()
        
        # 初始化模型
        self.model = self._create_model()
        
        logger.info(f"貨幣預測器已初始化，使用模型: {model_name}")
    
    def _create_model(self):
        """創建指定的模型"""
        if self.model_name.lower() == "patchtst":
            return PatchTST(**self.model_params)
        else:
            raise ValueError(f"不支援的模型類型: {self.model_name}")
    
    def collect_and_store_data(
        self, 
        symbols: List[str], 
        period: str = "1y",
        interval: str = "1d",
        force_update: bool = False
    ) -> Dict[str, bool]:
        """
        收集並儲存貨幣資料
        
        Args:
            symbols: 貨幣對符號列表
            period: 資料期間
            interval: 資料間隔
            force_update: 是否強制更新資料
            
        Returns:
            各貨幣對的收集結果
        """
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"開始收集 {symbol} 資料")
                
                # 檢查是否已有資料且不需要強制更新
                if not force_update:
                    existing_data = self.data_storage.load_raw_data(
                        symbol.replace('=X', ''), period
                    )
                    if existing_data is not None:
                        logger.info(f"{symbol} 資料已存在，跳過收集")
                        results[symbol] = True
                        continue
                
                # 收集資料
                data = self.data_collector.get_currency_data(
                    symbol, period, interval
                )
                
                if data is not None and not data.empty:
                    # 儲存資料
                    clean_symbol = symbol.replace('=X', '')
                    success = self.data_storage.save_raw_data(
                        data, clean_symbol, period
                    )
                    results[symbol] = success
                    
                    if success:
                        logger.info(f"✅ {symbol} 資料收集並儲存成功")
                    else:
                        logger.error(f"❌ {symbol} 資料儲存失敗")
                else:
                    logger.error(f"❌ 無法收集 {symbol} 資料")
                    results[symbol] = False
                    
            except Exception as e:
                logger.error(f"收集 {symbol} 資料時發生錯誤: {str(e)}")
                results[symbol] = False
        
        return results
    
    def prepare_training_data(
        self, 
        symbol: str, 
        period: str = "1y",
        target_column: str = 'Close',
        feature_columns: Optional[List[str]] = None
    ) -> tuple:
        """
        準備訓練資料
        
        Args:
            symbol: 貨幣對符號
            period: 資料期間
            target_column: 目標欄位
            feature_columns: 特徵欄位列表
            
        Returns:
            (X_train, y_train, X_test, y_test) 或 (X, y)
        """
        # 載入原始資料
        clean_symbol = symbol.replace('=X', '')
        raw_data = self.data_storage.load_raw_data(clean_symbol, period)
        
        if raw_data is None or raw_data.empty:
            raise ValueError(f"找不到 {symbol} 的資料")
        
        logger.info(f"載入 {symbol} 資料，共 {len(raw_data)} 筆")
        
        # 資料清理和處理
        cleaned_data = self.data_processor.clean_data(raw_data)
        
        # 特徵工程 - 創建技術指標
        data_with_indicators = self.data_processor.create_technical_indicators(cleaned_data)
        
        # 創建滯後特徵 (使用較短的滯後期)
        processed_data = self.data_processor.create_lagged_features(data_with_indicators, lags=[1, 2, 3])
        
        # 準備特徵和目標
        if feature_columns is None:
            # 使用所有欄位except目標欄位作為特徵
            feature_columns = [col for col in processed_data.columns if col != target_column]
        
        X = processed_data[feature_columns]
        y = processed_data[target_column]
        
        # 簡單的時間分割（80% 訓練，20% 測試）
        split_idx = int(len(processed_data) * 0.8)
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"訓練資料: {len(X_train)} 筆，測試資料: {len(X_test)} 筆")
        
        return X_train, y_train, X_test, y_test
    
    def train_model(
        self, 
        symbol: str, 
        period: str = "1y",
        target_column: str = 'Close',
        feature_columns: Optional[List[str]] = None,
        **train_kwargs
    ) -> Dict[str, Any]:
        """
        訓練模型
        
        Args:
            symbol: 貨幣對符號
            period: 資料期間
            target_column: 目標欄位
            feature_columns: 特徵欄位
            **train_kwargs: 訓練參數
            
        Returns:
            訓練結果字典
        """
        try:
            # 準備訓練資料
            X_train, y_train, X_test, y_test = self.prepare_training_data(
                symbol, period, target_column, feature_columns
            )
            
            # 訓練模型
            logger.info(f"開始訓練 {self.model_name} 模型")
            self.model.fit(X_train, y_train, **train_kwargs)
            
            # 評估模型
            train_metrics = self._evaluate_model(X_train, y_train, "訓練")
            test_metrics = self._evaluate_model(X_test, y_test, "測試")
            
            training_results = {
                'symbol': symbol,
                'model_name': self.model_name,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'training_completed': True
            }
            
            logger.info("模型訓練完成")
            return training_results
            
        except Exception as e:
            logger.error(f"模型訓練失敗: {str(e)}")
            return {
                'symbol': symbol,
                'model_name': self.model_name,
                'error': str(e),
                'training_completed': False
            }
    
    def _evaluate_model(
        self, 
        X: pd.DataFrame, 
        y_true: pd.Series, 
        dataset_name: str
    ) -> Dict[str, float]:
        """評估模型性能"""
        try:
            if hasattr(self.model, 'evaluate'):
                metrics = self.model.evaluate(X, y_true)
            else:
                # 基本評估
                predictions = self.model.predict(X)
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                mse = mean_squared_error(y_true[-len(predictions):], predictions)
                mae = mean_absolute_error(y_true[-len(predictions):], predictions)
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
            
            logger.info(f"{dataset_name}集評估結果: MSE={metrics.get('mse', 0):.6f}")
            return metrics
            
        except Exception as e:
            logger.error(f"模型評估失敗: {str(e)}")
            return {}
    
    def predict(
        self, 
        symbol: str, 
        horizon: int = 7,
        period: str = "1y",
        return_uncertainty: bool = False
    ) -> Dict[str, Any]:
        """
        進行預測
        
        Args:
            symbol: 貨幣對符號
            horizon: 預測時間範圍
            period: 用於預測的歷史資料期間
            return_uncertainty: 是否返回不確定性
            
        Returns:
            預測結果字典
        """
        try:
            if not self.model.is_fitted:
                raise ValueError("模型尚未訓練，請先調用 train_model()")
            
            # 載入最新資料
            clean_symbol = symbol.replace('=X', '')
            raw_data = self.data_storage.load_raw_data(clean_symbol, period)
            
            if raw_data is None or raw_data.empty:
                raise ValueError(f"找不到 {symbol} 的資料")
            
            # 資料處理
            cleaned_data = self.data_processor.clean_data(raw_data)
            data_with_indicators = self.data_processor.create_technical_indicators(cleaned_data)
            processed_data = self.data_processor.create_lagged_features(data_with_indicators, lags=[1, 2, 3])
            
            # 進行預測
            if return_uncertainty and hasattr(self.model, 'predict_with_uncertainty'):
                prediction_result = self.model.predict_with_uncertainty(
                    processed_data, horizon
                )
            else:
                predictions = self.model.predict(processed_data, horizon)
                prediction_result = {'predictions': predictions}
            
            # 生成預測日期
            last_date = processed_data.index[-1]
            prediction_dates = [
                last_date + timedelta(days=i+1) for i in range(len(prediction_result['predictions']))
            ]
            
            result = {
                'symbol': symbol,
                'prediction_dates': prediction_dates,
                'last_known_date': last_date,
                'last_known_value': processed_data['Close'].iloc[-1],
                **prediction_result
            }
            
            logger.info(f"成功預測 {symbol}，預測 {len(prediction_dates)} 個時間點")
            return result
            
        except Exception as e:
            logger.error(f"預測 {symbol} 時發生錯誤: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'success': False
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
            # 確保目錄存在
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # 儲存模型
            success = self.model.save_model(filepath)
            
            if success:
                logger.info(f"模型已儲存至: {filepath}")
            
            return success
            
        except Exception as e:
            logger.error(f"儲存模型失敗: {str(e)}")
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
            success = self.model.load_model(filepath)
            
            if success:
                logger.info(f"模型已從 {filepath} 載入")
            
            return success
            
        except Exception as e:
            logger.error(f"載入模型失敗: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        取得模型資訊
        
        Returns:
            模型資訊字典
        """
        base_info = {
            'predictor_model': self.model_name,
            'model_params': self.model_params,
            'data_storage_path': str(self.data_storage.base_dir)
        }
        
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
            base_info.update(model_info)
        
        return base_info