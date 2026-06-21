"""
貨幣預測執行器(Facade)

`CurrencyPredictor` 是薄門面,把職責委派給三個協作者:
  - DataManager      收集 / 儲存 / 前處理
  - ModelTrainer     建立 / 訓練 / 評估 / 存載模型
  - PredictionEngine 產生預測

公開方法簽名與既有屬性(`model` / `data_collector` / `data_storage` /
`data_processor`)保持不變,使現有 tests / examples / main.py / pipeline 照常運作。
"""

import logging
from typing import Any, Dict, List, Optional

from ..data.manager import DataManager
from .trainer import ModelTrainer
from .engine import PredictionEngine

logger = logging.getLogger(__name__)


class CurrencyPredictor:
    """
    貨幣預測執行器(Facade)

    整合資料收集、處理和模型預測功能。
    """

    def __init__(
        self,
        model_name: str = "patchtst_sklearn",
        model_params: Optional[Dict[str, Any]] = None,
        data_storage_path: str = "data",
        *,
        storage=None,
    ):
        """
        初始化預測器

        Args:
            model_name: 模型名稱 ('patchtst_sklearn' 或 'patchtst_transformer')
            model_params: 模型參數
            data_storage_path: 資料儲存路徑
            storage: 選用的儲存後端(需提供 load_raw_data/save_raw_data 介面);
                預設用檔案式 DataStorage。這是日後接父專案 PostgreSQL adapter 的接縫。
        """
        self.model_name = model_name
        self.model_params = model_params or {}

        # 協作者
        self.data_manager = DataManager(data_storage_path, storage=storage)
        self.trainer = ModelTrainer(model_name, self.model_params)
        self.engine = PredictionEngine(self.trainer, self.data_manager)

        # 向後相容的屬性別名(指向協作者持有的同一物件)
        self.data_collector = self.data_manager.collector
        self.data_storage = self.data_manager.storage
        self.data_processor = self.data_manager.processor
        self.model = self.trainer.model

        logger.info(f"貨幣預測器已初始化，使用模型: {model_name}")

    def collect_and_store_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        force_update: bool = False,
    ) -> Dict[str, bool]:
        """收集並儲存貨幣資料。"""
        return self.data_manager.collect_and_store(symbols, period, interval, force_update)

    def prepare_training_data(
        self,
        symbol: str,
        period: str = "1y",
        target_column: str = 'Close',
        feature_columns: Optional[List[str]] = None,
    ) -> tuple:
        """準備訓練資料,回傳 (X_train, y_train, X_test, y_test)。"""
        return self.data_manager.prepare_training_data(
            symbol, period, target_column, feature_columns
        )

    def train_model(
        self,
        symbol: str,
        period: str = "1y",
        target_column: str = 'Close',
        feature_columns: Optional[List[str]] = None,
        **train_kwargs,
    ) -> Dict[str, Any]:
        """訓練模型。"""
        try:
            X_train, y_train, X_test, y_test = self.data_manager.prepare_training_data(
                symbol, period, target_column, feature_columns
            )
        except Exception as e:
            logger.error(f"模型訓練失敗: {str(e)}")
            return {
                'symbol': symbol,
                'model_name': self.model_name,
                'error': str(e),
                'training_completed': False,
            }

        result = self.trainer.train(X_train, y_train, X_test, y_test, **train_kwargs)
        result['symbol'] = symbol
        return result

    def predict(
        self,
        symbol: str,
        horizon: int = 7,
        period: str = "1y",
        return_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        """進行預測。"""
        return self.engine.predict(symbol, horizon, period, return_uncertainty)

    def save_model(self, filepath: str) -> bool:
        """儲存模型。"""
        return self.trainer.save(filepath)

    def load_model(self, filepath: str) -> bool:
        """載入模型。"""
        return self.trainer.load(filepath)

    def get_model_info(self) -> Dict[str, Any]:
        """取得模型資訊。"""
        base_info = {
            'predictor_model': self.model_name,
            'model_params': self.model_params,
            'data_storage_path': str(self.data_storage.base_dir),
        }

        if hasattr(self.model, 'get_model_info'):
            base_info.update(self.model.get_model_info())

        return base_info
