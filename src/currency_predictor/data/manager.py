"""
資料管理器

封裝「收集 → 儲存 → 清理 → 特徵工程 → 切分」的資料流程,讓上層
(CurrencyPredictor Facade / PredictionEngine)只依賴本類別,不直接碰
collector / storage / processor。

儲存後端透過建構子注入(預設 DataStorage 檔案後端),這是日後接父專案
PostgreSQL adapter 的接縫 —— 只要新後端提供 `load_raw_data` /
`save_raw_data` 介面即可替換,毋須改動上層。本次不建 DB adapter(YAGNI)。
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from .collectors import YahooFinanceCollector
from .storage import DataStorage
from ..data_processor import DataProcessor

logger = logging.getLogger(__name__)


class DataManager:
    """資料收集 + 儲存 + 前處理的單一入口。"""

    def __init__(
        self,
        data_storage_path: str = "data",
        *,
        collector: Optional[YahooFinanceCollector] = None,
        storage: Optional[DataStorage] = None,
        processor: Optional[DataProcessor] = None,
    ):
        # 注入點:預設檔案後端;未來可換 DB 後端而不動上層。
        self.collector = collector or YahooFinanceCollector()
        self.storage = storage or DataStorage(base_dir=data_storage_path)
        self.processor = processor or DataProcessor()

    def collect_and_store(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        force_update: bool = False,
    ) -> Dict[str, bool]:
        """收集並儲存貨幣資料,回傳各符號的成功與否。"""
        results: Dict[str, bool] = {}

        for symbol in symbols:
            try:
                logger.info(f"開始收集 {symbol} 資料")

                # 已有資料且不強制更新 → 跳過
                if not force_update:
                    existing_data = self.storage.load_raw_data(
                        symbol.replace('=X', ''), period
                    )
                    if existing_data is not None:
                        logger.info(f"{symbol} 資料已存在，跳過收集")
                        results[symbol] = True
                        continue

                data = self.collector.get_currency_data(symbol, period, interval)

                if data is not None and not data.empty:
                    clean_symbol = symbol.replace('=X', '')
                    success = self.storage.save_raw_data(data, clean_symbol, period)
                    results[symbol] = success

                    if success:
                        logger.info(f"[OK] {symbol} 資料收集並儲存成功")
                    else:
                        logger.error(f"[FAIL] {symbol} 資料儲存失敗")
                else:
                    logger.error(f"[FAIL] 無法收集 {symbol} 資料")
                    results[symbol] = False

            except Exception as e:
                logger.error(f"收集 {symbol} 資料時發生錯誤: {str(e)}")
                results[symbol] = False

        return results

    def process(self, symbol: str, period: str) -> pd.DataFrame:
        """載入 → 清理 → 技術指標 → 滯後特徵,回傳處理後 DataFrame。

        訓練與預測共用同一前處理路徑(滯後 lags=[1, 2, 3])。
        """
        clean_symbol = symbol.replace('=X', '')
        raw_data = self.storage.load_raw_data(clean_symbol, period)

        if raw_data is None or raw_data.empty:
            raise ValueError(f"找不到 {symbol} 的資料")

        logger.info(f"載入 {symbol} 資料，共 {len(raw_data)} 筆")

        cleaned_data = self.processor.clean_data(raw_data)
        data_with_indicators = self.processor.create_technical_indicators(cleaned_data)
        return self.processor.create_lagged_features(data_with_indicators, lags=[1, 2, 3])

    def prepare_training_data(
        self,
        symbol: str,
        period: str = "1y",
        target_column: str = 'Close',
        feature_columns: Optional[List[str]] = None,
    ) -> tuple:
        """準備訓練資料,回傳 (X_train, y_train, X_test, y_test)(80/20 時間切分)。"""
        processed_data = self.process(symbol, period)

        if feature_columns is None:
            feature_columns = [col for col in processed_data.columns if col != target_column]

        X = processed_data[feature_columns]
        y = processed_data[target_column]

        split_idx = int(len(processed_data) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        logger.info(f"訓練資料: {len(X_train)} 筆，測試資料: {len(X_test)} 筆")
        return X_train, y_train, X_test, y_test
