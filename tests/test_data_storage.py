"""
測試資料儲存模組

測試 DataStorage 類別的所有功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

from src.currency_predictor.data.storage import DataStorage


@pytest.fixture
def temp_storage_dir(tmp_path):
    """創建臨時儲存目錄"""
    storage_dir = tmp_path / "test_data"
    storage_dir.mkdir()
    yield storage_dir
    # 清理
    if storage_dir.exists():
        shutil.rmtree(storage_dir)


@pytest.fixture
def sample_currency_data():
    """創建樣本貨幣資料"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Open': np.random.uniform(30, 35, 100),
        'High': np.random.uniform(30, 35, 100),
        'Low': np.random.uniform(30, 35, 100),
        'Close': np.random.uniform(30, 35, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    return data


class TestDataStorage:
    """測試 DataStorage 類別"""

    def test_init(self, temp_storage_dir):
        """測試初始化"""
        storage = DataStorage(base_dir=str(temp_storage_dir))
        assert storage.base_dir.exists()
        assert (storage.base_dir / 'raw').exists()
        assert (storage.base_dir / 'processed').exists()

    def test_save_and_load_raw_data(self, temp_storage_dir, sample_currency_data):
        """測試儲存和載入原始資料"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 儲存資料
        success = storage.save_raw_data(sample_currency_data, 'USDTWD', '1y')
        assert success is True

        # 載入資料
        loaded_data = storage.load_raw_data('USDTWD', '1y')
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_currency_data)
        assert list(loaded_data.columns) == list(sample_currency_data.columns)

    def test_save_and_load_processed_data(self, temp_storage_dir, sample_currency_data):
        """測試儲存和載入處理後資料"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 儲存資料
        success = storage.save_processed_data(sample_currency_data, 'USDTWD', '1y')
        assert success is True

        # 載入資料
        loaded_data = storage.load_processed_data('USDTWD', '1y')
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_currency_data)

    def test_list_available_data(self, temp_storage_dir, sample_currency_data):
        """測試列出可用資料"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 儲存多個資料
        storage.save_raw_data(sample_currency_data, 'USDTWD', '1y')
        storage.save_raw_data(sample_currency_data, 'EURUSD', '6mo')

        # 列出資料
        available = storage.list_available_data()
        assert len(available) >= 2

    def test_data_exists(self, temp_storage_dir, sample_currency_data):
        """測試檢查資料是否存在"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 資料不存在
        assert storage.data_exists('USDTWD', '1y', 'raw') is False

        # 儲存資料
        storage.save_raw_data(sample_currency_data, 'USDTWD', '1y')

        # 資料存在
        assert storage.data_exists('USDTWD', '1y', 'raw') is True

    def test_get_data_info(self, temp_storage_dir, sample_currency_data):
        """測試取得資料資訊"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 儲存資料
        storage.save_raw_data(sample_currency_data, 'USDTWD', '1y')

        # 取得資訊
        info = storage.get_data_info('USDTWD', '1y', 'raw')
        assert info is not None
        assert 'file_path' in info
        assert 'file_size' in info
        assert 'last_modified' in info

    def test_delete_data(self, temp_storage_dir, sample_currency_data):
        """測試刪除資料"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 儲存資料
        storage.save_raw_data(sample_currency_data, 'USDTWD', '1y')
        assert storage.data_exists('USDTWD', '1y', 'raw') is True

        # 刪除資料
        success = storage.delete_data('USDTWD', '1y', 'raw')
        assert success is True
        assert storage.data_exists('USDTWD', '1y', 'raw') is False

    def test_load_nonexistent_data(self, temp_storage_dir):
        """測試載入不存在的資料"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 載入不存在的資料應返回 None
        data = storage.load_raw_data('NONEXISTENT', '1y')
        assert data is None

    def test_save_invalid_data(self, temp_storage_dir):
        """測試儲存無效資料"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 嘗試儲存 None
        success = storage.save_raw_data(None, 'TEST', '1y')
        assert success is False

        # 嘗試儲存空 DataFrame
        empty_df = pd.DataFrame()
        success = storage.save_raw_data(empty_df, 'TEST', '1y')
        assert success is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
