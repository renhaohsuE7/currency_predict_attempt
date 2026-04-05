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

from currency_predictor.data.storage import DataStorage


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

    def test_processed_directory_exists(self, temp_storage_dir):
        """測試處理後資料目錄已建立"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 確認 processed 目錄存在
        assert storage.processed_dir.exists()
        assert storage.processed_dir.is_dir()

    def test_list_available_data(self, temp_storage_dir, sample_currency_data):
        """測試列出可用資料"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 儲存多個資料
        storage.save_raw_data(sample_currency_data, 'USDTWD', '1y')
        storage.save_raw_data(sample_currency_data, 'EURUSD', '6mo')

        # 列出資料
        available = storage.list_available_data()
        assert len(available) >= 2

    def test_data_existence_via_load(self, temp_storage_dir, sample_currency_data):
        """測試透過 load_raw_data 檢查資料是否存在"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 資料不存在時返回 None
        assert storage.load_raw_data('USDTWD', '1y') is None

        # 儲存資料後可載入
        storage.save_raw_data(sample_currency_data, 'USDTWD', '1y')
        assert storage.load_raw_data('USDTWD', '1y') is not None

    def test_get_data_info(self, temp_storage_dir, sample_currency_data):
        """測試取得資料資訊"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 儲存資料
        storage.save_raw_data(sample_currency_data, 'USDTWD', '1y')

        # 找到儲存的檔案路徑
        files = list(storage.raw_dir.glob("USDTWD_1y_*.csv"))
        assert len(files) > 0

        # 取得資訊（get_data_info 接受 filepath 參數）
        info = storage.get_data_info(str(files[0]))
        assert info is not None
        assert 'filepath' in info
        assert 'size_bytes' in info
        assert 'rows' in info

    def test_list_available_data_structure(self, temp_storage_dir, sample_currency_data):
        """測試列出可用資料的結構"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 儲存資料
        storage.save_raw_data(sample_currency_data, 'USDTWD', '1y')

        # 確認結構包含 raw_files 和 processed_files
        available = storage.list_available_data()
        assert 'raw_files' in available
        assert 'processed_files' in available
        assert len(available['raw_files']) >= 1

    def test_load_nonexistent_data(self, temp_storage_dir):
        """測試載入不存在的資料"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 載入不存在的資料應返回 None
        data = storage.load_raw_data('NONEXISTENT', '1y')
        assert data is None

    def test_save_invalid_data(self, temp_storage_dir):
        """測試儲存無效資料"""
        storage = DataStorage(base_dir=str(temp_storage_dir))

        # 嘗試儲存 None（save_raw_data 內部 try/except 會捕獲 AttributeError）
        success = storage.save_raw_data(None, 'TEST', '1y')
        assert success is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
