"""
測試資料收集器模組
"""

import unittest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import tempfile
import shutil

# 添加 src 目錄到 Python 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from currency_predictor.data.collectors import YahooFinanceCollector
from currency_predictor.data.storage import DataStorage


class TestYahooFinanceCollector(unittest.TestCase):
    """測試 Yahoo Finance 資料收集器"""
    
    def setUp(self):
        """測試前的設定"""
        self.collector = YahooFinanceCollector()
    
    def test_init(self):
        """測試初始化"""
        self.assertIsInstance(self.collector.supported_pairs, list)
        self.assertIn('EURUSD=X', self.collector.supported_pairs)
        self.assertIn('USDTWD=X', self.collector.supported_pairs)
    
    @patch('yfinance.Ticker')
    def test_get_currency_data_success(self, mock_ticker_class):
        """測試成功取得資料"""
        # 模擬 yfinance 回應
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # 建立模擬資料
        mock_data = pd.DataFrame({
            'Open': [1.1000, 1.1050, 1.1100],
            'High': [1.1050, 1.1100, 1.1150],
            'Low': [1.0950, 1.1000, 1.1050],
            'Close': [1.1020, 1.1080, 1.1120],
            'Volume': [10000, 12000, 11000]
        }, index=pd.date_range('2025-09-26', periods=3, freq='D'))
        
        mock_ticker.history.return_value = mock_data
        
        # 執行測試
        result = self.collector.get_currency_data('EURUSD=X', period='3d')
        
        # 驗證結果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(col in result.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']))
        
        # 驗證方法調用
        mock_ticker_class.assert_called_once_with('EURUSD=X')
        mock_ticker.history.assert_called_once_with(period='3d', interval='1d')
    
    @patch('yfinance.Ticker')
    def test_get_currency_data_empty(self, mock_ticker_class):
        """測試取得空資料的情況"""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()  # 空的 DataFrame
        
        result = self.collector.get_currency_data('INVALID=X')
        
        self.assertIsNone(result)
    
    @patch('yfinance.Ticker')
    def test_get_currency_data_exception(self, mock_ticker_class):
        """測試異常處理"""
        mock_ticker_class.side_effect = Exception("網路錯誤")
        
        result = self.collector.get_currency_data('EURUSD=X')
        
        self.assertIsNone(result)
    
    def test_clean_data(self):
        """測試資料清理功能"""
        # 建立包含問題的測試資料
        dirty_data = pd.DataFrame({
            'Open': [1.1000, None, 1.1100, 1.1000],  # 包含 None
            'High': [1.1050, 1.1100, 1.1150, 1.1050],
            'Low': [1.0950, 1.1000, 1.1050, 1.0950],
            'Close': [1.1020, 1.1080, 1.1120, 1.1020],
            'Volume': [10000, 12000, 11000, 10000]
        }, index=['2025-09-26', '2025-09-27', '2025-09-28', '2025-09-26'])  # 重複的日期
        
        cleaned = self.collector._clean_data(dirty_data)
        
        # 驗證清理結果
        self.assertFalse(cleaned.isnull().any().any())  # 沒有空值
        self.assertEqual(len(cleaned), len(cleaned.drop_duplicates()))  # 沒有重複
        self.assertIsInstance(cleaned.index, pd.DatetimeIndex)  # 索引是日期時間格式
    
    def test_is_currency_pair(self):
        """測試 is_currency_pair 靜態方法"""
        self.assertTrue(YahooFinanceCollector.is_currency_pair('USDTWD=X'))
        self.assertTrue(YahooFinanceCollector.is_currency_pair('EURUSD=X'))
        self.assertFalse(YahooFinanceCollector.is_currency_pair('AAPL'))
        self.assertFalse(YahooFinanceCollector.is_currency_pair('TSLA'))
        self.assertFalse(YahooFinanceCollector.is_currency_pair('BTC-USD'))

    def test_default_symbols_backward_compat(self):
        """測試 supported_pairs 向後相容 alias"""
        self.assertIs(self.collector.supported_pairs, self.collector.default_symbols)

    @patch('yfinance.Ticker')
    def test_check_symbol_availability(self, mock_ticker_class):
        """測試符號可用性檢查"""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # 測試可用的符號
        mock_ticker.history.return_value = pd.DataFrame({'Close': [1.1]}, 
                                                       index=pd.date_range('2025-09-28', periods=1))
        result = self.collector.check_symbol_availability('EURUSD=X')
        self.assertTrue(result)
        
        # 測試不可用的符號
        mock_ticker.history.return_value = pd.DataFrame()
        result = self.collector.check_symbol_availability('INVALID=X')
        self.assertFalse(result)


class TestDataStorage(unittest.TestCase):
    """測試資料儲存模組"""
    
    def setUp(self):
        """測試前的設定"""
        # 建立臨時目錄用於測試
        self.test_dir = tempfile.mkdtemp()
        self.storage = DataStorage(base_dir=self.test_dir)
        
        # 建立測試資料
        self.test_data = pd.DataFrame({
            'Open': [1.1000, 1.1050, 1.1100],
            'High': [1.1050, 1.1100, 1.1150],
            'Low': [1.0950, 1.1000, 1.1050],
            'Close': [1.1020, 1.1080, 1.1120],
            'Volume': [10000, 12000, 11000]
        }, index=pd.date_range('2025-09-26', periods=3, freq='D'))
    
    def tearDown(self):
        """測試後的清理"""
        # 刪除臨時目錄
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_init(self):
        """測試初始化"""
        # 檢查目錄是否被建立
        self.assertTrue(self.storage.base_dir.exists())
        self.assertTrue(self.storage.raw_dir.exists())
        self.assertTrue(self.storage.processed_dir.exists())
    
    def test_save_and_load_raw_data(self):
        """測試儲存和載入原始資料"""
        # 儲存資料
        success = self.storage.save_raw_data(
            data=self.test_data,
            symbol='EURUSD=X',
            period='3d',
            metadata={'source': 'yahoo_finance', 'test': True}
        )
        
        self.assertTrue(success)
        
        # 載入資料
        loaded_data = self.storage.load_raw_data('EURUSD=X', '3d')
        
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), len(self.test_data))
        
        # 檢查資料內容 (忽略索引名稱差異)
        loaded_data.index.name = self.test_data.index.name
        pd.testing.assert_frame_equal(loaded_data, self.test_data, check_dtype=False, check_freq=False)
    
    def test_list_available_data(self):
        """測試列出可用資料"""
        # 先儲存一些資料
        self.storage.save_raw_data(self.test_data, 'EURUSD=X', '3d')
        
        # 列出可用資料
        available = self.storage.list_available_data()
        
        self.assertIn('raw_files', available)
        self.assertIn('processed_files', available)
        self.assertGreater(len(available['raw_files']), 0)
    
    def test_get_data_info(self):
        """測試取得資料檔案資訊"""
        # 先儲存資料
        self.storage.save_raw_data(self.test_data, 'EURUSD=X', '3d')
        
        # 取得檔案清單
        available = self.storage.list_available_data()
        filename = available['raw_files'][0]
        
        # 取得檔案資訊
        info = self.storage.get_data_info(f"raw/{filename}")
        
        self.assertIsNotNone(info)
        self.assertIn('filename', info)
        self.assertIn('rows', info)
        self.assertIn('columns', info)
        self.assertEqual(info['rows'], 3)
        self.assertEqual(info['columns'], 5)


if __name__ == '__main__':
    # 設定日誌
    import logging
    logging.basicConfig(level=logging.INFO)
    
    unittest.main(verbosity=2)