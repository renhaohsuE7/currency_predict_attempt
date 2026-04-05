"""
Basic unit tests for the currency_predictor package.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from currency_predictor import (
    DataProcessor,
    CurrencyPredictor,
    validate_currency_pair,
    format_currency_pair
)
from currency_predictor.data.collectors import YahooFinanceCollector
from currency_predictor.prediction.predictor import _clean_symbol


class TestYahooFinanceCollector(unittest.TestCase):
    """Test cases for YahooFinanceCollector class."""

    def setUp(self):
        self.collector = YahooFinanceCollector()

    def test_init(self):
        """Test initialization of YahooFinanceCollector."""
        self.assertIsInstance(self.collector.supported_pairs, list)
        self.assertIn('EURUSD=X', self.collector.supported_pairs)

    @patch('currency_predictor.data.collectors.yf.Ticker')
    def test_get_currency_data_success(self, mock_ticker):
        """Test successful data retrieval from Yahoo Finance."""
        dates = pd.date_range('2024-01-01', periods=2, freq='D')
        mock_data = pd.DataFrame({
            'Open': [1.1, 1.2],
            'High': [1.15, 1.25],
            'Low': [1.05, 1.15],
            'Close': [1.12, 1.22],
            'Volume': [1000, 1200]
        }, index=dates)

        mock_ticker.return_value.history.return_value = mock_data

        result = self.collector.get_currency_data('EURUSD=X', period='1d')

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        mock_ticker.assert_called_once_with('EURUSD=X')

    @patch('currency_predictor.data.collectors.yf.Ticker')
    def test_get_currency_data_empty(self, mock_ticker):
        """Test handling of empty data from Yahoo Finance."""
        mock_ticker.return_value.history.return_value = pd.DataFrame()

        result = self.collector.get_currency_data('INVALID=X')

        self.assertIsNone(result)


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        self.processor = DataProcessor()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Add some duplicates and NaN values
        dirty_data = self.sample_data.copy()
        dirty_data.loc['2023-01-15'] = np.nan
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:5]])  # Add duplicates
        
        clean_data = self.processor.clean_data(dirty_data)
        
        self.assertFalse(clean_data.isnull().any().any())
        self.assertEqual(len(clean_data), len(clean_data.drop_duplicates()))
    
    def test_create_technical_indicators(self):
        """Test creation of technical indicators."""
        result = self.processor.create_technical_indicators(self.sample_data)
        
        # Check if technical indicators are created
        # MA_20 is skipped for 100-row data (adaptive window: max_rolling=15)
        expected_indicators = ['MA_5', 'MA_10', 'RSI', 'MACD', 'BB_Upper']
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)

        # Check if data types are numeric
        for indicator in expected_indicators:
            self.assertTrue(pd.api.types.is_numeric_dtype(result[indicator]))
    
    def test_prepare_features_target(self):
        """Test feature and target preparation."""
        processed_data = self.processor.create_technical_indicators(self.sample_data)
        
        X, y = self.processor.prepare_features_target(processed_data)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X.columns), 0)


class TestCurrencyPredictor(unittest.TestCase):
    """Test cases for CurrencyPredictor class."""

    def setUp(self):
        self.predictor = CurrencyPredictor()

    def test_init_creates_model(self):
        """Test that CurrencyPredictor creates a model instance on init."""
        self.assertIsNotNone(self.predictor.model)

    def test_model_is_accessible(self):
        """Test that predictor.model is a BaseModel subclass."""
        from currency_predictor.models.base import BaseModel
        self.assertIsNotNone(self.predictor.model)
        self.assertIsInstance(self.predictor.model, BaseModel)

    def test_model_name_validation(self):
        """Test that an invalid model name falls back to the default sklearn model."""
        predictor = CurrencyPredictor(model_name='nonexistent_model')
        # The predictor falls back to patchtst_sklearn when an invalid name is given
        self.assertIsNotNone(predictor.model)
        self.assertEqual(predictor.model.model_name, 'PatchTST_Sklearn')

    def test_predictor_attributes(self):
        """Test that data_processor and data_storage exist on the predictor."""
        self.assertIsNotNone(self.predictor.data_processor)
        self.assertIsNotNone(self.predictor.data_storage)


class TestCleanSymbol(unittest.TestCase):
    """Test _clean_symbol helper function."""

    def test_currency_pair_strips_suffix(self):
        """Currency pairs (=X) should have the suffix removed."""
        self.assertEqual(_clean_symbol('USDTWD=X'), 'USDTWD')
        self.assertEqual(_clean_symbol('EURUSD=X'), 'EURUSD')

    def test_stock_ticker_unchanged(self):
        """Stock tickers should remain unchanged."""
        self.assertEqual(_clean_symbol('AAPL'), 'AAPL')
        self.assertEqual(_clean_symbol('TSLA'), 'TSLA')

    def test_crypto_unchanged(self):
        """Crypto symbols should remain unchanged."""
        self.assertEqual(_clean_symbol('BTC-USD'), 'BTC-USD')


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_format_currency_pair(self):
        """Test currency pair formatting."""
        test_cases = [
            ('EURUSD', 'EURUSD=X'),
            ('EUR/USD', 'EURUSD=X'),
            ('eur usd', 'EURUSD=X'),
            ('EURUSD=X', 'EURUSD=X')
        ]
        
        for input_pair, expected in test_cases:
            result = format_currency_pair(input_pair)
            self.assertEqual(result, expected)
    
    def test_validate_currency_pair(self):
        """Test currency pair validation."""
        valid_pairs = ['EURUSD=X', 'GBPUSD', 'JPY/USD', 'AUD USD']
        invalid_pairs = ['INVALID', 'EUR', 'EURUSD123', '']
        
        for pair in valid_pairs:
            self.assertTrue(validate_currency_pair(pair), f"Should be valid: {pair}")
        
        for pair in invalid_pairs:
            self.assertFalse(validate_currency_pair(pair), f"Should be invalid: {pair}")


if __name__ == '__main__':
    unittest.main()