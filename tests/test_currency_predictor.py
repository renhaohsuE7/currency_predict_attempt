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
    CurrencyDataCollector,
    DataProcessor,
    CurrencyPredictor,
    validate_currency_pair,
    format_currency_pair
)


class TestCurrencyDataCollector(unittest.TestCase):
    """Test cases for CurrencyDataCollector class."""
    
    def setUp(self):
        self.collector = CurrencyDataCollector()
    
    def test_init(self):
        """Test initialization of CurrencyDataCollector."""
        self.assertIsInstance(self.collector.supported_pairs, list)
        self.assertIn('EURUSD=X', self.collector.supported_pairs)
    
    @patch('yfinance.Ticker')
    def test_get_yahoo_finance_data_success(self, mock_ticker):
        """Test successful data retrieval from Yahoo Finance."""
        # Mock the yfinance response
        mock_data = pd.DataFrame({
            'Open': [1.1, 1.2],
            'High': [1.15, 1.25],
            'Low': [1.05, 1.15],
            'Close': [1.12, 1.22],
            'Volume': [1000, 1200]
        })
        
        mock_ticker.return_value.history.return_value = mock_data
        
        result = self.collector.get_yahoo_finance_data('EURUSD=X', period='1d')
        
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 2)
        mock_ticker.assert_called_once_with('EURUSD=X')
    
    @patch('yfinance.Ticker')
    def test_get_yahoo_finance_data_empty(self, mock_ticker):
        """Test handling of empty data from Yahoo Finance."""
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        
        result = self.collector.get_yahoo_finance_data('INVALID=X')
        
        self.assertTrue(result.empty)


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
        expected_indicators = ['MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'BB_Upper']
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
        
        # Create sample training data
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.randn(100, 5), 
                                   columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        self.y_train = pd.Series(np.random.randn(100))
        self.X_test = pd.DataFrame(np.random.randn(20, 5),
                                  columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        self.y_test = pd.Series(np.random.randn(20))
    
    def test_setup_default_models(self):
        """Test default model setup."""
        self.predictor.setup_default_models()
        
        expected_models = ['linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']
        for model_name in expected_models:
            self.assertIn(model_name, self.predictor.models)
    
    def test_train_model(self):
        """Test model training."""
        self.predictor.setup_default_models()
        
        self.predictor.train_model('linear_regression', self.X_train, self.y_train)
        
        self.assertIn('linear_regression', self.predictor.trained_models)
    
    def test_predict(self):
        """Test model prediction."""
        self.predictor.setup_default_models()
        self.predictor.train_model('linear_regression', self.X_train, self.y_train)
        
        predictions = self.predictor.predict('linear_regression', self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        self.predictor.setup_default_models()
        self.predictor.train_model('linear_regression', self.X_train, self.y_train)
        
        metrics = self.predictor.evaluate_model('linear_regression', self.X_test, self.y_test)
        
        expected_metrics = ['mse', 'rmse', 'mae', 'r2', 'mape']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))


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