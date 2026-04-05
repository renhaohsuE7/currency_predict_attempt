"""
E2E Test: Error Recovery & Partial Failure

驗證各模組在部分失敗場景下的 graceful degradation。
使用 mock 模擬網路錯誤和資料缺失。

執行：uv run pytest tests/test_e2e_error_recovery.py -v
"""

from unittest.mock import patch, MagicMock

import pytest

from currency_predictor.prediction.predictor import CurrencyPredictor


class TestCollectPartialFailure:
    """Test data collection partial failure handling."""

    @patch('currency_predictor.prediction.predictor.YahooFinanceCollector')
    def test_collect_partial_failure(self, mock_collector_cls):
        """2 symbols, 1 fails → the other still succeeds."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector

        import pandas as pd
        import numpy as np

        # First symbol returns data, second raises
        good_data = pd.DataFrame({
            'Open': [30.0], 'High': [31.0], 'Low': [29.0],
            'Close': [30.5], 'Volume': [1000],
        })

        def side_effect(symbol, period, interval):
            if symbol == 'GOOD=X':
                return good_data
            raise ConnectionError("Network error")

        mock_collector.get_currency_data.side_effect = side_effect

        predictor = CurrencyPredictor(
            model_name='patchtst_sklearn',
            data_storage_path='/tmp/test_error_recovery',
        )

        results = predictor.collect_and_store_data(
            ['GOOD=X', 'BAD=X'], period='1y'
        )

        assert results['GOOD=X'] is True
        assert results['BAD=X'] is False

    @patch('currency_predictor.prediction.predictor.YahooFinanceCollector')
    def test_invalid_symbol_graceful(self, mock_collector_cls):
        """Invalid symbol returns False without crashing."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.get_currency_data.return_value = None

        predictor = CurrencyPredictor(
            model_name='patchtst_sklearn',
            data_storage_path='/tmp/test_error_recovery',
        )

        results = predictor.collect_and_store_data(
            ['INVALID_SYMBOL'], period='1y'
        )

        assert results['INVALID_SYMBOL'] is False


class TestTrainPredictErrors:
    """Test training and prediction error handling."""

    def test_train_failure_returns_error_dict(self):
        """train_model() with no data → returns error dict."""
        predictor = CurrencyPredictor(
            model_name='patchtst_sklearn',
            data_storage_path='/tmp/test_error_recovery_nonexistent',
        )

        result = predictor.train_model(
            symbol='NONEXISTENT=X', period='1y'
        )

        assert result['training_completed'] is False
        assert 'error' in result

    def test_predict_unfitted_model_error(self):
        """predict() on unfitted model → returns error dict."""
        predictor = CurrencyPredictor(
            model_name='patchtst_sklearn',
            data_storage_path='/tmp/test_error_recovery_nonexistent',
        )

        result = predictor.predict(symbol='USDTWD=X', horizon=7)

        assert 'error' in result


class TestComparerPartialFailure:
    """Test ModelComparer with partial model failures."""

    @patch('currency_predictor.prediction.comparer.ModelComparer._init_predictors')
    def test_comparer_one_model_fails(self, mock_init):
        """2 models, 1 fails → other still produces results."""
        from currency_predictor.prediction.comparer import ModelComparer

        # Create two mock predictors
        good_predictor = MagicMock()
        bad_predictor = MagicMock()

        good_predictor.collect_and_store_data.return_value = {'SYM': True}
        bad_predictor.collect_and_store_data.return_value = {'SYM': True}

        import numpy as np

        good_predictor.prepare_training_data.return_value = (
            MagicMock(), MagicMock(), MagicMock(),
            MagicMock(values=np.array([30.0, 30.1, 30.2, 30.3, 30.4]))
        )
        good_predictor.train_model.return_value = {
            'training_completed': True,
            'train_metrics': {'mse': 0.01, 'mae': 0.1, 'rmse': 0.1},
            'test_metrics': {'mse': 0.02, 'mae': 0.12, 'rmse': 0.14},
        }
        good_predictor.predict.return_value = {
            'predictions': np.array([30.5, 30.6, 30.7]),
        }

        # Bad predictor fails during training
        bad_predictor.prepare_training_data.return_value = (
            MagicMock(), MagicMock(), MagicMock(),
            MagicMock(values=np.array([30.0, 30.1, 30.2, 30.3, 30.4]))
        )
        bad_predictor.train_model.side_effect = RuntimeError("GPU out of memory")

        mock_init.return_value = {
            'patchtst_sklearn': good_predictor,
            'patchtst_huggingface': bad_predictor,
        }

        comparer = ModelComparer(
            model_names=['sklearn', 'huggingface'],
            config={},
        )
        comparer.predictors = mock_init.return_value

        results = comparer.compare(symbols=['SYM'], prediction_horizon=3)

        # Good model should have results
        sym_result = results['symbols_results']['SYM']
        good_result = sym_result['models']['patchtst_sklearn']
        assert good_result.get('training_completed') is True

        # Bad model should have error
        bad_result = sym_result['models']['patchtst_huggingface']
        assert 'error' in bad_result
