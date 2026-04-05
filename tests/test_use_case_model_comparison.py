"""
Use Case Test: Multi-Model Comparison

驗證 ModelComparer 使用真實 fixture 資料（兩個不同 config 的 sklearn 模型）
執行完整 compare 流程。

標記 @pytest.mark.slow — 訓練真實 sklearn 模型。
執行：uv run pytest -m slow tests/test_use_case_model_comparison.py -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from currency_predictor.prediction.comparer import ModelComparer
from currency_predictor.prediction.predictor import CurrencyPredictor
from currency_predictor.data.storage import DataStorage
from currency_predictor.reporting.formatter import ResultFormatter


@pytest.fixture
def comparison_setup(e2e_fixture_path, tmp_path):
    """Set up two sklearn predictors with fixture data pre-loaded.

    Creates two CurrencyPredictor instances with different n_estimators
    and injects fixture data into their shared DataStorage.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Inject fixture data
    df = pd.read_csv(e2e_fixture_path, index_col="Date", parse_dates=True)
    storage = DataStorage(base_dir=str(data_dir))
    storage.save_raw_data(df, "USDTWD", "1y")

    config = {
        'data_collection': {'period': '1y', 'interval': '1d'},
        'model_training': {'period': '1y', 'target_column': 'Close'},
        'data_storage_path': str(data_dir),
        'model_params': {
            'seq_len': 50,
            'pred_len': 5,
            'patch_len': 10,
            'stride': 5,
            'max_depth': 3,
            'random_state': 42,
        },
    }

    return config, str(data_dir)


@pytest.mark.slow
class TestModelComparisonWithRealData:
    """Test ModelComparer with two real sklearn models and fixture data."""

    def _create_comparer_with_injected_data(self, config, data_dir):
        """Create ModelComparer and patch collect_and_store_data to no-op."""
        # Both models use sklearn but with different n_estimators
        # We override _init_predictors to create custom configs
        comparer = ModelComparer.__new__(ModelComparer)
        comparer.config = config
        comparer.output_dir = __import__('pathlib').Path(data_dir) / "results"
        comparer.output_dir.mkdir(parents=True, exist_ok=True)
        comparer.model_names = ['patchtst_sklearn_10', 'patchtst_sklearn_50']

        # Create two predictors with different estimator counts
        params_10 = dict(config.get('model_params', {}), n_estimators=10)
        params_50 = dict(config.get('model_params', {}), n_estimators=50)

        p1 = CurrencyPredictor(
            model_name='patchtst_sklearn',
            model_params=params_10,
            data_storage_path=data_dir,
        )
        p2 = CurrencyPredictor(
            model_name='patchtst_sklearn',
            model_params=params_50,
            data_storage_path=data_dir,
        )

        comparer.predictors = {
            'patchtst_sklearn_10': p1,
            'patchtst_sklearn_50': p2,
        }
        return comparer

    def test_two_sklearn_configs_comparison(self, comparison_setup):
        """Two sklearn models → compare() returns valid structure."""
        config, data_dir = comparison_setup
        comparer = self._create_comparer_with_injected_data(config, data_dir)

        results = comparer.compare(
            symbols=['USDTWD=X'], prediction_horizon=5
        )

        assert 'symbols_results' in results
        assert 'overall_ranking' in results
        assert 'model_names' in results
        assert 'USDTWD=X' in results['symbols_results']

    def test_comparison_returns_valid_ranking(self, comparison_setup):
        """overall_ranking is non-empty and sorted by RMSE ascending."""
        config, data_dir = comparison_setup
        comparer = self._create_comparer_with_injected_data(config, data_dir)

        results = comparer.compare(
            symbols=['USDTWD=X'], prediction_horizon=5
        )

        ranking = results['overall_ranking']
        assert len(ranking) > 0
        # Sorted ascending by RMSE
        rmse_values = [r[1] for r in ranking]
        assert rmse_values == sorted(rmse_values)

    def test_comparison_best_model_per_symbol(self, comparison_setup):
        """Each symbol has a best_model selected."""
        config, data_dir = comparison_setup
        comparer = self._create_comparer_with_injected_data(config, data_dir)

        results = comparer.compare(
            symbols=['USDTWD=X'], prediction_horizon=5
        )

        sym_result = results['symbols_results']['USDTWD=X']
        assert sym_result.get('best_model') is not None

    def test_comparison_all_models_have_metrics(self, comparison_setup):
        """Each model result has test_metrics with rmse."""
        config, data_dir = comparison_setup
        comparer = self._create_comparer_with_injected_data(config, data_dir)

        results = comparer.compare(
            symbols=['USDTWD=X'], prediction_horizon=5
        )

        sym_result = results['symbols_results']['USDTWD=X']
        for model_name, model_result in sym_result['models'].items():
            if model_result.get('training_completed'):
                assert 'test_metrics' in model_result
                assert 'rmse' in model_result['test_metrics']

    def test_comparison_report_generation(self, comparison_setup):
        """compare results → generate_comparison_report() produces markdown."""
        config, data_dir = comparison_setup
        comparer = self._create_comparer_with_injected_data(config, data_dir)

        results = comparer.compare(
            symbols=['USDTWD=X'], prediction_horizon=5
        )

        formatter = ResultFormatter(use_logger=False)
        report = formatter.generate_comparison_report(results)

        assert len(report) > 50
        assert 'USDTWD' in report or 'comparison' in report.lower() or 'model' in report.lower()
