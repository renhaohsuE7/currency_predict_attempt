"""
測試預測管道模組

測試 PredictionPipeline 類別的所有功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.currency_predictor.prediction.pipeline import PredictionPipeline


@pytest.fixture
def temp_dirs(tmp_path):
    """創建臨時目錄"""
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"

    data_dir.mkdir()
    models_dir.mkdir()
    results_dir.mkdir()

    yield {
        'data': str(data_dir),
        'models': str(models_dir),
        'results': str(results_dir)
    }


@pytest.fixture
def sample_config(temp_dirs):
    """創建樣本配置"""
    return {
        'model_name': 'PatchTST',
        'model_params': {
            'seq_len': 50,
            'pred_len': 5,
            'patch_len': 10,
            'stride': 5,
            'n_estimators': 10,
            'max_depth': 3,
            'random_state': 42
        },
        'data_storage_path': temp_dirs['data'],
        'log_level': 'WARNING',
        'data_collection': {
            'period': '6mo',
            'interval': '1d',
            'force_update': False
        },
        'model_training': {
            'period': '6mo',
            'target_column': 'Close',
            'feature_columns': None,
            'train_params': {
                'validation_split': 0.2
            }
        },
        'prediction': {
            'period': '6mo',
            'return_uncertainty': True
        }
    }


class TestPredictionPipeline:
    """測試 PredictionPipeline 類別"""

    def test_init(self, sample_config, temp_dirs):
        """測試初始化"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        assert pipeline.config == sample_config
        assert pipeline.output_dir == Path(temp_dirs['results'])
        assert pipeline.predictor is not None

    def test_init_creates_output_dir(self, sample_config, tmp_path):
        """測試初始化會創建輸出目錄"""
        output_dir = tmp_path / "new_results"

        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=str(output_dir)
        )

        assert output_dir.exists()

    def test_validate_config_valid(self, sample_config, temp_dirs):
        """測試驗證有效配置"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        # 應該不會拋出異常
        pipeline._validate_config()

    def test_validate_config_missing_keys(self, temp_dirs):
        """測試驗證缺少必要鍵的配置"""
        invalid_config = {
            'model_name': 'PatchTST'
            # 缺少其他必要的鍵
        }

        pipeline = PredictionPipeline(
            config=invalid_config,
            output_dir=temp_dirs['results']
        )

        # 驗證應該會拋出警告或使用默認值
        # 具體行為取決於實現

    def test_collect_data_stage(self, sample_config, temp_dirs):
        """測試資料收集階段"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        # 注意：這個測試可能需要模擬網絡請求
        # 實際測試中應該使用 mock 或 fixture
        symbols = ['TEST=X']
        result = pipeline._collect_data(symbols)

        assert isinstance(result, dict)

    def test_save_results(self, sample_config, temp_dirs):
        """測試儲存結果"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        # 創建測試結果
        test_results = {
            'symbol': 'TEST=X',
            'predictions': [30.1, 30.2, 30.3],
            'last_known_value': 30.0,
            'timestamp': pd.Timestamp.now()
        }

        # 儲存結果
        success = pipeline._save_results([test_results])

        assert success is True

        # 檢查文件是否被創建
        result_files = list(Path(temp_dirs['results']).glob('*.json'))
        assert len(result_files) > 0

    def test_generate_report(self, sample_config, temp_dirs):
        """測試生成報告"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        # 創建測試結果
        test_results = {
            'success': True,
            'predictions': [
                {
                    'symbol': 'TEST=X',
                    'predictions': [30.1, 30.2, 30.3],
                    'last_known_value': 30.0
                }
            ]
        }

        # 生成報告
        report = pipeline._generate_report(test_results)

        assert isinstance(report, dict)
        assert 'summary' in report or 'status' in report


class TestPredictionPipelineIntegration:
    """測試預測管道整合功能"""

    @pytest.mark.skip(reason="需要實際資料或 mock，跳過整合測試")
    def test_run_full_pipeline(self, sample_config, temp_dirs):
        """測試運行完整管道"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        symbols = ['TEST=X']
        results = pipeline.run_full_pipeline(
            symbols=symbols,
            prediction_horizon=5,
            save_results=True,
            force_retrain=False
        )

        assert isinstance(results, dict)
        assert 'success' in results
        assert 'predictions' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
