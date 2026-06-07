"""
測試預測管道模組

測試 PredictionPipeline 類別和 convert_numpy_to_native
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from currency_predictor.prediction.pipeline import (
    PredictionPipeline,
    convert_numpy_to_native,
)


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
        'model_name': 'patchtst_sklearn',
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

    def test_pipeline_initializes_with_valid_status(self, sample_config, temp_dirs):
        """測試管道初始化後具有正確的 pipeline_status"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        # 驗證 pipeline_status 存在且所有階段初始為 False
        assert hasattr(pipeline, 'pipeline_status')
        assert isinstance(pipeline.pipeline_status, dict)
        assert pipeline.pipeline_status['data_collection'] is False
        assert pipeline.pipeline_status['model_training'] is False
        assert pipeline.pipeline_status['prediction'] is False
        assert pipeline.pipeline_status['results_saved'] is False

    def test_config_missing_keys_falls_back_to_defaults(self, temp_dirs):
        """缺少必要鍵的配置不應 crash：PredictionPipeline 以 config.get 取預設值。

        PredictionPipeline 沒有獨立的 _validate_config 步驟，建構子用
        config.get(...) 對缺漏鍵套用預設值，因此即使只給 model_name 也能
        成功建立 predictor 與初始化 pipeline_status，不會拋例外。
        """
        minimal_config = {
            "model_name": "patchtst_sklearn"
            # 刻意缺少 model_params / data_storage_path / 各階段設定
        }

        # 不應拋例外
        pipeline = PredictionPipeline(
            config=minimal_config,
            output_dir=temp_dirs["results"],
        )

        # predictor 用預設值建立成功
        assert pipeline.predictor is not None
        # 缺漏的 model_params 退回空 dict
        assert pipeline.predictor.model_params == {}
        # pipeline_status 初始化為全 False
        assert pipeline.pipeline_status == {
            "data_collection": False,
            "model_training": False,
            "prediction": False,
            "results_saved": False,
        }

    def test_collect_data_stage(self, sample_config, temp_dirs):
        """測試資料收集階段"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        # 注意：這個測試可能需要模擬網絡請求
        # 實際測試中應該使用 mock 或 fixture
        symbols = ['TEST=X']
        result = pipeline._collect_data_phase(symbols)

        assert isinstance(result, dict)

    def test_output_dir_exists(self, sample_config, temp_dirs):
        """測試 output_dir 存在且可用"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        assert pipeline.output_dir.exists()
        assert pipeline.output_dir.is_dir()

    def test_generate_report(self, sample_config, temp_dirs):
        """測試生成報告"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        # 創建測試結果
        test_results = {
            'success': True,
            'start_time': '2026-01-01T00:00:00',
            'end_time': '2026-01-01T01:00:00',
            'data_collection': {'TEST=X': True},
            'training': [
                {
                    'symbol': 'TEST=X',
                    'training_completed': True,
                    'test_metrics': {'rmse': 0.01}
                }
            ],
            'predictions': [
                {
                    'symbol': 'TEST=X',
                    'predictions': [30.1, 30.2, 30.3],
                    'last_known_value': 30.0
                }
            ]
        }

        # _generate_report 只需一個參數：results（run 目錄已含時間戳）
        pipeline._generate_report(test_results)

        # 驗證報告檔案已被產生
        report_file = Path(temp_dirs['results']) / "prediction_report.md"
        assert report_file.exists()

        # 驗證報告內容為字串
        content = report_file.read_text(encoding='utf-8')
        assert isinstance(content, str)
        assert '貨幣預測流程報告' in content


    def test_generate_report_with_failures(self, sample_config, temp_dirs):
        """測試有失敗項目的報告"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        test_results = {
            'start_time': '2026-01-01T00:00:00',
            'data_collection': {'SYM1=X': True, 'SYM2=X': False},
            'training': [
                {'symbol': 'SYM1=X', 'training_completed': True, 'test_metrics': {'rmse': 0.05}},
                {'symbol': 'SYM2=X', 'training_completed': False},
            ],
            'predictions': [
                {'symbol': 'SYM1=X', 'predictions': [1.0], 'last_known_value': 1.0},
                {'symbol': 'SYM2=X', 'error': 'no data'},
            ],
        }

        pipeline._generate_report(test_results)

        report_file = Path(temp_dirs['results']) / "prediction_report.md"
        content = report_file.read_text(encoding='utf-8')
        assert '[FAIL]' in content
        assert '[OK]' in content

    def test_save_results_phase(self, sample_config, temp_dirs):
        """測試結果儲存階段"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        pipeline_results = {
            'start_time': '2026-01-01T00:00:00',
            'end_time': '2026-01-01T01:00:00',
            'data_collection': {'TEST=X': True},
            'training': [{'symbol': 'TEST=X', 'training_completed': True, 'test_metrics': {'rmse': 0.01}}],
            'predictions': [
                {
                    'symbol': 'TEST=X',
                    'predictions': np.array([30.1, 30.2]),
                    'prediction_dates': ['2026-01-02', '2026-01-03'],
                    'last_known_value': np.float64(30.0),
                }
            ],
        }

        result = pipeline._save_results_phase(pipeline_results)
        assert result is True

        # Check that files were created
        result_file = Path(temp_dirs['results']) / "pipeline_results.json"
        assert result_file.exists()

    def test_save_predictions_csv(self, sample_config, temp_dirs):
        """測試儲存預測 CSV"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        predictions = [
            {
                'symbol': 'TEST=X',
                'predictions': [30.1, 30.2, 30.3],
                'prediction_dates': ['2026-01-02', '2026-01-03', '2026-01-04'],
            },
            {
                'symbol': 'FAIL=X',
                'error': 'no data',
            },
        ]

        pipeline._save_predictions_csv(predictions)

        csv_files = list(Path(temp_dirs['results']).glob("TEST*_predictions*.csv"))
        assert len(csv_files) == 1

    def test_log_pipeline_summary(self, sample_config, temp_dirs):
        """測試 pipeline summary logging"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        results = {
            'symbols': ['TEST=X'],
            'success': True,
            'pipeline_status': {
                'data_collection': True,
                'model_training': True,
                'prediction': True,
                'results_saved': True,
            },
        }

        pipeline._log_pipeline_summary(results)  # should not raise

    def test_from_config_file(self, tmp_path):
        """測試從配置檔案創建 pipeline"""
        config = {
            'model_name': 'patchtst_sklearn',
            'model_params': {},
            'data_storage_path': str(tmp_path / 'data'),
            'log_level': 'WARNING',
        }
        config_file = tmp_path / "pipeline_config.json"
        config_file.write_text(json.dumps(config))

        pipeline = PredictionPipeline.from_config_file(
            str(config_file),
            output_dir=str(tmp_path / 'results'),
        )
        assert pipeline.config['model_name'] == 'patchtst_sklearn'

    def test_run_batch_prediction_no_model(self, sample_config, temp_dirs):
        """測試 batch prediction with missing model path"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results']
        )

        results = pipeline.run_batch_prediction(
            symbols=['TEST=X'],
            model_paths={'TEST=X': '/nonexistent/model.joblib'},
            prediction_horizon=3,
        )

        assert isinstance(results, list)
        assert len(results) == 1


class TestConvertNumpyToNative:
    """測試 numpy 轉原生類型"""

    def test_ndarray(self):
        assert convert_numpy_to_native(np.array([1, 2, 3])) == [1, 2, 3]

    def test_int64(self):
        result = convert_numpy_to_native(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_float64(self):
        result = convert_numpy_to_native(np.float64(3.14))
        assert abs(result - 3.14) < 1e-6
        assert isinstance(result, float)

    def test_bool(self):
        result = convert_numpy_to_native(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_nested_dict(self):
        data = {'a': np.int64(1), 'b': np.array([2.0, 3.0])}
        result = convert_numpy_to_native(data)
        assert result == {'a': 1, 'b': [2.0, 3.0]}

    def test_nested_list(self):
        data = [np.int64(1), np.float64(2.5), 'text']
        result = convert_numpy_to_native(data)
        assert result == [1, 2.5, 'text']

    def test_passthrough_native(self):
        assert convert_numpy_to_native("hello") == "hello"
        assert convert_numpy_to_native(42) == 42


class TestRunTrainOnly:
    """測試 PredictionPipeline.run_train_only"""

    def test_train_only_returns_result_dict(self, sample_config, temp_dirs):
        """run_train_only 回傳結構正確"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline, '_collect_data_phase', return_value={'SYM=X': True}), \
             patch.object(pipeline, '_training_phase', return_value=[
                 {'symbol': 'SYM=X', 'training_completed': True, 'test_metrics': {'rmse': 0.1}}
             ]):
            result = pipeline.run_train_only(symbols=['SYM=X'])

        assert result['operation_type'] == 'train_only'
        assert result['success'] is True
        assert 'data_collection' in result
        assert 'training' in result
        assert 'start_time' in result
        assert 'end_time' in result
        # pipeline_status 結構正確
        ps = result['pipeline_status']
        assert ps['data_collection'] is True
        assert ps['model_training'] is True
        assert ps['prediction'] is True      # 不執行，視為通過
        assert ps['results_saved'] is True

    def test_train_only_saves_json(self, sample_config, temp_dirs):
        """run_train_only 儲存 pipeline_results.json"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline, '_collect_data_phase', return_value={'SYM=X': True}), \
             patch.object(pipeline, '_training_phase', return_value=[
                 {'symbol': 'SYM=X', 'training_completed': True}
             ]):
            pipeline.run_train_only(symbols=['SYM=X'])

        results_file = Path(temp_dirs['results']) / "pipeline_results.json"
        assert results_file.exists()

        data = json.loads(results_file.read_text())
        assert data['operation_type'] == 'train_only'

    def test_train_only_no_prediction_phase(self, sample_config, temp_dirs):
        """run_train_only 不呼叫 _prediction_phase"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline, '_collect_data_phase', return_value={}), \
             patch.object(pipeline, '_training_phase', return_value=[]), \
             patch.object(pipeline, '_prediction_phase') as mock_predict:
            pipeline.run_train_only(symbols=['SYM=X'])

        mock_predict.assert_not_called()

    def test_train_only_failure(self, sample_config, temp_dirs):
        """run_train_only 全部訓練失敗 → success=False"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline, '_collect_data_phase', return_value={'SYM=X': False}), \
             patch.object(pipeline, '_training_phase', return_value=[
                 {'symbol': 'SYM=X', 'training_completed': False, 'error': 'no data'}
             ]):
            result = pipeline.run_train_only(symbols=['SYM=X'])

        assert result['success'] is False
        assert result['pipeline_status']['data_collection'] is False
        assert result['pipeline_status']['model_training'] is False

    def test_train_only_exception_handling(self, sample_config, temp_dirs):
        """run_train_only 異常時回傳 error"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline, '_collect_data_phase', side_effect=RuntimeError("boom")):
            result = pipeline.run_train_only(symbols=['SYM=X'])

        assert result['success'] is False
        assert 'error' in result


class TestRunPredictOnly:
    """測試 PredictionPipeline.run_predict_only"""

    def test_predict_only_returns_result_dict(self, sample_config, temp_dirs):
        """run_predict_only 回傳結構正確"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline.predictor, 'load_model', return_value=True), \
             patch.object(pipeline, '_prediction_phase', return_value=[
                 {'symbol': 'SYM=X', 'predictions': [1.0, 2.0]}
             ]), \
             patch.object(pipeline, '_save_predictions_csv'), \
             patch.object(pipeline, '_generate_report'):
            result = pipeline.run_predict_only(
                symbols=['SYM=X'],
                model_paths={'SYM=X': '/tmp/model.joblib'},
            )

        assert result['operation_type'] == 'predict_only'
        assert result['success'] is True
        assert 'predictions' in result
        assert 'model_loading' in result
        # pipeline_status 結構正確
        ps = result['pipeline_status']
        assert ps['data_collection'] is True   # 不執行，視為通過
        assert ps['model_training'] is True    # 不執行，視為通過
        assert ps['prediction'] is True
        assert ps['results_saved'] is True

    def test_predict_only_no_training_phase(self, sample_config, temp_dirs):
        """run_predict_only 不呼叫 _training_phase"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline.predictor, 'load_model', return_value=True), \
             patch.object(pipeline, '_prediction_phase', return_value=[]), \
             patch.object(pipeline, '_training_phase') as mock_train, \
             patch.object(pipeline, '_save_predictions_csv'), \
             patch.object(pipeline, '_generate_report'):
            pipeline.run_predict_only(symbols=['SYM=X'], model_paths={'SYM=X': '/tmp/m'})

        mock_train.assert_not_called()

    def test_predict_only_model_not_found(self, sample_config, temp_dirs):
        """run_predict_only 找不到模型時記錄失敗"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline, '_prediction_phase', return_value=[
                 {'symbol': 'SYM=X', 'error': 'no model'}
             ]), \
             patch.object(pipeline, '_save_predictions_csv'), \
             patch.object(pipeline, '_generate_report'):
            result = pipeline.run_predict_only(symbols=['SYM=X'])

        assert result['model_loading']['SYM=X'] is False
        assert result['pipeline_status']['prediction'] is False
        assert result['success'] is False

    def test_predict_only_saves_json(self, sample_config, temp_dirs):
        """run_predict_only 儲存 pipeline_results.json"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline.predictor, 'load_model', return_value=True), \
             patch.object(pipeline, '_prediction_phase', return_value=[
                 {'symbol': 'SYM=X', 'predictions': [1.0]}
             ]), \
             patch.object(pipeline, '_save_predictions_csv'), \
             patch.object(pipeline, '_generate_report'):
            pipeline.run_predict_only(
                symbols=['SYM=X'],
                model_paths={'SYM=X': '/tmp/m'},
            )

        results_file = Path(temp_dirs['results']) / "pipeline_results.json"
        assert results_file.exists()

    def test_predict_only_exception_handling(self, sample_config, temp_dirs):
        """run_predict_only 異常時回傳 error"""
        pipeline = PredictionPipeline(
            config=sample_config,
            output_dir=temp_dirs['results'],
        )

        with patch.object(pipeline.predictor, 'load_model', side_effect=RuntimeError("boom")):
            result = pipeline.run_predict_only(symbols=['SYM=X'], model_paths={'SYM=X': '/tmp/m'})

        assert result['success'] is False
        assert 'error' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
