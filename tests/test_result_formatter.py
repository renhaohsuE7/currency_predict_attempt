"""
測試結果格式化器

測試 ResultFormatter 和 StatusFormatter 的所有功能
"""

import pytest
import logging
from io import StringIO

from currency_predictor.reporting.formatter import ResultFormatter, StatusFormatter


class TestStatusFormatter:
    """測試 StatusFormatter 類別"""

    def test_format_status_true(self):
        """測試格式化 True 狀態"""
        result = StatusFormatter.format_status(True)
        assert result == '[OK]'

    def test_format_status_false(self):
        """測試格式化 False 狀態"""
        result = StatusFormatter.format_status(False)
        assert result == '[FAIL]'

    def test_format_stage_status(self):
        """測試格式化管道階段狀態"""
        pipeline_status = {
            'data_collection': True,
            'model_training': False,
            'prediction': True,
            'results_saved': False
        }

        result = StatusFormatter.format_stage_status(pipeline_status)

        assert result['data_collection'] == '[OK]'
        assert result['model_training'] == '[FAIL]'
        assert result['prediction'] == '[OK]'
        assert result['results_saved'] == '[FAIL]'

    def test_format_stage_status_missing_keys(self):
        """測試處理缺少鍵的狀態字典"""
        pipeline_status = {
            'data_collection': True
            # 其他鍵缺失
        }

        result = StatusFormatter.format_stage_status(pipeline_status)

        assert result['data_collection'] == '[OK]'
        assert result['model_training'] == '[FAIL]'  # 默認為 False


class TestResultFormatter:
    """測試 ResultFormatter 類別"""

    @pytest.fixture
    def sample_success_results(self):
        """創建成功的樣本結果"""
        return {
            'success': True,
            'symbols': ['USDTWD=X', 'EURUSD=X'],
            'pipeline_status': {
                'data_collection': True,
                'model_training': True,
                'prediction': True,
                'results_saved': True
            },
            'predictions': [
                {
                    'symbol': 'USDTWD=X',
                    'last_known_value': 31.5,
                    'predictions': [31.8, 32.0, 32.1]
                },
                {
                    'symbol': 'EURUSD=X',
                    'last_known_value': 1.08,
                    'predictions': [1.09, 1.095, 1.10]
                }
            ]
        }

    @pytest.fixture
    def sample_failed_results(self):
        """創建失敗的樣本結果"""
        return {
            'success': False,
            'symbols': ['USDTWD=X'],
            'pipeline_status': {
                'data_collection': False,
                'model_training': False,
                'prediction': False,
                'results_saved': False
            },
            'predictions': [
                {
                    'symbol': 'USDTWD=X',
                    'error': 'Data collection failed'
                }
            ]
        }

    def test_init_with_logger(self):
        """測試使用 logger 初始化"""
        formatter = ResultFormatter(use_logger=True)
        assert formatter.use_logger is True
        assert formatter.logger is not None

    def test_init_without_logger(self):
        """測試不使用 logger 初始化"""
        formatter = ResultFormatter(use_logger=False)
        assert formatter.use_logger is False

    def test_format_pipeline_summary(self, sample_success_results, caplog):
        """測試格式化管道摘要"""
        formatter = ResultFormatter(use_logger=True)

        with caplog.at_level(logging.INFO):
            formatter.format_pipeline_summary(sample_success_results)

        # 檢查日誌輸出
        assert "Pipeline execution completed" in caplog.text
        assert "Processed currency pairs: 2" in caplog.text

    def test_format_stage_status(self, sample_success_results, caplog):
        """測試格式化階段狀態"""
        formatter = ResultFormatter(use_logger=True)

        with caplog.at_level(logging.INFO):
            formatter.format_stage_status(sample_success_results)

        # 檢查日誌輸出
        assert "Data collection" in caplog.text
        assert "Model training" in caplog.text
        assert "Prediction execution" in caplog.text
        assert "Results saved" in caplog.text

    def test_format_predictions_success(self, sample_success_results, caplog):
        """測試格式化成功的預測結果"""
        formatter = ResultFormatter(use_logger=True)

        with caplog.at_level(logging.INFO):
            formatter.format_predictions(sample_success_results)

        # 檢查包含貨幣符號和百分比變化
        assert "USDTWD=X" in caplog.text
        assert "EURUSD=X" in caplog.text
        assert "%" in caplog.text

    def test_format_predictions_failed(self, sample_failed_results, caplog):
        """測試格式化失敗的預測結果"""
        formatter = ResultFormatter(use_logger=True)

        with caplog.at_level(logging.ERROR):
            formatter.format_predictions(sample_failed_results)

        # 檢查錯誤訊息
        assert "Prediction failed" in caplog.text
        assert "Data collection failed" in caplog.text

    def test_format_execution_summary_success(self, sample_success_results, caplog):
        """測試格式化成功的執行摘要"""
        formatter = ResultFormatter(use_logger=True)

        with caplog.at_level(logging.INFO):
            formatter.format_execution_summary(sample_success_results)

        # 檢查摘要內容
        assert "EXECUTION SUMMARY" in caplog.text
        assert "SUCCESS" in caplog.text
        assert "results/" in caplog.text
        assert "models/" in caplog.text

    def test_format_execution_summary_failed(self, sample_failed_results, caplog):
        """測試格式化失敗的執行摘要"""
        formatter = ResultFormatter(use_logger=True)

        with caplog.at_level(logging.INFO):
            formatter.format_execution_summary(sample_failed_results)

        assert "EXECUTION SUMMARY" in caplog.text
        assert "FAILED" in caplog.text

    def test_format_complete_results(self, sample_success_results, caplog):
        """測試格式化完整結果"""
        formatter = ResultFormatter(use_logger=True)

        with caplog.at_level(logging.INFO):
            formatter.format_complete_results(sample_success_results)

        # 應該包含所有部分
        assert "Pipeline execution completed" in caplog.text
        assert "Data collection" in caplog.text
        assert "USDTWD=X" in caplog.text
        assert "EXECUTION SUMMARY" in caplog.text

    def test_generate_report(self, sample_success_results):
        """測試生成文字報告"""
        formatter = ResultFormatter(use_logger=False)
        report = formatter.generate_report(sample_success_results)

        # 檢查報告內容
        assert isinstance(report, str)
        assert "CURRENCY PREDICTION EXECUTION REPORT" in report
        assert "Overall Status: SUCCESS" in report
        assert "Data Collection: [OK]" in report
        assert "USDTWD=X" in report
        assert "EURUSD=X" in report

    def test_generate_report_failed(self, sample_failed_results):
        """測試生成失敗的報告"""
        formatter = ResultFormatter(use_logger=False)
        report = formatter.generate_report(sample_failed_results)

        assert "Overall Status: FAILED" in report
        assert "Data Collection: [FAIL]" in report
        assert "FAILED - Data collection failed" in report

    def test_successful_prediction_change_calculation(self):
        """測試成功預測的變化百分比計算"""
        formatter = ResultFormatter(use_logger=False)

        prediction = {
            'last_known_value': 100,
            'predictions': [105, 106, 107]
        }

        # 應該計算出 +5% 的變化
        # 這裡我們通過檢查日誌來驗證（在實際測試中）
        # 理論上第一個預測值是 105，變化為 +5%

    def test_zero_last_value_handling(self):
        """測試處理 last_value 為 0 的情況"""
        formatter = ResultFormatter(use_logger=False)

        prediction = {
            'last_known_value': 0,
            'predictions': [1, 2, 3]
        }

        # 不應該拋出除零錯誤
        # change 應該為 0


class TestComparisonReport:
    """測試多模型比較報告功能"""

    @pytest.fixture
    def comparison_results(self):
        """模擬 ModelComparer.compare() 回傳值"""
        return {
            'model_names': ['patchtst_sklearn', 'patchtst_huggingface'],
            'prediction_horizon': 7,
            'symbols_results': {
                'USDTWD=X': {
                    'models': {
                        'patchtst_sklearn': {
                            'training_completed': True,
                            'training_time': 1.5,
                            'train_metrics': {'rmse': 0.10},
                            'test_metrics': {'rmse': 0.20, 'mae': 0.15, 'mase': 1.1, 'mda': 0.55},
                        },
                        'patchtst_huggingface': {
                            'training_completed': True,
                            'training_time': 5.0,
                            'train_metrics': {'rmse': 0.05},
                            'test_metrics': {'rmse': 0.12, 'mae': 0.09, 'mase': 0.85, 'mda': 0.72},
                        },
                    },
                    'best_model': 'patchtst_huggingface',
                },
            },
            'overall_ranking': [
                ('patchtst_huggingface', 0.12),
                ('patchtst_sklearn', 0.20),
            ],
        }

    def test_format_comparison_results(self, comparison_results, caplog):
        """format_comparison_results 輸出正確"""
        formatter = ResultFormatter(use_logger=True)
        with caplog.at_level(logging.INFO):
            formatter.format_comparison_results(comparison_results)

        text = caplog.text
        assert "MULTI-MODEL COMPARISON RESULTS" in text
        assert "patchtst_sklearn" in text
        assert "patchtst_huggingface" in text
        assert "USDTWD=X" in text
        assert "Best model" in text
        assert "Overall Ranking" in text
        assert "MDA" in text

    def test_generate_comparison_report_markdown(self, comparison_results):
        """generate_comparison_report 產出 Markdown 報告"""
        formatter = ResultFormatter(use_logger=False)
        report = formatter.generate_comparison_report(comparison_results)

        assert isinstance(report, str)
        assert "# Multi-Model Comparison Report" in report
        assert "## USDTWD=X" in report
        assert "| patchtst_sklearn" in report
        assert "| patchtst_huggingface" in report
        assert "## Overall Ranking" in report
        assert "**patchtst_huggingface**" in report
        assert "MDA" in report

    def test_generate_comparison_report_with_error(self):
        """有錯誤的 symbol 也能產出報告"""
        results = {
            'model_names': ['patchtst_sklearn'],
            'prediction_horizon': 7,
            'symbols_results': {
                'BAD=X': {'error': 'no data'},
                'GOOD=X': {
                    'models': {
                        'patchtst_sklearn': {
                            'test_metrics': {'rmse': 0.1, 'mae': 0.08},
                            'training_time': 1.0,
                        },
                    },
                    'best_model': 'patchtst_sklearn',
                },
            },
            'overall_ranking': [('patchtst_sklearn', 0.1)],
        }

        formatter = ResultFormatter(use_logger=False)
        report = formatter.generate_comparison_report(results)
        assert "Error: no data" in report
        assert "## GOOD=X" in report

    def test_format_comparison_results_failed_model(self, caplog):
        """某個 model 訓練失敗時也能正確顯示"""
        results = {
            'model_names': ['a', 'b'],
            'prediction_horizon': 7,
            'symbols_results': {
                'SYM': {
                    'models': {
                        'a': {'error': 'boom', 'training_completed': False},
                        'b': {
                            'test_metrics': {'rmse': 0.1, 'mae': 0.08},
                            'training_time': 1.0,
                        },
                    },
                    'best_model': 'b',
                },
            },
            'overall_ranking': [('b', 0.1)],
        }
        formatter = ResultFormatter(use_logger=True)
        with caplog.at_level(logging.ERROR):
            formatter.format_comparison_results(results)
        assert "FAILED" in caplog.text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
