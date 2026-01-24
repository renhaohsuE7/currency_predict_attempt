"""
結果格式化器

提供預測結果的格式化和展示功能
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class StatusFormatter:
    """狀態格式化器"""

    @staticmethod
    def format_status(status: bool) -> str:
        """
        格式化狀態為表情符號

        Args:
            status: 狀態（True/False）

        Returns:
            格式化的狀態字符串
        """
        return '✅' if status else '❌'

    @staticmethod
    def format_stage_status(pipeline_status: Dict[str, bool]) -> Dict[str, str]:
        """
        格式化管道各階段的狀態

        Args:
            pipeline_status: 管道狀態字典

        Returns:
            格式化後的狀態字典
        """
        return {
            'data_collection': StatusFormatter.format_status(
                pipeline_status.get('data_collection', False)
            ),
            'model_training': StatusFormatter.format_status(
                pipeline_status.get('model_training', False)
            ),
            'prediction': StatusFormatter.format_status(
                pipeline_status.get('prediction', False)
            ),
            'results_saved': StatusFormatter.format_status(
                pipeline_status.get('results_saved', False)
            )
        }


class ResultFormatter:
    """
    結果格式化器

    負責格式化和展示預測結果
    """

    def __init__(self, use_logger: bool = True):
        """
        初始化結果格式化器

        Args:
            use_logger: 是否使用 logger 輸出（否則使用 print）
        """
        self.use_logger = use_logger
        self.logger = logging.getLogger(__name__) if use_logger else None

    def _output(self, message: str, level: str = 'info'):
        """
        輸出訊息

        Args:
            message: 訊息內容
            level: 日誌級別
        """
        if self.use_logger and self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
        else:
            print(message)

    def format_pipeline_summary(self, results: Dict[str, Any]):
        """
        格式化管道執行摘要

        Args:
            results: 管道執行結果字典
        """
        success = results.get('success', False)
        symbols = results.get('symbols', [])

        self._output(f"Pipeline execution completed, success: {success}")
        self._output(f"Processed currency pairs: {len(symbols)}")

    def format_stage_status(self, results: Dict[str, Any]):
        """
        格式化並顯示各階段狀態

        Args:
            results: 管道執行結果字典
        """
        pipeline_status = results.get('pipeline_status', {})
        formatted = StatusFormatter.format_stage_status(pipeline_status)

        self._output(f"Data collection: {formatted['data_collection']}")
        self._output(f"Model training: {formatted['model_training']}")
        self._output(f"Prediction execution: {formatted['prediction']}")
        self._output(f"Results saved: {formatted['results_saved']}")

    def format_predictions(self, results: Dict[str, Any]):
        """
        格式化並顯示預測結果

        Args:
            results: 管道執行結果字典
        """
        predictions = results.get('predictions', [])

        for prediction in predictions:
            symbol = prediction['symbol']

            if not prediction.get('error'):
                self._format_successful_prediction(symbol, prediction)
            else:
                self._format_failed_prediction(symbol, prediction)

    def _format_successful_prediction(self, symbol: str, prediction: Dict[str, Any]):
        """
        格式化成功的預測結果

        Args:
            symbol: 貨幣符號
            prediction: 預測結果字典
        """
        last_value = prediction.get('last_known_value', 0)
        first_pred = prediction.get('predictions', [0])[0] if prediction.get('predictions') else 0
        change = ((first_pred - last_value) / last_value * 100) if last_value != 0 else 0

        self._output(f"{symbol}: Predicted change {change:+.2f}%")

    def _format_failed_prediction(self, symbol: str, prediction: Dict[str, Any]):
        """
        格式化失敗的預測結果

        Args:
            symbol: 貨幣符號
            prediction: 預測結果字典
        """
        error = prediction.get('error', 'Unknown error')
        self._output(f"{symbol}: Prediction failed - {error}", level='error')

    def format_execution_summary(self, results: Dict[str, Any]):
        """
        格式化執行摘要

        Args:
            results: 管道執行結果字典
        """
        self._output("\n" + "="*50)
        self._output("EXECUTION SUMMARY")
        self._output("="*50)

        overall_success = results.get('success', False)
        status_symbol = StatusFormatter.format_status(overall_success)
        status_text = 'SUCCESS' if overall_success else 'FAILED'

        self._output(f"Overall execution: {status_symbol} {status_text}")

        if overall_success:
            self._output("\nCheck the following directories for results:")
            self._output("- 'results/' : Prediction results and reports")
            self._output("- 'models/'  : Trained models")
            self._output("- 'data/'    : Collected currency data")

    def format_complete_results(self, results: Dict[str, Any]):
        """
        格式化完整的執行結果

        Args:
            results: 管道執行結果字典
        """
        self.format_pipeline_summary(results)
        self.format_stage_status(results)
        self.format_predictions(results)
        self.format_execution_summary(results)

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成文字報告

        Args:
            results: 管道執行結果字典

        Returns:
            報告文字
        """
        report_lines = []

        # 標題
        report_lines.append("="*60)
        report_lines.append("CURRENCY PREDICTION EXECUTION REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # 執行狀態
        success = results.get('success', False)
        report_lines.append(f"Overall Status: {'SUCCESS' if success else 'FAILED'}")
        report_lines.append("")

        # 階段狀態
        pipeline_status = results.get('pipeline_status', {})
        formatted = StatusFormatter.format_stage_status(pipeline_status)

        report_lines.append("Pipeline Stages:")
        report_lines.append(f"  Data Collection: {formatted['data_collection']}")
        report_lines.append(f"  Model Training: {formatted['model_training']}")
        report_lines.append(f"  Prediction: {formatted['prediction']}")
        report_lines.append(f"  Results Saved: {formatted['results_saved']}")
        report_lines.append("")

        # 預測結果
        predictions = results.get('predictions', [])
        report_lines.append(f"Predictions ({len(predictions)} currency pairs):")

        for prediction in predictions:
            symbol = prediction['symbol']
            if not prediction.get('error'):
                last_value = prediction.get('last_known_value', 0)
                first_pred = prediction.get('predictions', [0])[0] if prediction.get('predictions') else 0
                change = ((first_pred - last_value) / last_value * 100) if last_value != 0 else 0
                report_lines.append(f"  {symbol}: {change:+.2f}% change predicted")
            else:
                error = prediction.get('error', 'Unknown error')
                report_lines.append(f"  {symbol}: FAILED - {error}")

        report_lines.append("")
        report_lines.append("="*60)

        return "\n".join(report_lines)
