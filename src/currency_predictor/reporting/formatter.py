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
        return '[OK]' if status else '[FAIL]'

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
        predictions = prediction.get('predictions', [])
        first_pred = predictions[0] if predictions is not None and len(predictions) > 0 else 0
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
                predictions_array = prediction.get('predictions', [])
                first_pred = predictions_array[0] if predictions_array is not None and len(predictions_array) > 0 else 0
                change = ((first_pred - last_value) / last_value * 100) if last_value != 0 else 0
                report_lines.append(f"  {symbol}: {change:+.2f}% change predicted")
            else:
                error = prediction.get('error', 'Unknown error')
                report_lines.append(f"  {symbol}: FAILED - {error}")

        report_lines.append("")
        report_lines.append("="*60)

        return "\n".join(report_lines)

    # ------------------------------------------------------------------
    # 多模型比較報告
    # ------------------------------------------------------------------

    def format_comparison_results(self, comparison_results: Dict[str, Any]):
        """
        格式化並顯示多模型比較結果

        Args:
            comparison_results: ModelComparer.compare() 的回傳值
        """
        model_names = comparison_results.get('model_names', [])
        symbols_results = comparison_results.get('symbols_results', {})
        overall_ranking = comparison_results.get('overall_ranking', [])

        self._output("=" * 60)
        self._output("MULTI-MODEL COMPARISON RESULTS")
        self._output("=" * 60)
        self._output(f"Models: {', '.join(model_names)}")
        self._output("")

        for symbol, sym_data in symbols_results.items():
            if 'error' in sym_data:
                self._output(f"{symbol}: ERROR - {sym_data['error']}", level='error')
                continue

            self._output(f"--- {symbol} ---")
            models = sym_data.get('models', {})
            for mname, mresult in models.items():
                if mresult.get('error'):
                    self._output(f"  {mname}: FAILED - {mresult['error']}", level='error')
                else:
                    test_metrics = mresult.get('test_metrics', {})
                    rmse = test_metrics.get('rmse', '-')
                    mae = test_metrics.get('mae', '-')
                    mase_val = test_metrics.get('mase')
                    mda_val = test_metrics.get('mda')
                    t_time = mresult.get('training_time', 0)
                    n_origins = test_metrics.get('n_origins')
                    mase_str = f"  MASE={mase_val:.3f}" if isinstance(mase_val, (int, float)) else ""
                    mda_str = f"  MDA={mda_val:.1%}" if isinstance(mda_val, (int, float)) else ""
                    origins_str = f"  origins={int(n_origins)}" if isinstance(n_origins, (int, float)) and n_origins > 0 else ""
                    self._output(
                        f"  {mname}: RMSE={rmse:.6f}  MAE={mae:.6f}{mase_str}{mda_str}{origins_str}  "
                        f"time={t_time:.1f}s"
                        if isinstance(rmse, (int, float)) else
                        f"  {mname}: no metrics"
                    )
                    # Per-horizon RMSE (if available)
                    per_horizon = test_metrics.get('per_horizon', {})
                    if per_horizon:
                        h_parts = [f"h{h}={m.get('rmse', 0):.4f}" for h, m in sorted(per_horizon.items())]
                        self._output(f"    Per-horizon RMSE: {', '.join(h_parts)}")

            best = sym_data.get('best_model')
            if best:
                self._output(f"  Best model: {best}")
            self._output("")

        if overall_ranking:
            self._output("--- Overall Ranking (by avg RMSE) ---")
            for rank, (mname, avg_rmse) in enumerate(overall_ranking, 1):
                self._output(f"  #{rank} {mname}: avg RMSE={avg_rmse:.6f}")

    def generate_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        生成多模型比較的文字報告

        Args:
            comparison_results: ModelComparer.compare() 的回傳值

        Returns:
            Markdown 格式報告字串
        """
        lines: List[str] = []
        model_names = comparison_results.get('model_names', [])
        symbols_results = comparison_results.get('symbols_results', {})
        overall_ranking = comparison_results.get('overall_ranking', [])
        horizon = comparison_results.get('prediction_horizon', '?')

        lines.append("# Multi-Model Comparison Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Prediction Horizon: {horizon} days")
        lines.append(f"Models: {', '.join(model_names)}")
        lines.append("")

        # Per-symbol table
        for symbol, sym_data in symbols_results.items():
            lines.append(f"## {symbol}")
            if 'error' in sym_data:
                lines.append(f"Error: {sym_data['error']}")
                lines.append("")
                continue

            lines.append("")
            lines.append("| Model | RMSE | MAE | MASE | MDA | Training Time (s) |")
            lines.append("| --- | --- | --- | --- | --- | --- |")

            models = sym_data.get('models', {})
            for mname, mresult in models.items():
                if mresult.get('error'):
                    lines.append(f"| {mname} | FAILED | - | - | - | - |")
                else:
                    tm = mresult.get('test_metrics', {})
                    rmse = tm.get('rmse')
                    mae = tm.get('mae')
                    mase_val = tm.get('mase')
                    mda_val = tm.get('mda')
                    t_time = mresult.get('training_time', 0)
                    rmse_str = f"{rmse:.6f}" if isinstance(rmse, (int, float)) else "-"
                    mae_str = f"{mae:.6f}" if isinstance(mae, (int, float)) else "-"
                    mase_str = f"{mase_val:.3f}" if isinstance(mase_val, (int, float)) else "-"
                    mda_str = f"{mda_val:.1%}" if isinstance(mda_val, (int, float)) else "-"
                    lines.append(f"| {mname} | {rmse_str} | {mae_str} | {mase_str} | {mda_str} | {t_time:.1f} |")

            best = sym_data.get('best_model')
            if best:
                lines.append(f"\nBest model: **{best}**")
            lines.append("")

        # Overall ranking
        if overall_ranking:
            lines.append("## Overall Ranking")
            lines.append("")
            lines.append("| Rank | Model | Avg RMSE |")
            lines.append("| --- | --- | --- |")
            for rank, (mname, avg_rmse) in enumerate(overall_ranking, 1):
                lines.append(f"| {rank} | {mname} | {avg_rmse:.6f} |")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Backtesting results
    # ------------------------------------------------------------------

    def format_backtest_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Display backtest results to console/logger.

        Args:
            results: {symbol: {model_name: BacktestResult}}
        """
        for symbol, model_results in results.items():
            self._output(f"\n{'='*60}")
            self._output(f"Backtest Results: {symbol}")
            self._output(f"{'='*60}")

            for model_name, bt in model_results.items():
                self._output(f"\n  Model: {model_name}")
                self._output(f"  Strategy: {bt.strategy} | Folds: {bt.n_folds}")
                self._output(f"  Total training time: {bt.total_training_time:.1f}s")

                # Prediction metrics
                self._output(f"\n  Prediction Metrics (avg +/- std across folds):")
                for key in ['rmse', 'mae', 'mape', 'direction_accuracy']:
                    avg = bt.avg_prediction_metrics.get(key, 0)
                    std = bt.std_prediction_metrics.get(key, 0)
                    if key == 'mape':
                        self._output(f"    {key:>20s}: {avg:>10.2f}% +/- {std:.2f}%")
                    elif key == 'direction_accuracy':
                        self._output(f"    {key:>20s}: {avg:>10.1%} +/- {std:.1%}")
                    else:
                        self._output(f"    {key:>20s}: {avg:>10.6f} +/- {std:.6f}")

                # Per-horizon metrics (if available)
                if bt.avg_per_horizon_metrics:
                    self._output(f"\n  Per-Horizon RMSE (avg across folds):")
                    h_parts = [
                        f"h{h}={m.get('rmse', 0):.4f}"
                        for h, m in sorted(bt.avg_per_horizon_metrics.items())
                    ]
                    self._output(f"    {', '.join(h_parts)}")

                # Financial metrics
                self._output(f"\n  Financial Metrics (avg across folds):")
                fin = bt.avg_financial_metrics
                self._output(f"    {'Sharpe Ratio':>20s}: {fin.get('sharpe_ratio', 0):>10.3f}")
                self._output(f"    {'Max Drawdown':>20s}: {fin.get('max_drawdown', 0):>10.1%}")
                self._output(f"    {'Win Rate':>20s}: {fin.get('win_rate', 0):>10.1%}")
                self._output(f"    {'Profit Factor':>20s}: {fin.get('profit_factor', 0):>10.3f}")
                self._output(f"    {'Cumulative Return':>20s}: {fin.get('cumulative_return', 0):>10.1%}")
