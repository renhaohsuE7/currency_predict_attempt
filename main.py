"""
Currency Prediction Main Script

簡潔的主程式，使用模組化組件運行貨幣預測管道
"""

import logging

from currency_predictor.config.manager import ConfigManager
from currency_predictor.prediction import PredictionPipeline
from currency_predictor.reporting.formatter import ResultFormatter
from currency_predictor.utils import setup_logging, ensure_directories


def main():
    """主函數 - 運行貨幣預測管道"""

    # 設置日誌
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    logger.info("Starting Currency Prediction Pipeline")

    # 確保必要目錄存在
    ensure_directories(['data', 'models', 'results'])

    try:
        # 載入配置
        config_manager = ConfigManager()
        config = config_manager.get_config()

        # 創建並運行預測管道
        pipeline = PredictionPipeline(config, output_dir="results")

        results = pipeline.run_full_pipeline(
            symbols=config_manager.get_symbols(),
            prediction_horizon=config_manager.get_prediction_horizon(),
            save_results=True,
            force_retrain=False
        )

        # 格式化並顯示結果
        formatter = ResultFormatter(use_logger=True)
        formatter.format_complete_results(results)

        return 0 if results.get('success', False) else 1

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    """應用程式入口點"""
    exit_code = main()
    exit(exit_code)
