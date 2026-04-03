"""
Currency Prediction Main Script

簡潔的主程式，使用模組化組件運行貨幣預測管道
"""

import logging
import argparse

from currency_predictor.config.manager import ConfigManager
from currency_predictor.prediction import PredictionPipeline
from currency_predictor.reporting.formatter import ResultFormatter
from currency_predictor.visualization import CurrencyVisualizer
from currency_predictor.utils import setup_logging, ensure_directories


def main(visualize=False):
    """
    主函數 - 運行貨幣預測管道

    Args:
        visualize: 是否生成視覺化圖表
    """

    # 設置日誌
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    logger.info("Starting Currency Prediction Pipeline")

    # 確保必要目錄存在
    ensure_directories(['data', 'models', 'results', 'results/figures'])

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

        # 生成視覺化圖表（如果啟用）
        if visualize and results.get('success', False):
            logger.info("Generating visualization charts...")

            try:
                visualizer = CurrencyVisualizer(
                    data_path="data",
                    output_dir="results/figures"
                )

                symbols = config_manager.get_symbols()

                for symbol in symbols:
                    try:
                        # 載入資料
                        df = visualizer.load_data(symbol)

                        # 生成儀表板
                        visualizer.create_dashboard(
                            df=df,
                            symbol=symbol,
                            save_path=f"{symbol.replace('=X', '')}_dashboard.png"
                        )

                        logger.info(f"Dashboard created for {symbol}")

                    except FileNotFoundError:
                        logger.warning(f"Data file not found for {symbol}, skipping visualization")
                    except Exception as e:
                        logger.warning(f"Failed to create visualization for {symbol}: {e}")

                logger.info(f"Visualizations saved to: results/figures/")

            except Exception as e:
                logger.error(f"Visualization generation failed: {e}")
                # 視覺化失敗不影響主要流程

        return 0 if results.get('success', False) else 1

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    """應用程式入口點"""
    parser = argparse.ArgumentParser(
        description="Currency Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run prediction pipeline
  python main.py --visualize        # Run with visualization
  python main.py -v                 # Same as --visualize
        """
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Generate visualization charts after prediction'
    )

    args = parser.parse_args()

    exit_code = main(visualize=args.visualize)
    exit(exit_code)
