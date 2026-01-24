"""
基本使用範例

展示如何使用 Currency Predictor 進行基本的貨幣預測
"""

import sys
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from currency_predictor.config.manager import ConfigManager
from currency_predictor.prediction import PredictionPipeline
from currency_predictor.reporting.formatter import ResultFormatter
from currency_predictor.utils import setup_logging, ensure_directories


def main():
    """基本使用範例主函數"""

    # 1. 設置日誌
    setup_logging(level="INFO")
    print("="*60)
    print("Currency Predictor - Basic Usage Example")
    print("="*60)
    print()

    # 2. 確保必要目錄存在
    ensure_directories(['data', 'models', 'results'])

    # 3. 載入配置
    config_manager = ConfigManager()
    config = config_manager.get_config()

    print(f"Model: {config_manager.get('model_name')}")
    print(f"Symbols: {config_manager.get_symbols()}")
    print(f"Prediction horizon: {config_manager.get_prediction_horizon()} days")
    print()

    # 4. 創建預測管道
    pipeline = PredictionPipeline(config, output_dir="results")

    # 5. 運行預測（使用少量貨幣對進行測試）
    test_symbols = config_manager.get_symbols()[:1]  # 只使用第一個貨幣對
    print(f"Running prediction for: {test_symbols}")
    print()

    results = pipeline.run_full_pipeline(
        symbols=test_symbols,
        prediction_horizon=config_manager.get_prediction_horizon(),
        save_results=True,
        force_retrain=False
    )

    # 6. 格式化並顯示結果
    formatter = ResultFormatter(use_logger=True)
    formatter.format_complete_results(results)

    # 7. 生成文字報告
    report = formatter.generate_report(results)
    print("\n" + report)

    return 0 if results.get('success', False) else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
