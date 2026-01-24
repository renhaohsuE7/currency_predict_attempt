"""
自定義配置使用範例

展示如何使用自定義配置進行貨幣預測
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


def create_custom_config():
    """創建自定義配置"""
    return {
        "model_name": "PatchTST",
        "model_params": {
            "seq_len": 60,       # 使用較短的序列長度
            "pred_len": 7,       # 預測7天
            "patch_len": 10,     # 調整 patch 大小
            "stride": 5,         # 調整步長
            "n_estimators": 50,  # 減少估計器數量以加快訓練
            "max_depth": 5,      # 減少深度
            "random_state": 42
        },
        "data_storage_path": "data",
        "log_level": "INFO",
        "data_collection": {
            "period": "6mo",     # 只使用6個月資料
            "interval": "1d",
            "force_update": False
        },
        "model_training": {
            "period": "6mo",
            "target_column": "Close",
            "feature_columns": None,
            "train_params": {
                "validation_split": 0.2
            }
        },
        "prediction": {
            "period": "6mo",
            "return_uncertainty": True
        },
        "symbols": [
            "USDTWD=X",        # 只預測一個貨幣對
        ],
        "prediction_horizon": 3  # 只預測3天
    }


def main():
    """自定義配置使用範例主函數"""

    # 1. 設置日誌
    setup_logging(level="INFO")
    print("="*60)
    print("Currency Predictor - Custom Configuration Example")
    print("="*60)
    print()

    # 2. 確保必要目錄存在
    ensure_directories(['data', 'models', 'results'])

    # 3. 創建自定義配置
    custom_config = create_custom_config()

    # 可以選擇性地儲存配置
    config_manager = ConfigManager(validate=False)
    config_manager.update(custom_config)
    # config_manager.save('custom_config.json')  # 取消註解以儲存

    print("Custom Configuration:")
    print(f"  Model: {custom_config['model_name']}")
    print(f"  Sequence Length: {custom_config['model_params']['seq_len']}")
    print(f"  Prediction Length: {custom_config['model_params']['pred_len']}")
    print(f"  Symbols: {custom_config['symbols']}")
    print(f"  Data Period: {custom_config['data_collection']['period']}")
    print()

    # 4. 創建預測管道（使用自定義配置）
    pipeline = PredictionPipeline(custom_config, output_dir="results")

    # 5. 運行預測
    results = pipeline.run_full_pipeline(
        symbols=custom_config['symbols'],
        prediction_horizon=custom_config['prediction_horizon'],
        save_results=True,
        force_retrain=False
    )

    # 6. 格式化並顯示結果
    formatter = ResultFormatter(use_logger=True)
    formatter.format_complete_results(results)

    print("\n" + "="*60)
    print("Custom configuration example completed!")
    print("="*60)

    return 0 if results.get('success', False) else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
