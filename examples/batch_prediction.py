"""
批次預測範例

展示如何使用已訓練的模型進行批次預測
適用於已有模型的情況，無需重新訓練
"""

import sys
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from currency_predictor.config.manager import ConfigManager
from currency_predictor.prediction import PredictionPipeline
from currency_predictor.utils import setup_logging, ensure_directories


def main():
    """批次預測範例主函數"""

    # 設置日誌
    setup_logging(level="INFO")
    print("="*60)
    print("Currency Predictor - Batch Prediction Example")
    print("="*60)
    print()

    # 確保必要目錄存在
    ensure_directories(['data', 'models', 'results'])

    # 配置（最小配置，因為我們使用已訓練的模型）
    config = {
        "model_name": "patchtst_sklearn",
        "data_storage_path": "data",
        "log_level": "INFO"
    }

    # 要預測的貨幣對列表
    symbols = ["USDTWD=X", "EURUSD=X", "GBPUSD=X"]

    # 對應的模型路徑
    # 注意：這些模型必須已經存在（可以先執行 single_currency_prediction.py 訓練模型）
    model_paths = {
        "USDTWD=X": "models/USDTWD_PatchTST.joblib",
        "EURUSD=X": "models/EURUSD_PatchTST.joblib",
        "GBPUSD=X": "models/GBPUSD_PatchTST.joblib"
    }

    print("Configuration:")
    print(f"  Symbols: {symbols}")
    print(f"  Model paths:")
    for symbol, path in model_paths.items():
        exists = Path(path).exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"    {symbol}: {path} {status}")
    print()

    # 檢查模型是否都存在
    missing_models = [symbol for symbol, path in model_paths.items() if not Path(path).exists()]
    if missing_models:
        print(f"[WARNING] Missing models for: {missing_models}")
        print(f"          Please train these models first using:")
        print(f"          python examples/single_currency_prediction.py")
        print()
        print("Continuing with available models...")
        # 只使用存在的模型
        symbols = [s for s in symbols if s not in missing_models]
        model_paths = {s: p for s, p in model_paths.items() if s not in missing_models}

        if not symbols:
            print("[ERROR] No models available for prediction!")
            return 1
        print()

    try:
        # 創建預測管道
        print("Creating prediction pipeline...")
        pipeline = PredictionPipeline(config)
        print("[OK] Pipeline created")
        print()

        # 執行批次預測
        print(f"Running batch prediction for {len(symbols)} symbols...")
        print("-" * 60)

        results = pipeline.run_batch_prediction(
            symbols=symbols,
            model_paths=model_paths,
            prediction_horizon=7  # 預測未來 7 天
        )

        print()
        print("="*60)
        print("Batch Prediction Results")
        print("="*60)
        print()

        # 顯示結果
        successful = 0
        failed = 0

        for result in results:
            symbol = result['symbol']
            print(f"Symbol: {symbol}")
            print("-" * 40)

            if not result.get('error'):
                successful += 1
                print(f"[OK] Prediction successful")

                # 顯示預測值
                predictions = result.get('predictions', [])
                if predictions is not None and len(predictions) > 0:
                    print(f"Predictions ({len(predictions)} days):")
                    # 顯示前 3 個預測值
                    for i, pred in enumerate(predictions[:3], 1):
                        print(f"  Day {i}: {pred:.4f}")
                    if len(predictions) > 3:
                        print(f"  ... ({len(predictions) - 3} more)")

                # 計算變化
                last_value = result.get('last_known_value', 0)
                if last_value and predictions is not None and len(predictions) > 0:
                    first_pred = predictions[0]
                    change = ((first_pred - last_value) / last_value * 100) if last_value != 0 else 0
                    print(f"Expected change: {change:+.2f}%")

                # 顯示不確定性（如果有）
                if 'uncertainty' in result:
                    print(f"Uncertainty: {result['uncertainty']}")
            else:
                failed += 1
                print(f"[FAIL] {result.get('error', 'Unknown error')}")

            print()

        # 摘要
        print("="*60)
        print("Batch Prediction Summary")
        print("="*60)
        print(f"Total symbols: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print()

        if successful > 0:
            print("Results saved to 'results/' directory")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n[ERROR] Batch prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    """
    使用說明:
    1. 確保已經訓練好模型（使用 single_currency_prediction.py）
    2. 確保模型檔案存在於 models/ 目錄
    3. 執行此腳本進行批次預測
    """
    exit_code = main()
    exit(exit_code)
