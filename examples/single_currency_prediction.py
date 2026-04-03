"""
單一貨幣預測範例

展示如何使用 CurrencyPredictor 類別進行低階的貨幣預測
包含：資料收集、模型訓練、預測、模型儲存
"""

import sys
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from currency_predictor.prediction import CurrencyPredictor
from currency_predictor.utils import setup_logging, ensure_directories


def main():
    """單一貨幣預測範例主函數"""

    # 設置日誌
    setup_logging(level="INFO")
    print("="*60)
    print("Currency Predictor - Single Currency Prediction Example")
    print("="*60)
    print()

    # 確保必要目錄存在
    ensure_directories(['data', 'models', 'results'])

    # 創建預測器（使用自定義參數）
    print("Creating predictor with custom parameters...")
    predictor = CurrencyPredictor(
        model_name="patchtst_sklearn",
        model_params={
            'seq_len': 168,      # 使用 168 小時（7天）的歷史資料
            'pred_len': 24,      # 預測 24 小時（1天）
            'patch_len': 16,     # Patch 長度
            'stride': 8,         # Patch 步長
            'n_estimators': 50,  # 隨機森林估計器數量
            'max_depth': 8,      # 決策樹最大深度
            'random_state': 42   # 隨機種子（確保可重現）
        }
    )
    print("[OK] Predictor created")
    print()

    # 要預測的貨幣對
    symbol = "USDTWD=X"
    print(f"Target currency pair: {symbol}")
    print()

    try:
        # 步驟 1: 收集資料
        print("Step 1: Collecting data...")
        print("-" * 60)
        data_result = predictor.collect_and_store_data(
            symbols=[symbol],
            period="1y"  # 收集最近 1 年的資料 (need enough data for seq_len=168)
        )

        # data_result is Dict[str, bool] where keys are symbols
        if data_result.get(symbol, False):
            print(f"[OK] Data collection successful")
            print(f"     Data stored for {symbol}")
        else:
            print(f"[FAIL] Data collection failed for {symbol}")
            return 1
        print()

        # 步驟 2: 訓練模型
        print("Step 2: Training model...")
        print("-" * 60)
        training_result = predictor.train_model(
            symbol=symbol,
            period="1y"  # Use same period as data collection
        )

        if training_result.get('training_completed'):
            print(f"[OK] Training completed")

            # 顯示訓練指標
            if training_result.get('test_metrics'):
                metrics = training_result['test_metrics']
                print(f"     Test RMSE: {metrics.get('rmse', 0):.6f}")
                print(f"     Test MAE: {metrics.get('mae', 0):.6f}")
                print(f"     Test R2: {metrics.get('r2', 0):.6f}")
        else:
            print(f"[FAIL] Training failed: {training_result.get('error', 'Unknown error')}")
            return 1
        print()

        # 步驟 3: 進行預測
        print("Step 3: Making predictions...")
        print("-" * 60)
        prediction_result = predictor.predict(
            symbol=symbol,
            horizon=7,              # 預測未來 7 天
            return_uncertainty=True # 返回不確定性估計
        )

        if not prediction_result.get('error'):
            print(f"[OK] Prediction successful")
            print(f"     Last known value: {prediction_result['last_known_value']:.4f}")

            predictions = prediction_result['predictions']
            print(f"     Predictions ({len(predictions)} days):")
            for i, pred in enumerate(predictions, 1):
                print(f"       Day {i}: {pred:.4f}")

            # 計算預測變化
            last_value = prediction_result['last_known_value']
            first_pred = predictions[0] if predictions is not None and len(predictions) > 0 else last_value
            change_percent = ((first_pred - last_value) / last_value * 100) if last_value != 0 else 0
            print(f"     Expected change: {change_percent:+.2f}%")

            # 顯示不確定性（如果有）
            if 'uncertainty' in prediction_result:
                uncertainty = prediction_result['uncertainty']
                print(f"     Uncertainty (std): {uncertainty}")
        else:
            print(f"[FAIL] Prediction failed: {prediction_result['error']}")
            return 1
        print()

        # 步驟 4: 儲存模型
        print("Step 4: Saving model...")
        print("-" * 60)
        model_filename = f"{symbol.replace('=X', '')}_PatchTST.joblib"
        model_path = f"models/{model_filename}"

        try:
            predictor.save_model(model_path)
            print(f"[OK] Model saved to: {model_path}")
        except Exception as e:
            print(f"[FAIL] Failed to save model: {str(e)}")
        print()

        # 完成
        print("="*60)
        print("Example completed successfully!")
        print("="*60)
        print()
        print("Next steps:")
        print("  1. Check 'results/' directory for output files")
        print("  2. Check 'models/' directory for saved models")
        print("  3. Check 'data/' directory for collected data")
        print(f"  4. You can now load this model using:")
        print(f"     predictor.load_model('{model_path}')")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Example execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
