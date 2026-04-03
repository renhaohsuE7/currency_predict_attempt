"""
PatchTST Transformer 預測範例

展示如何使用 HuggingFace PatchTST Transformer 進行貨幣預測
"""

import sys
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

from currency_predictor.models.factory import ModelFactory, create_patchtst_model
from currency_predictor.prediction.predictor import CurrencyPredictor
from currency_predictor.data.collectors import YahooFinanceCollector
from currency_predictor.data.storage import DataStorage
from currency_predictor.utils import setup_logging, ensure_directories


def example_direct_model_usage():
    """
    範例 1: 直接使用 PatchTSTTransformer 模型
    """
    print("=" * 60)
    print("範例 1: 直接使用 PatchTSTTransformer 模型")
    print("=" * 60)

    # 檢查 transformer 是否可用
    available_models = ModelFactory.get_available_models()
    if not available_models['patchtst_transformer']['available']:
        print("[WARN] PatchTSTTransformer 不可用，請確認已安裝 transformers 和 torch")
        return

    # 創建模型 (使用較小的參數以加快訓練)
    model = ModelFactory.create_model(
        'patchtst_transformer',
        seq_len=32,           # 較短的輸入序列
        pred_len=7,           # 預測 7 天
        patch_len=4,          # 較小的 patch
        stride=2,
        d_model=32,           # 較小的模型維度
        num_attention_heads=2,
        num_hidden_layers=1,  # 單層 transformer
    )

    print(f"模型類型: {model.model_type.value}")
    print(f"設備: {model.device}")
    print(f"參數: context_length={model.context_length}, prediction_length={model.prediction_length}")

    # 生成模擬資料 (500 筆)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='D')
    close_prices = 30 + np.cumsum(np.random.randn(500) * 0.1)

    df = pd.DataFrame({
        'Date': dates,
        'Close': close_prices,
        'Open': close_prices * (1 + np.random.randn(500) * 0.001),
        'High': close_prices * (1 + np.abs(np.random.randn(500)) * 0.005),
        'Low': close_prices * (1 - np.abs(np.random.randn(500)) * 0.005),
    })
    df.set_index('Date', inplace=True)

    print(f"\n資料形狀: {df.shape}")
    print(f"資料範圍: {df.index[0]} ~ {df.index[-1]}")

    # 訓練模型
    print("\n開始訓練 Transformer 模型...")
    model.fit(
        df,
        num_epochs=5,        # 較少的訓練輪數
        batch_size=16,
        learning_rate=1e-3,
        early_stopping_patience=3
    )

    print(f"訓練完成! is_fitted={model.is_fitted}")

    # 進行預測
    print("\n進行預測...")
    predictions = model.predict(df, horizon=7)
    print(f"預測結果: {predictions}")

    # 帶不確定性的預測
    print("\n進行不確定性預測...")
    result = model.predict_with_uncertainty(df, horizon=7, confidence_level=0.95)
    print(f"預測值: {result['predictions']}")
    print(f"標準差: {result['std']}")
    print(f"下界: {result['lower_bound']}")
    print(f"上界: {result['upper_bound']}")

    # 獲取模型資訊
    print("\n模型資訊:")
    info = model.get_model_info()
    for key, value in info.items():
        if key != 'training_history':
            print(f"  {key}: {value}")

    return model


def example_with_real_data():
    """
    範例 2: 使用真實貨幣資料進行預測
    """
    print("\n" + "=" * 60)
    print("範例 2: 使用真實貨幣資料進行 Transformer 預測")
    print("=" * 60)

    # 設置日誌和目錄
    setup_logging(level="INFO")
    ensure_directories(['data', 'models', 'results'])

    # 檢查 transformer 是否可用
    available_models = ModelFactory.get_available_models()
    if not available_models['patchtst_transformer']['available']:
        print("[WARN] PatchTSTTransformer 不可用")
        return

    # 收集資料
    symbol = "USDTWD=X"
    collector = YahooFinanceCollector()
    storage = DataStorage(base_dir="data")

    print(f"\n收集 {symbol} 資料...")

    # 嘗試載入現有資料
    clean_symbol = symbol.replace('=X', '')
    data = storage.load_raw_data(clean_symbol, "1y")

    if data is None:
        print("從 Yahoo Finance 收集資料...")
        data = collector.get_currency_data(symbol, period="1y", interval="1d")
        if data is not None:
            storage.save_raw_data(data, clean_symbol, "1y")
    else:
        print("使用已存在的資料")

    if data is None or len(data) < 100:
        print("[FAIL] 資料不足，無法進行訓練")
        return

    print(f"資料形狀: {data.shape}")
    print(f"資料範圍: {data.index[0]} ~ {data.index[-1]}")

    # 創建 Transformer 模型
    model = ModelFactory.create_model(
        'patchtst_transformer',
        seq_len=64,
        pred_len=7,
        patch_len=8,
        stride=4,
        d_model=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        dropout=0.1
    )

    # 訓練模型
    print("\n開始訓練...")
    try:
        model.fit(
            data,
            num_epochs=20,
            batch_size=16,
            learning_rate=1e-4,
            early_stopping_patience=5
        )

        # 預測
        predictions = model.predict(data, horizon=7)
        print(f"\n未來 7 天預測:")
        last_value = data['Close'].iloc[-1]
        print(f"最後已知價格: {last_value:.4f}")

        for i, pred in enumerate(predictions, 1):
            change = (pred - last_value) / last_value * 100
            print(f"  Day {i}: {pred:.4f} ({change:+.2f}%)")

        # 儲存模型
        model_path = f"results/models/{clean_symbol}_transformer"
        print(f"\n儲存模型至: {model_path}")
        model.save_model(model_path)

    except Exception as e:
        print(f"[FAIL] 訓練失敗: {str(e)}")


def example_compare_models():
    """
    範例 3: 比較 sklearn 和 transformer 版本
    """
    print("\n" + "=" * 60)
    print("範例 3: 比較 sklearn 和 transformer 版本")
    print("=" * 60)

    # 生成測試資料
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=300, freq='D')
    close_prices = 30 + np.cumsum(np.random.randn(300) * 0.1)

    df = pd.DataFrame({
        'Close': close_prices,
    }, index=dates)

    # 分割資料
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    print(f"訓練資料: {len(train_data)} 筆")
    print(f"測試資料: {len(test_data)} 筆")

    results = {}

    # 測試 sklearn 版本
    print("\n--- sklearn 版本 ---")
    sklearn_model = ModelFactory.create_model(
        'patchtst_sklearn',
        seq_len=32,
        pred_len=7,
        n_estimators=50
    )
    sklearn_model.fit(train_data, train_data['Close'])
    sklearn_pred = sklearn_model.predict(test_data)
    results['sklearn'] = sklearn_pred
    print(f"sklearn 預測: {sklearn_pred[:5]}...")

    # 測試 transformer 版本
    available_models = ModelFactory.get_available_models()
    if available_models['patchtst_transformer']['available']:
        print("\n--- transformer 版本 ---")
        transformer_model = ModelFactory.create_model(
            'patchtst_transformer',
            seq_len=32,
            pred_len=7,
            d_model=32,
            num_hidden_layers=1
        )
        transformer_model.fit(train_data, num_epochs=5, batch_size=16)
        transformer_pred = transformer_model.predict(test_data, horizon=7)
        results['transformer'] = transformer_pred
        print(f"transformer 預測: {transformer_pred[:5]}...")
    else:
        print("\n[WARN] transformer 版本不可用")

    # 比較結果
    print("\n--- 預測比較 ---")
    for name, pred in results.items():
        if len(pred) > 0:
            print(f"{name}: mean={np.mean(pred):.4f}, std={np.std(pred):.4f}")


def example_predictor_with_transformer():
    """
    範例 4: 使用 CurrencyPredictor 搭配 transformer
    """
    print("\n" + "=" * 60)
    print("範例 4: 使用 CurrencyPredictor 搭配 Transformer")
    print("=" * 60)

    # 設置
    setup_logging(level="INFO")
    ensure_directories(['data', 'models', 'results'])

    # 檢查 transformer 是否可用
    available_models = ModelFactory.get_available_models()
    if not available_models['patchtst_transformer']['available']:
        print("[WARN] transformer 不可用，使用 sklearn 版本")
        model_name = "patchtst_sklearn"
    else:
        model_name = "patchtst_transformer"

    # 創建預測器
    predictor = CurrencyPredictor(
        model_name=model_name,
        model_params={
            'seq_len': 64,
            'pred_len': 7,
            'd_model': 64,
            'num_hidden_layers': 2,
        },
        data_storage_path="data"
    )

    symbol = "USDTWD=X"

    # 收集資料
    print(f"\n收集 {symbol} 資料...")
    data_result = predictor.collect_and_store_data([symbol], period="1y")

    if data_result.get(symbol):
        print("[OK] 資料收集成功")

        # 訓練模型
        print("\n訓練模型...")
        training_result = predictor.train_model(symbol, period="1y")

        if training_result.get('training_completed'):
            print("[OK] 訓練完成")
            print(f"測試指標: {training_result.get('test_metrics', {})}")

            # 預測
            print("\n進行預測...")
            prediction_result = predictor.predict(symbol, horizon=7)

            if prediction_result.get('predictions') is not None:
                predictions = prediction_result['predictions']
                print(f"預測結果: {predictions}")
    else:
        print("[FAIL] 資料收集失敗")


def main():
    """主函數"""
    print("=" * 60)
    print("PatchTST Transformer 使用範例")
    print("=" * 60)

    # 列印可用模型
    print("\n可用模型:")
    ModelFactory.print_model_info()

    # 執行範例
    try:
        # 範例 1: 直接使用模型
        example_direct_model_usage()

        # 範例 2: 真實資料 (可選，需要網路)
        # example_with_real_data()

        # 範例 3: 比較模型
        example_compare_models()

        # 範例 4: 使用 CurrencyPredictor (可選)
        # example_predictor_with_transformer()

    except Exception as e:
        print(f"\n[ERROR] 執行失敗: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
