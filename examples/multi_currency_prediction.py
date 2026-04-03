"""
多貨幣預測範例

展示如何同時預測多個貨幣對並比較結果
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
    """多貨幣預測範例主函數"""

    # 1. 設置日誌
    setup_logging(level="INFO")
    print("="*60)
    print("Currency Predictor - Multi-Currency Prediction Example")
    print("="*60)
    print()

    # 2. 確保必要目錄存在
    ensure_directories(['data', 'models', 'results'])

    # 3. 設置要預測的貨幣對
    currency_pairs = [
        "USDTWD=X",   # 美元對台幣
        "EURUSD=X",   # 歐元對美元
        "GBPUSD=X",   # 英鎊對美元
        "USDJPY=X",   # 美元對日圓
    ]

    print(f"Will predict {len(currency_pairs)} currency pairs:")
    for i, pair in enumerate(currency_pairs, 1):
        print(f"  {i}. {pair}")
    print()

    # 4. 載入並修改配置
    config_manager = ConfigManager()
    config_manager.update({
        'symbols': currency_pairs,
        'prediction_horizon': 7  # 預測7天
    })

    config = config_manager.get_config()

    # 5. 創建預測管道
    pipeline = PredictionPipeline(config, output_dir="results")

    # 6. 運行預測
    print("Starting prediction pipeline...")
    print()

    results = pipeline.run_full_pipeline(
        symbols=currency_pairs,
        prediction_horizon=7,
        save_results=True,
        force_retrain=False
    )

    # 7. 分析和比較結果
    print("\n" + "="*60)
    print("PREDICTION COMPARISON")
    print("="*60)

    predictions = results.get('predictions', [])

    # 按預測變化排序
    successful_predictions = [
        p for p in predictions if not p.get('error')
    ]

    if successful_predictions:
        # 計算每個貨幣對的預測變化
        prediction_changes = []
        for pred in successful_predictions:
            symbol = pred['symbol']
            last_value = pred.get('last_known_value', 0)
            predictions_array = pred.get('predictions', [])
            first_pred = predictions_array[0] if predictions_array is not None and len(predictions_array) > 0 else 0
            change = ((first_pred - last_value) / last_value * 100) if last_value != 0 else 0

            prediction_changes.append({
                'symbol': symbol,
                'change': change,
                'last_value': last_value,
                'predicted_value': first_pred
            })

        # 排序（從大到小）
        prediction_changes.sort(key=lambda x: x['change'], reverse=True)

        print("\nPredicted Changes (sorted by magnitude):")
        print("-" * 60)
        for i, item in enumerate(prediction_changes, 1):
            symbol = item['symbol']
            change = item['change']
            last = item['last_value']
            predicted = item['predicted_value']

            status = "[UP]" if change > 0 else "[DN]" if change < 0 else "[--]"
            print(f"{i}. {status} {symbol:12s}: {change:+7.2f}% ({last:.4f} -> {predicted:.4f})")

        # 找出最大漲幅和跌幅
        print("\nHighlights:")
        print("-" * 60)
        if prediction_changes:
            biggest_gain = prediction_changes[0]
            biggest_loss = prediction_changes[-1]

            print(f"Biggest Expected Gain: {biggest_gain['symbol']} ({biggest_gain['change']:+.2f}%)")
            print(f"Biggest Expected Loss: {biggest_loss['symbol']} ({biggest_loss['change']:+.2f}%)")

    # 8. 顯示完整結果
    print("\n" + "="*60)
    print("FULL RESULTS")
    print("="*60)

    formatter = ResultFormatter(use_logger=True)
    formatter.format_complete_results(results)

    # 9. 生成文字報告
    report = formatter.generate_report(results)
    report_file = Path("results") / "multi_currency_report.txt"
    report_file.write_text(report, encoding='utf-8')
    print(f"\nReport saved to: {report_file}")

    return 0 if results.get('success', False) else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
