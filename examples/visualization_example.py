"""
視覺化範例

展示如何使用 CurrencyVisualizer 進行資料視覺化
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from currency_predictor.visualization import CurrencyVisualizer
from currency_predictor.data.collectors import YahooFinanceCollector
from currency_predictor.utils import setup_logging, ensure_directories
import pandas as pd


def collect_sample_data():
    """收集示例資料"""
    print("Collecting sample currency data...")
    print("-" * 60)

    collector = YahooFinanceCollector()

    symbols = ["USDTWD=X", "EURUSD=X"]
    data_dict = {}

    for symbol in symbols:
        try:
            print(f"Downloading {symbol}...")
            df = collector.collect(symbol, period="3mo", interval="1d")

            if df is not None and not df.empty:
                print(f"  [OK] {len(df)} rows collected")
                data_dict[symbol] = df
            else:
                print(f"  [SKIP] No data")
        except Exception as e:
            print(f"  [FAIL] {e}")

    print()
    return data_dict


def example_basic_plots(visualizer, df, symbol):
    """基本圖表範例"""
    print("\n" + "="*60)
    print("1. Basic Plots Examples")
    print("="*60)
    print()

    # 價格歷史圖
    print("Creating price history chart...")
    fig = visualizer.plot_price_history(
        df=df,
        symbol=symbol,
        columns=['Close'],
        title=f'{symbol} 收盤價歷史',
        save_path=f'{symbol}_price_history.png'
    )
    print(f"[OK] Price history saved")
    print()

    # 價格和成交量組合圖
    print("Creating price and volume chart...")
    fig = visualizer.plot_price_and_volume(
        df=df,
        symbol=symbol,
        save_path=f'{symbol}_price_volume.png'
    )
    print(f"[OK] Price and volume chart saved")
    print()


def example_technical_analysis(visualizer, df, symbol):
    """技術分析圖表範例"""
    print("\n" + "="*60)
    print("2. Technical Analysis Examples")
    print("="*60)
    print()

    # 移動平均線
    print("Creating moving averages chart...")
    fig = visualizer.plot_moving_averages(
        df=df,
        symbol=symbol,
        windows=[5, 20, 60],
        save_path=f'{symbol}_moving_averages.png'
    )
    print(f"[OK] Moving averages chart saved")
    print()

    # 報酬率圖
    print("Creating returns chart...")
    fig = visualizer.plot_returns(
        df=df,
        symbol=symbol,
        period='daily',
        save_path=f'{symbol}_returns.png'
    )
    print(f"[OK] Returns chart saved")
    print()

    # K 線圖
    print("Creating candlestick chart...")
    try:
        # 只取最近 30 天的資料來繪製 K 線
        recent_df = df.tail(30)
        fig = visualizer.plot_candlestick(
            df=recent_df,
            symbol=symbol,
            save_path=f'{symbol}_candlestick.png'
        )
        print(f"[OK] Candlestick chart saved")
    except Exception as e:
        print(f"[SKIP] Candlestick chart: {e}")
    print()


def example_comparison(visualizer, data_dict):
    """多貨幣對比範例"""
    print("\n" + "="*60)
    print("3. Multi-Currency Comparison Example")
    print("="*60)
    print()

    if len(data_dict) < 2:
        print("[SKIP] Need at least 2 currencies for comparison")
        return

    print("Creating comparison chart...")
    fig = visualizer.plot_comparison(
        data_dict=data_dict,
        column='Close',
        normalize=True,
        title='貨幣對比 (標準化)',
        save_path='currency_comparison.png'
    )
    print(f"[OK] Comparison chart saved")
    print()


def example_prediction_visualization(visualizer, df, symbol):
    """預測結果視覺化範例"""
    print("\n" + "="*60)
    print("4. Prediction Visualization Example")
    print("="*60)
    print()

    # 使用最後 10 天的資料模擬預測
    print("Creating prediction results chart (simulated)...")

    actual = df['Close'].tail(10)

    # 模擬預測值（實際應該來自模型）
    import numpy as np
    np.random.seed(42)
    predicted = actual + np.random.randn(10) * 0.5

    # 計算信心區間
    std = actual.std()
    lower = predicted - 1.96 * std
    upper = predicted + 1.96 * std

    fig = visualizer.plot_prediction_results(
        actual=actual,
        predicted=pd.Series(predicted.values, index=actual.index),
        symbol=symbol,
        confidence_interval=(lower.values, upper.values),
        save_path=f'{symbol}_prediction.png'
    )
    print(f"[OK] Prediction chart saved")
    print()


def example_dashboard(visualizer, df, symbol):
    """儀表板範例"""
    print("\n" + "="*60)
    print("5. Dashboard Example")
    print("="*60)
    print()

    print("Creating comprehensive dashboard...")
    fig = visualizer.create_dashboard(
        df=df,
        symbol=symbol,
        save_path=f'{symbol}_dashboard.png'
    )
    print(f"[OK] Dashboard saved")
    print()


def main():
    """主函數"""

    # 設置日誌
    setup_logging(level="INFO")
    print("="*60)
    print("Currency Predictor - Visualization Example")
    print("="*60)
    print()

    # 確保目錄存在
    ensure_directories(['data', 'results/figures'])

    # 創建視覺化器
    print("Initializing visualizer...")
    visualizer = CurrencyVisualizer(
        output_dir="results/figures",
        figsize=(12, 6)
    )
    print("[OK] Visualizer initialized")
    print()

    # 收集資料
    data_dict = collect_sample_data()

    if not data_dict:
        print("[ERROR] No data collected. Please check your internet connection.")
        return 1

    # 選擇第一個貨幣對進行示範
    symbol = list(data_dict.keys())[0]
    df = data_dict[symbol]

    print(f"\nUsing {symbol} for examples")
    print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total rows: {len(df)}")

    # 執行各種範例
    example_basic_plots(visualizer, df, symbol)
    example_technical_analysis(visualizer, df, symbol)
    example_comparison(visualizer, data_dict)
    example_prediction_visualization(visualizer, df, symbol)
    example_dashboard(visualizer, df, symbol)

    # 總結
    print("\n" + "="*60)
    print("All Examples Completed!")
    print("="*60)
    print()
    print("Generated charts:")
    print("  - Price history")
    print("  - Price and volume")
    print("  - Moving averages")
    print("  - Returns distribution")
    print("  - Candlestick chart")
    print("  - Currency comparison")
    print("  - Prediction results")
    print("  - Comprehensive dashboard")
    print()
    print(f"All charts saved to: {visualizer.output_dir}")
    print()
    print("Tips:")
    print("  1. Open the saved PNG files to view the charts")
    print("  2. Adjust figsize in CurrencyVisualizer for different sizes")
    print("  3. Use custom font_path for better Chinese character display")
    print("  4. All charts support save_path parameter for custom naming")

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
