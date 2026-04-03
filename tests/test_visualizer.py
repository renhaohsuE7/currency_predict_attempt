"""
測試視覺化模組

測試 CurrencyVisualizer 的所有功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from currency_predictor.visualization import CurrencyVisualizer, setup_chinese_font


@pytest.fixture
def sample_data():
    """創建測試資料"""
    # 創建 30 天的模擬資料
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

    np.random.seed(42)
    base_price = 30.0
    data = {
        'Date': dates,
        'Open': base_price + np.random.randn(30) * 0.1,
        'High': base_price + np.abs(np.random.randn(30) * 0.2),
        'Low': base_price - np.abs(np.random.randn(30) * 0.2),
        'Close': base_price + np.random.randn(30) * 0.1,
        'Volume': np.random.randint(1000000, 10000000, 30)
    }

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    return df


@pytest.fixture
def visualizer(tmp_path):
    """創建視覺化器實例"""
    output_dir = tmp_path / "figures"
    return CurrencyVisualizer(output_dir=str(output_dir))


class TestSetupChineseFont:
    """測試中文字型設定"""

    def test_setup_without_custom_font(self):
        """測試使用系統字型"""
        result = setup_chinese_font()
        assert result is True

    def test_setup_with_invalid_font_path(self):
        """測試使用無效的字型路徑"""
        result = setup_chinese_font("/invalid/path/to/font.ttf")
        # 應該回退到系統字型
        assert result is True


class TestCurrencyVisualizer:
    """測試 CurrencyVisualizer 類別"""

    def test_init(self, tmp_path):
        """測試初始化"""
        output_dir = tmp_path / "figures"
        visualizer = CurrencyVisualizer(output_dir=str(output_dir))

        assert visualizer.output_dir == output_dir
        assert output_dir.exists()

    def test_init_with_custom_params(self, tmp_path):
        """測試使用自定義參數初始化"""
        output_dir = tmp_path / "figures"
        figsize = (14, 8)

        visualizer = CurrencyVisualizer(
            output_dir=str(output_dir),
            figsize=figsize,
            style='default'
        )

        assert visualizer.figsize == figsize

    def test_plot_price_history(self, visualizer, sample_data):
        """測試繪製價格歷史圖"""
        fig = visualizer.plot_price_history(
            df=sample_data,
            symbol="USDTWD=X"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_price_history_multiple_columns(self, visualizer, sample_data):
        """測試繪製多個欄位"""
        fig = visualizer.plot_price_history(
            df=sample_data,
            symbol="USDTWD=X",
            columns=['Open', 'Close', 'High', 'Low']
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_price_history_with_save(self, visualizer, sample_data):
        """測試儲存圖表"""
        save_path = "test_price_history.png"

        fig = visualizer.plot_price_history(
            df=sample_data,
            symbol="USDTWD=X",
            save_path=save_path
        )

        # 檢查檔案是否存在
        full_path = visualizer.output_dir / save_path
        assert full_path.exists()

        plt.close(fig)

    def test_plot_candlestick(self, visualizer, sample_data):
        """測試繪製 K 線圖"""
        fig = visualizer.plot_candlestick(
            df=sample_data,
            symbol="USDTWD=X"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_candlestick_missing_columns(self, visualizer):
        """測試缺少必要欄位時的錯誤處理"""
        df = pd.DataFrame({'Close': [1, 2, 3]})

        with pytest.raises(ValueError, match="must contain columns"):
            visualizer.plot_candlestick(df, "TEST")

    def test_plot_volume(self, visualizer, sample_data):
        """測試繪製成交量圖"""
        fig = visualizer.plot_volume(
            df=sample_data,
            symbol="USDTWD=X"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_volume_missing_column(self, visualizer):
        """測試缺少 Volume 欄位"""
        df = pd.DataFrame({'Close': [1, 2, 3], 'Open': [1, 2, 3]})

        with pytest.raises(ValueError, match="must contain 'Volume'"):
            visualizer.plot_volume(df, "TEST")

    def test_plot_price_and_volume(self, visualizer, sample_data):
        """測試繪製價格和成交量組合圖"""
        fig = visualizer.plot_price_and_volume(
            df=sample_data,
            symbol="USDTWD=X"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # 兩個子圖

        plt.close(fig)

    def test_plot_returns(self, visualizer, sample_data):
        """測試繪製報酬率圖"""
        fig = visualizer.plot_returns(
            df=sample_data,
            symbol="USDTWD=X",
            period='daily'
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_returns_different_periods(self, visualizer, sample_data):
        """測試不同週期的報酬率"""
        for period in ['daily', 'weekly', 'monthly']:
            fig = visualizer.plot_returns(
                df=sample_data,
                symbol="USDTWD=X",
                period=period
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_plot_moving_averages(self, visualizer, sample_data):
        """測試繪製移動平均線"""
        fig = visualizer.plot_moving_averages(
            df=sample_data,
            symbol="USDTWD=X",
            windows=[5, 10, 20]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_moving_averages_default_windows(self, visualizer, sample_data):
        """測試使用預設窗口"""
        fig = visualizer.plot_moving_averages(
            df=sample_data,
            symbol="USDTWD=X"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_comparison(self, visualizer, sample_data):
        """測試繪製多個貨幣對比圖"""
        # 創建兩個資料集
        data_dict = {
            'USDTWD=X': sample_data,
            'EURUSD=X': sample_data.copy()
        }

        fig = visualizer.plot_comparison(
            data_dict=data_dict,
            normalize=True
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_comparison_without_normalize(self, visualizer, sample_data):
        """測試不標準化的對比圖"""
        data_dict = {
            'USDTWD=X': sample_data,
            'EURUSD=X': sample_data.copy()
        }

        fig = visualizer.plot_comparison(
            data_dict=data_dict,
            normalize=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_prediction_results(self, visualizer, sample_data):
        """測試繪製預測結果"""
        # 創建預測資料
        actual = sample_data['Close'][-10:]
        predicted = actual + np.random.randn(10) * 0.05

        fig = visualizer.plot_prediction_results(
            actual=actual,
            predicted=pd.Series(predicted, index=actual.index),
            symbol="USDTWD=X"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_prediction_results_with_confidence_interval(self, visualizer, sample_data):
        """測試帶信心區間的預測結果"""
        actual = sample_data['Close'][-10:]
        predicted = actual + np.random.randn(10) * 0.05

        # 創建信心區間
        lower = predicted - 0.1
        upper = predicted + 0.1

        fig = visualizer.plot_prediction_results(
            actual=actual,
            predicted=pd.Series(predicted, index=actual.index),
            symbol="USDTWD=X",
            confidence_interval=(lower, upper)
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_dashboard(self, visualizer, sample_data):
        """測試創建儀表板"""
        fig = visualizer.create_dashboard(
            df=sample_data,
            symbol="USDTWD=X"
        )

        assert isinstance(fig, plt.Figure)
        # 儀表板應該有多個子圖
        assert len(fig.axes) >= 5

        plt.close(fig)

    def test_create_dashboard_with_save(self, visualizer, sample_data):
        """測試儲存儀表板"""
        save_path = "test_dashboard.png"

        fig = visualizer.create_dashboard(
            df=sample_data,
            symbol="USDTWD=X",
            save_path=save_path
        )

        full_path = visualizer.output_dir / save_path
        assert full_path.exists()

        plt.close(fig)

    def test_load_data_from_csv(self, visualizer, tmp_path, sample_data):
        """測試從 CSV 載入資料"""
        # 儲存測試資料
        data_file = tmp_path / "USDTWD=X.csv"
        sample_data.to_csv(data_file)

        # 創建新的視覺化器
        vis = CurrencyVisualizer(data_path=str(tmp_path))

        # 載入資料
        df = vis.load_data("USDTWD=X")

        assert len(df) == len(sample_data)
        assert list(df.columns) == list(sample_data.columns)

    def test_load_data_with_date_range(self, visualizer, tmp_path, sample_data):
        """測試載入指定日期範圍的資料"""
        # 儲存測試資料
        data_file = tmp_path / "USDTWD=X.csv"
        sample_data.to_csv(data_file)

        vis = CurrencyVisualizer(data_path=str(tmp_path))

        # 載入部分資料
        df = vis.load_data(
            "USDTWD=X",
            start_date="2024-01-10",
            end_date="2024-01-20"
        )

        assert len(df) <= len(sample_data)
        assert df.index[0] >= pd.to_datetime("2024-01-10")
        assert df.index[-1] <= pd.to_datetime("2024-01-20")

    def test_load_data_file_not_found(self, visualizer, tmp_path):
        """測試載入不存在的檔案"""
        vis = CurrencyVisualizer(data_path=str(tmp_path))

        with pytest.raises(FileNotFoundError):
            vis.load_data("NONEXISTENT")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
