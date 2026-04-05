"""
測試資料處理模組

測試 DataProcessor 類別的所有功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from currency_predictor.data_processor import DataProcessor


@pytest.fixture
def sample_raw_data():
    """創建樣本原始資料"""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    np.random.seed(42)

    price = 30 + np.cumsum(np.random.randn(200) * 0.5)
    data = pd.DataFrame({
        'Open': price + np.random.randn(200) * 0.1,
        'High': price + abs(np.random.randn(200) * 0.2),
        'Low': price - abs(np.random.randn(200) * 0.2),
        'Close': price,
        'Volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)

    return data


@pytest.fixture
def sample_data_with_nulls(sample_raw_data):
    """創建包含空值的樣本資料"""
    data = sample_raw_data.copy()
    # 添加一些 NaN
    data.loc[data.index[10], 'Close'] = np.nan
    data.loc[data.index[20], 'Volume'] = np.nan
    return data


class TestDataProcessor:
    """測試 DataProcessor 類別"""

    def test_init(self):
        """測試初始化"""
        processor = DataProcessor()
        assert processor.scaler is None
        assert processor.feature_columns == []

    def test_clean_data(self, sample_data_with_nulls):
        """測試資料清理"""
        processor = DataProcessor()
        cleaned = processor.clean_data(sample_data_with_nulls)

        # 檢查空值已被移除
        assert cleaned.isnull().sum().sum() == 0

        # 檢查沒有重複值
        assert cleaned.duplicated().sum() == 0

        # 檢查索引是 DatetimeIndex
        assert isinstance(cleaned.index, pd.DatetimeIndex)

        # 檢查資料已排序
        assert cleaned.index.is_monotonic_increasing

    def test_create_technical_indicators(self, sample_raw_data):
        """測試技術指標創建"""
        processor = DataProcessor()
        data_with_indicators = processor.create_technical_indicators(sample_raw_data)

        # 檢查新指標是否被添加
        expected_indicators = ['MA_5', 'MA_10', 'EMA_12', 'EMA_26', 'MACD', 'RSI']
        for indicator in expected_indicators:
            assert indicator in data_with_indicators.columns

        # 檢查指標值在合理範圍內
        assert data_with_indicators['RSI'].min() >= 0
        assert data_with_indicators['RSI'].max() <= 100

    def test_create_lagged_features(self, sample_raw_data):
        """測試滯後特徵創建"""
        processor = DataProcessor()
        data_with_lags = processor.create_lagged_features(
            sample_raw_data,
            lags=[1, 2, 3, 5]
        )

        # 檢查滯後特徵是否被添加
        assert 'Close_lag_1' in data_with_lags.columns
        assert 'Close_lag_2' in data_with_lags.columns
        assert 'Close_lag_3' in data_with_lags.columns
        assert 'Close_lag_5' in data_with_lags.columns

    def test_prepare_features_target(self, sample_raw_data):
        """測試準備特徵和目標變數"""
        processor = DataProcessor()

        # 先創建技術指標
        data_with_indicators = processor.create_technical_indicators(sample_raw_data)

        # 準備特徵和目標
        X, y = processor.prepare_features_target(
            data_with_indicators,
            target_column='Close',
            prediction_horizon=1
        )

        # 檢查返回的資料
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_scale_features(self, sample_raw_data):
        """測試特徵縮放"""
        processor = DataProcessor()

        # 創建特徵並分割為訓練和測試集
        features = sample_raw_data[['Open', 'High', 'Low', 'Close']]
        split = int(len(features) * 0.8)
        X_train = features.iloc[:split]
        X_test = features.iloc[split:]

        # 縮放
        scaled_train, scaled_test = processor.scale_features(X_train, X_test)

        # 檢查縮放後的資料
        assert isinstance(scaled_train, pd.DataFrame)
        assert isinstance(scaled_test, pd.DataFrame)
        assert scaled_train.shape == X_train.shape
        assert scaled_test.shape == X_test.shape
        assert processor.scaler is not None

    def test_train_test_split(self, sample_raw_data):
        """測試訓練測試集分割"""
        processor = DataProcessor()

        X = sample_raw_data[['Open', 'High', 'Low', 'Close']]
        y = sample_raw_data['Close']

        # 分割資料
        X_train, X_test, y_train, y_test = processor.train_test_split(
            X, y, test_size=0.2
        )

        # 檢查分割結果
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        # 檢查比例
        test_ratio = len(X_test) / len(X)
        assert 0.15 <= test_ratio <= 0.25  # 允許一些誤差

    def test_empty_dataframe(self):
        """測試處理空 DataFrame"""
        processor = DataProcessor()
        empty_df = pd.DataFrame()

        # 清理空資料應返回空 DataFrame
        cleaned = processor.clean_data(empty_df)
        assert len(cleaned) == 0

    def test_insufficient_data_for_indicators(self):
        """測試資料不足時的技術指標計算"""
        processor = DataProcessor()

        # 創建只有5筆資料的 DataFrame
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        small_data = pd.DataFrame({
            'Close': [30, 31, 32, 31, 30],
            'High': [31, 32, 33, 32, 31],
            'Low': [29, 30, 31, 30, 29],
            'Open': [30, 31, 32, 31, 30],
            'Volume': [1000000] * 5
        }, index=dates)

        # 應該能夠處理，但某些指標會有 NaN
        result = processor.create_technical_indicators(small_data)
        assert isinstance(result, pd.DataFrame)


class TestAdaptiveFeatureEngineering:
    """測試自適應特徵工程"""

    @pytest.fixture
    def short_data(self):
        """30 行短資料"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        np.random.seed(42)
        price = 30 + np.cumsum(np.random.randn(30) * 0.5)
        return pd.DataFrame({
            'Open': price + np.random.randn(30) * 0.1,
            'High': price + abs(np.random.randn(30) * 0.2),
            'Low': price - abs(np.random.randn(30) * 0.2),
            'Close': price,
            'Volume': np.random.randint(1000000, 5000000, 30),
        }, index=dates)

    @pytest.fixture
    def long_data(self):
        """300 行長資料"""
        dates = pd.date_range(start='2024-01-01', periods=300, freq='D')
        np.random.seed(42)
        price = 30 + np.cumsum(np.random.randn(300) * 0.5)
        return pd.DataFrame({
            'Open': price + np.random.randn(300) * 0.1,
            'High': price + abs(np.random.randn(300) * 0.2),
            'Low': price - abs(np.random.randn(300) * 0.2),
            'Close': price,
            'Volume': np.random.randint(1000000, 5000000, 300),
        }, index=dates)

    # --- _compute_adaptive_windows / _compute_adaptive_lags ---

    def test_adaptive_windows_short(self):
        """30 行：max_rolling = max(3, int(30*0.15)=4) = 4"""
        processor = DataProcessor()
        w = processor._compute_adaptive_windows(30)
        # MA windows: only those <= 4
        assert all(mw <= 4 for mw in w['ma_windows'])
        assert 20 not in w['ma_windows']
        assert 10 not in w['ma_windows']

    def test_adaptive_windows_long(self):
        """300 行：max_rolling = 45 → 所有預設窗口都在範圍內"""
        processor = DataProcessor()
        w = processor._compute_adaptive_windows(300)
        assert w['ma_windows'] == [5, 10, 20]
        assert w['rsi_window'] == 10
        assert w['bb_window'] == 15
        assert w['price_change_periods'] == [1, 5, 10]

    def test_adaptive_lags_short(self):
        """30 行：max_lag = 3 → lag 5 被排除"""
        processor = DataProcessor()
        lags = processor._compute_adaptive_lags(30)
        assert lags == [1, 2, 3]

    def test_adaptive_lags_long(self):
        """300 行：max_lag = 30 → 所有預設 lags 都在範圍內"""
        processor = DataProcessor()
        lags = processor._compute_adaptive_lags(300)
        assert lags == [1, 2, 3, 5]

    # --- create_technical_indicators adaptive ---

    def test_indicators_short_no_excessive_nan(self, short_data):
        """30 行：任何單一指標欄位不應有 >50% NaN（填充前）"""
        processor = DataProcessor()
        df = short_data.copy()
        windows = processor._compute_adaptive_windows(len(df))

        # Build indicators without bfill/ffill to check raw NaN count
        for w in windows['ma_windows']:
            df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()

        for col in df.columns:
            nan_ratio = df[col].isna().sum() / len(df)
            assert nan_ratio <= 0.5, f"{col} has {nan_ratio:.0%} NaN in 30-row data"

    def test_indicators_short_has_ema_macd(self, short_data):
        """30 行：EMA/MACD 不受資料長度影響，永遠產生"""
        processor = DataProcessor()
        result = processor.create_technical_indicators(short_data)
        for col in ['EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram']:
            assert col in result.columns

    def test_indicators_short_skips_large_windows(self, short_data):
        """30 行：MA_20 應被跳過"""
        processor = DataProcessor()
        result = processor.create_technical_indicators(short_data)
        assert 'MA_20' not in result.columns

    def test_indicators_long_backward_compatible(self, long_data):
        """300 行：所有原始指標都應存在"""
        processor = DataProcessor()
        result = processor.create_technical_indicators(long_data)
        expected = ['MA_5', 'MA_10', 'MA_20', 'EMA_12', 'EMA_26',
                    'MACD', 'RSI', 'BB_Middle', 'Volatility',
                    'Price_Change', 'Price_Change_5', 'Price_Change_10']
        for col in expected:
            assert col in result.columns, f"Missing {col} in 300-row data"

    # --- create_lagged_features adaptive ---

    def test_lagged_auto_short(self, short_data):
        """30 行 auto_lags=True：Close_lag_5 不應存在"""
        processor = DataProcessor()
        result = processor.create_lagged_features(short_data)
        assert 'Close_lag_1' in result.columns
        assert 'Close_lag_2' in result.columns
        assert 'Close_lag_3' in result.columns
        assert 'Close_lag_5' not in result.columns

    def test_lagged_auto_long(self, long_data):
        """300 行 auto_lags=True：所有預設 lags 都存在"""
        processor = DataProcessor()
        result = processor.create_lagged_features(long_data)
        for lag in [1, 2, 3, 5]:
            assert f'Close_lag_{lag}' in result.columns

    def test_lagged_manual_override(self, short_data):
        """手動指定 lags 時不觸發 adaptive 邏輯"""
        processor = DataProcessor()
        result = processor.create_lagged_features(short_data, lags=[1, 2, 3, 5])
        # All specified lags should be present, regardless of data length
        for lag in [1, 2, 3, 5]:
            assert f'Close_lag_{lag}' in result.columns

    def test_lagged_auto_false_legacy(self, short_data):
        """auto_lags=False 且 lags=None：使用舊預設 [1,2,3,5]"""
        processor = DataProcessor()
        result = processor.create_lagged_features(short_data, auto_lags=False)
        for lag in [1, 2, 3, 5]:
            assert f'Close_lag_{lag}' in result.columns


class TestCAPMFeatures:
    """Test create_capm_features() method."""

    @pytest.fixture
    def stock_data(self):
        """200-day stock OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        np.random.seed(42)
        price = 150 + np.cumsum(np.random.randn(200) * 1.0)
        return pd.DataFrame({
            'Open': price + np.random.randn(200) * 0.5,
            'High': price + abs(np.random.randn(200) * 1.0),
            'Low': price - abs(np.random.randn(200) * 1.0),
            'Close': price,
            'Volume': np.random.randint(1_000_000, 50_000_000, 200),
        }, index=dates)

    @pytest.fixture
    def market_data(self):
        """200-day market index data (e.g. S&P 500)."""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        np.random.seed(99)
        price = 4500 + np.cumsum(np.random.randn(200) * 5.0)
        return pd.DataFrame({
            'Open': price + np.random.randn(200) * 2.0,
            'High': price + abs(np.random.randn(200) * 5.0),
            'Low': price - abs(np.random.randn(200) * 5.0),
            'Close': price,
            'Volume': np.random.randint(1_000_000, 100_000_000, 200),
        }, index=dates)

    def test_capm_features_created(self, stock_data, market_data):
        """CAPM features should be added to the DataFrame."""
        processor = DataProcessor()
        result = processor.create_capm_features(stock_data, market_data, risk_free_rate=0.04)
        expected_cols = [
            'Daily_Return', 'Market_Return', 'Excess_Return',
            'Rolling_Beta', 'Rolling_Alpha', 'Sharpe_Ratio',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing CAPM column: {col}"

    def test_capm_no_nan(self, stock_data, market_data):
        """After bfill/ffill, no NaN should remain."""
        processor = DataProcessor()
        result = processor.create_capm_features(stock_data, market_data)
        assert result.isnull().sum().sum() == 0

    def test_capm_preserves_original_columns(self, stock_data, market_data):
        """Original OHLCV columns should be preserved."""
        processor = DataProcessor()
        result = processor.create_capm_features(stock_data, market_data)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert col in result.columns

    def test_capm_row_count_unchanged(self, stock_data, market_data):
        """Row count should stay the same (no rows dropped)."""
        processor = DataProcessor()
        result = processor.create_capm_features(stock_data, market_data)
        assert len(result) == len(stock_data)

    def test_capm_custom_rolling_window(self, stock_data, market_data):
        """Custom rolling window should be respected."""
        processor = DataProcessor()
        result = processor.create_capm_features(
            stock_data, market_data, rolling_window=50
        )
        assert 'Rolling_Beta' in result.columns
        assert result.isnull().sum().sum() == 0

    def test_capm_zero_risk_free_rate(self, stock_data, market_data):
        """Zero risk-free rate should work without errors."""
        processor = DataProcessor()
        result = processor.create_capm_features(
            stock_data, market_data, risk_free_rate=0.0
        )
        assert 'Sharpe_Ratio' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
