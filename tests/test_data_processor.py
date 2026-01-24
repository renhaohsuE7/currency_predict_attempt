"""
測試資料處理模組

測試 DataProcessor 類別的所有功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.currency_predictor.data_processor import DataProcessor


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
            target_column='Close',
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

        # 創建特徵
        features = sample_raw_data[['Open', 'High', 'Low', 'Close']]

        # 縮放
        scaled = processor.scale_features(features)

        # 檢查縮放後的資料
        assert isinstance(scaled, pd.DataFrame)
        assert scaled.shape == features.shape
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
