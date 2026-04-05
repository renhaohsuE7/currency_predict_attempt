"""
Stock E2E 驗證測試

用真實股票資料 (AAPL) 驗證完整 pipeline：
collect → CAPM features → train → predict → visualize

標記 @pytest.mark.e2e — 需要網路，CI 中預設跳過。
執行：docker compose run --rm test uv run pytest -m e2e tests/test_e2e_stock.py -v
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from currency_predictor.data.collectors import YahooFinanceCollector
from currency_predictor.data_processor import DataProcessor
from currency_predictor.models.patchtst.sklearn.model import PatchTSTSklearn
from currency_predictor.visualization.visualizer import CurrencyVisualizer
from currency_predictor.utils.asset_type import AssetType, classify_symbol


# ---------------------------------------------------------------------------
# Network check — skip entire module when offline
# ---------------------------------------------------------------------------

def _stock_data_available():
    """Check if yfinance can fetch AAPL data."""
    try:
        import yfinance as yf
        data = yf.Ticker("AAPL").history(period="5d")
        return data is not None and not data.empty
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _stock_data_available(), reason="yfinance network unavailable"),
]


# ---------------------------------------------------------------------------
# Module-scoped fixtures (expensive, run once per module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stock_data():
    """Collect 6mo AAPL data from yfinance."""
    collector = YahooFinanceCollector()
    df = collector.get_currency_data("AAPL", period="6mo", interval="1d")
    assert df is not None and not df.empty
    return df


@pytest.fixture(scope="module")
def market_data():
    """Collect 6mo S&P 500 data for CAPM calculations."""
    collector = YahooFinanceCollector()
    df = collector.get_currency_data("^GSPC", period="6mo", interval="1d")
    assert df is not None and not df.empty
    return df


@pytest.fixture(scope="module")
def forex_data():
    """Collect 6mo USDTWD data for comparison."""
    collector = YahooFinanceCollector()
    df = collector.get_currency_data("USDTWD=X", period="6mo", interval="1d")
    assert df is not None and not df.empty
    return df


@pytest.fixture
def fast_sklearn_params():
    """Small PatchTST sklearn params for fast tests."""
    return {
        "seq_len": 20,
        "pred_len": 5,
        "patch_len": 5,
        "stride": 5,
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
    }


# ---------------------------------------------------------------------------
# Phase 1: Stock Data Collection
# ---------------------------------------------------------------------------

class TestStockDataCollection:
    """Verify stock data from Yahoo Finance has expected structure."""

    def test_stock_has_ohlcv_columns(self, stock_data):
        """AAPL data should have Open, High, Low, Close, Volume."""
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in stock_data.columns, f"Missing column: {col}"

    def test_stock_has_volume(self, stock_data):
        """Stocks should have non-zero Volume (unlike forex)."""
        assert stock_data["Volume"].max() > 0

    def test_stock_has_enough_rows(self, stock_data):
        """6mo should give ~120+ trading days."""
        assert len(stock_data) >= 100

    def test_classify_stock_symbol(self):
        """AAPL should be classified as STOCK."""
        assert classify_symbol("AAPL") == AssetType.STOCK

    def test_classify_forex_symbol(self):
        """USDTWD=X should be classified as FOREX."""
        assert classify_symbol("USDTWD=X") == AssetType.FOREX


# ---------------------------------------------------------------------------
# Phase 2: CAPM Feature Engineering
# ---------------------------------------------------------------------------

class TestStockCAPMFeatures:
    """Verify CAPM features are correctly computed for stock data."""

    def test_capm_columns_created(self, stock_data, market_data):
        """CAPM features should be added to stock data."""
        processor = DataProcessor()
        cleaned = processor.clean_data(stock_data)
        with_indicators = processor.create_technical_indicators(cleaned)
        result = processor.create_capm_features(with_indicators, market_data, risk_free_rate=0.04)

        expected = ["Daily_Return", "Market_Return", "Excess_Return",
                     "Rolling_Beta", "Rolling_Alpha", "Sharpe_Ratio"]
        for col in expected:
            assert col in result.columns, f"Missing CAPM column: {col}"

    def test_capm_no_nan_after_fill(self, stock_data, market_data):
        """No NaN should remain after bfill/ffill."""
        processor = DataProcessor()
        cleaned = processor.clean_data(stock_data)
        with_indicators = processor.create_technical_indicators(cleaned)
        result = processor.create_capm_features(with_indicators, market_data)
        assert result.isnull().sum().sum() == 0

    def test_forex_no_capm_features(self, forex_data, market_data):
        """Forex data processed without CAPM should NOT have CAPM columns."""
        processor = DataProcessor()
        cleaned = processor.clean_data(forex_data)
        result = processor.create_technical_indicators(cleaned)
        # Do NOT call create_capm_features — this simulates forex path
        capm_cols = ["Rolling_Beta", "Rolling_Alpha", "Sharpe_Ratio"]
        for col in capm_cols:
            assert col not in result.columns


# ---------------------------------------------------------------------------
# Phase 3: Train + Predict with CAPM
# ---------------------------------------------------------------------------

class TestStockTrainPredict:
    """Full train → predict cycle with CAPM-enhanced features."""

    def test_fit_predict_with_capm(self, stock_data, market_data, fast_sklearn_params):
        """Train PatchTSTSklearn on stock data with CAPM features, then predict."""
        processor = DataProcessor()
        cleaned = processor.clean_data(stock_data)
        with_indicators = processor.create_technical_indicators(cleaned)
        with_capm = processor.create_capm_features(with_indicators, market_data, risk_free_rate=0.04)
        processed = processor.create_lagged_features(with_capm, lags=[1, 2, 3])

        # Split features/target
        feature_cols = [c for c in processed.columns if c != "Close"]
        X = processed[feature_cols]
        y = processed["Close"]
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Train
        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X_train, y_train)
        assert model.is_fitted

        # Predict
        preds = model.predict(X_test, horizon=fast_sklearn_params["pred_len"])
        assert preds is not None
        assert len(preds) == fast_sklearn_params["pred_len"]
        assert not np.any(np.isnan(preds))

    def test_fit_predict_without_capm(self, stock_data, fast_sklearn_params):
        """Stock data without CAPM should also train successfully."""
        processor = DataProcessor()
        cleaned = processor.clean_data(stock_data)
        with_indicators = processor.create_technical_indicators(cleaned)
        processed = processor.create_lagged_features(with_indicators, lags=[1, 2, 3])

        feature_cols = [c for c in processed.columns if c != "Close"]
        X = processed[feature_cols]
        y = processed["Close"]
        split = int(len(X) * 0.8)

        model = PatchTSTSklearn(**fast_sklearn_params)
        model.fit(X.iloc[:split], y.iloc[:split])
        preds = model.predict(X.iloc[split:], horizon=fast_sklearn_params["pred_len"])
        assert len(preds) == fast_sklearn_params["pred_len"]


# ---------------------------------------------------------------------------
# Phase 4: Visualization
# ---------------------------------------------------------------------------

class TestStockVisualization:
    """Verify dashboard renders correctly with stock data (real Volume)."""

    def test_dashboard_shows_volume_for_stock(self, stock_data, tmp_path):
        """Stock data with Volume > 0 should display volume chart."""
        vis = CurrencyVisualizer(output_dir=str(tmp_path / "figures"))
        fig = vis.create_dashboard(df=stock_data, symbol="AAPL")
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 5
        plt.close(fig)

    def test_dashboard_forex_uses_fallback(self, forex_data, tmp_path):
        """Forex data with Volume=0 should use daily range fallback."""
        vis = CurrencyVisualizer(output_dir=str(tmp_path / "figures"))
        fig = vis.create_dashboard(df=forex_data, symbol="USDTWD=X")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
