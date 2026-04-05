"""
線上端到端冒煙測試 (Layer 2)

實際呼叫 yfinance，驗證完整 collect → process → train → predict → visualize → report 流程。

標記 @pytest.mark.e2e — 需要網路，CI 中預設跳過。
執行：uv run pytest -m e2e -v

注意：無網路時整個模組會自動 skip。
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from currency_predictor.data.collectors import YahooFinanceCollector
from currency_predictor.data.storage import DataStorage
from currency_predictor.data_processor import DataProcessor
from currency_predictor.models.patchtst.sklearn.model import PatchTSTSklearn
from currency_predictor.visualization.visualizer import CurrencyVisualizer
from currency_predictor.reporting.formatter import ResultFormatter


# ---------------------------------------------------------------------------
# Network check — skip entire module when offline
# ---------------------------------------------------------------------------

def _network_available():
    """Check if yfinance can reach Yahoo Finance."""
    try:
        import yfinance as yf

        data = yf.Ticker("USDTWD=X").history(period="5d")
        return data is not None and not data.empty
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _network_available(), reason="yfinance network unavailable"),
]


# ---------------------------------------------------------------------------
# Module-scoped fixtures (expensive, run once per module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def collected_data():
    """Collect real USDTWD data from yfinance (6 months).

    6mo gives ~120 trading days, enough for seq_len=50 + pred_len=5 after
    80/20 split (~96 train rows > 55 minimum).
    """
    collector = YahooFinanceCollector()
    df = collector.get_currency_data("USDTWD=X", period="6mo", interval="1d")
    assert df is not None and not df.empty
    return df


@pytest.fixture(scope="module")
def processed_data(collected_data):
    """Full DataProcessor pipeline on real data."""
    processor = DataProcessor()
    cleaned = processor.clean_data(collected_data)
    with_indicators = processor.create_technical_indicators(cleaned)
    return processor.create_lagged_features(with_indicators, lags=[1, 2, 3])


@pytest.fixture(scope="module")
def trained_model_and_data(processed_data):
    """Train a fast sklearn model on real data (once per module).

    Returns (model, full, y) where full includes Close so predict() sees the
    same dimension as during fit() (which concats X + y). The model's
    predict() uses only the last seq_len rows, so passing the full DataFrame
    is sufficient.
    """
    feature_cols = [c for c in processed_data.columns if c != "Close"]
    X = processed_data[feature_cols]
    y = processed_data["Close"]
    full = pd.concat([X, y.to_frame()], axis=1)
    split = int(len(X) * 0.8)

    model = PatchTSTSklearn(
        seq_len=50,
        pred_len=5,
        patch_len=10,
        stride=5,
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model.fit(X.iloc[:split], y.iloc[:split])
    return model, full, y


# ---------------------------------------------------------------------------
# Tests: Data Collection
# ---------------------------------------------------------------------------

class TestOnlineE2ECollect:
    """Test real data collection and storage."""

    def test_yfinance_returns_ohlcv(self, collected_data):
        """Real yfinance call returns DataFrame with expected columns and rows."""
        assert len(collected_data) >= 30  # 3mo ≈ 60 trading days
        for col in ["Open", "High", "Low", "Close"]:
            assert col in collected_data.columns

    def test_data_storage_round_trip(self, collected_data, tmp_path):
        """Real data can be saved and loaded via DataStorage."""
        storage = DataStorage(base_dir=str(tmp_path / "data"))
        success = storage.save_raw_data(collected_data, "USDTWD", "6mo")
        assert success is True

        loaded = storage.load_raw_data("USDTWD", "6mo")
        assert loaded is not None
        assert len(loaded) == len(collected_data)


# ---------------------------------------------------------------------------
# Tests: Full Pipeline
# ---------------------------------------------------------------------------

class TestOnlineE2EFullPipeline:
    """Test full collect → process → train → predict → visualize → report."""

    def test_train_predict_on_real_data(self, trained_model_and_data):
        """Model trains and predicts on real yfinance data."""
        model, full, y = trained_model_and_data
        assert model.is_fitted is True

        preds = model.predict(full)
        assert len(preds) == 5
        assert np.all(np.isfinite(preds))

    def test_full_chain_chart_and_report(self, trained_model_and_data, tmp_path):
        """Full chain: predict → chart → report all succeed on real data."""
        import matplotlib.pyplot as plt

        model, full, y = trained_model_and_data
        preds = model.predict(full)

        # Build Series for visualizer
        pred_dates = y.index[-len(preds):]
        predicted_series = pd.Series(preds, index=pred_dates)
        actual_series = y.iloc[-len(preds):]

        # Chart
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        chart_path = str(figures_dir / "online_e2e_chart.png")

        viz = CurrencyVisualizer(output_dir=str(figures_dir))
        fig = viz.plot_prediction_results(
            actual=actual_series,
            predicted=predicted_series,
            symbol="USDTWD",
            save_path=chart_path,
        )
        plt.close(fig)
        assert Path(chart_path).exists()

        # Report
        formatter = ResultFormatter(use_logger=False)
        report = formatter.generate_report(
            {
                "success": True,
                "pipeline_status": {
                    "data_collection": True,
                    "model_training": True,
                    "prediction": True,
                    "results_saved": True,
                },
                "predictions": [
                    {
                        "symbol": "USDTWD=X",
                        "predictions": preds.tolist(),
                        "last_known_value": float(y.iloc[-1]),
                    }
                ],
            }
        )
        assert "USDTWD" in report
