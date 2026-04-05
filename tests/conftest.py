"""
全域測試 fixtures

提供所有測試共用的 fixtures 和設定。
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib

# Force non-interactive backend for chart tests (must be before any pyplot import)
matplotlib.use("Agg")


@pytest.fixture
def sample_ohlcv_df():
    """Minimal OHLCV DataFrame for quick unit tests (~30 rows)."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    close = np.random.randn(n).cumsum() + 30.0
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.1,
            "High": close + abs(np.random.randn(n) * 0.2),
            "Low": close - abs(np.random.randn(n) * 0.2),
            "Close": close,
            "Volume": np.random.randint(1000, 50000, n),
        },
        index=dates,
    )


@pytest.fixture
def e2e_fixture_path():
    """Path to the committed CSV fixture for offline E2E tests."""
    return Path(__file__).parent / "fixtures" / "USDTWD_1y_sample.csv"


@pytest.fixture
def e2e_output_dir(tmp_path):
    """Temporary output directory tree for E2E test artifacts."""
    for sub in ["data/raw", "models", "results", "results/figures"]:
        (tmp_path / sub).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def processed_fixture_data(e2e_fixture_path):
    """Load fixture CSV and run full DataProcessor pipeline.

    Returns processed DataFrame with technical indicators and lagged features.
    """
    from currency_predictor.data_processor import DataProcessor

    df = pd.read_csv(e2e_fixture_path, index_col="Date", parse_dates=True)
    processor = DataProcessor()
    cleaned = processor.clean_data(df)
    with_indicators = processor.create_technical_indicators(cleaned)
    return processor.create_lagged_features(with_indicators, lags=[1, 2, 3])


@pytest.fixture
def fast_sklearn_params():
    """Small PatchTST sklearn params for fast E2E tests."""
    return {
        "seq_len": 50,
        "pred_len": 5,
        "patch_len": 10,
        "stride": 5,
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
    }
