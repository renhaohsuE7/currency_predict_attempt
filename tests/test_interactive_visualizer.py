"""
Tests for interactive visualization module (Plotly).

Covers ThemeManager and InteractiveVisualizer.
"""

import pytest
import pandas as pd
import numpy as np

from currency_predictor.visualization.themes import ThemeManager

# InteractiveVisualizer requires plotly — skip all tests if unavailable
plotly = pytest.importorskip("plotly")
from currency_predictor.visualization.interactive import InteractiveVisualizer  # noqa: E402


@pytest.fixture
def sample_ohlcv():
    """30-day OHLCV data for chart tests."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    np.random.seed(42)
    base = 30.0
    close = base + np.random.randn(30).cumsum() * 0.1
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(30) * 0.02,
            "High": close + np.abs(np.random.randn(30) * 0.05),
            "Low": close - np.abs(np.random.randn(30) * 0.05),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, 30),
        },
        index=dates,
    )


@pytest.fixture
def sample_ohlc_no_volume():
    """OHLC data without Volume column."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    np.random.seed(42)
    close = 30.0 + np.random.randn(30).cumsum() * 0.1
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(30) * 0.02,
            "High": close + np.abs(np.random.randn(30) * 0.05),
            "Low": close - np.abs(np.random.randn(30) * 0.05),
            "Close": close,
        },
        index=dates,
    )


@pytest.fixture
def visualizer(tmp_path):
    return InteractiveVisualizer(theme="light", output_dir=str(tmp_path / "figures"))


# ------------------------------------------------------------------
# ThemeManager
# ------------------------------------------------------------------


class TestThemeManager:
    def test_default_theme(self):
        tm = ThemeManager()
        assert tm.name == "light"

    def test_dark_theme(self):
        tm = ThemeManager("dark")
        assert tm.name == "dark"
        assert tm.template == "plotly_dark"

    def test_invalid_theme(self):
        with pytest.raises(ValueError, match="Unknown theme"):
            ThemeManager("neon")

    def test_set_theme(self):
        tm = ThemeManager("light")
        tm.set_theme("dark")
        assert tm.name == "dark"

    def test_line_color_cycles(self):
        tm = ThemeManager("light")
        # Index beyond list length should cycle
        c0 = tm.line_color(0)
        c_wrap = tm.line_color(8)  # 8 colors in palette → wraps to 0
        assert c0 == c_wrap

    def test_layout_defaults(self):
        tm = ThemeManager("dark")
        defaults = tm.layout_defaults()
        assert defaults["template"] == "plotly_dark"
        assert "paper_bgcolor" in defaults
        assert "font" in defaults

    def test_available_themes(self):
        themes = ThemeManager.available_themes()
        assert "light" in themes
        assert "dark" in themes

    def test_properties(self):
        tm = ThemeManager("light")
        assert isinstance(tm.bg_color, str)
        assert isinstance(tm.text_color, str)
        assert isinstance(tm.grid_color, str)
        assert isinstance(tm.up_color, str)
        assert isinstance(tm.down_color, str)


# ------------------------------------------------------------------
# InteractiveVisualizer — initialization
# ------------------------------------------------------------------


class TestInteractiveVisualizerInit:
    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "sub" / "figures"
        viz = InteractiveVisualizer(output_dir=str(out))
        assert out.exists()

    def test_set_theme(self, visualizer):
        visualizer.set_theme("dark")
        assert visualizer.theme.name == "dark"


# ------------------------------------------------------------------
# Technical Analysis Chart
# ------------------------------------------------------------------


class TestPlotTechnicalAnalysis:
    def test_returns_figure(self, visualizer, sample_ohlcv):
        fig = visualizer.plot_technical_analysis(sample_ohlcv, "USDTWD")
        assert fig is not None
        assert hasattr(fig, "data")  # plotly Figure

    def test_without_volume(self, visualizer, sample_ohlc_no_volume):
        fig = visualizer.plot_technical_analysis(sample_ohlc_no_volume, "TEST")
        assert fig is not None

    def test_volume_all_zeros(self, visualizer, sample_ohlcv):
        """Volume column exists but all zeros (forex case) — should skip volume subplot."""
        df = sample_ohlcv.copy()
        df["Volume"] = 0
        fig = visualizer.plot_technical_analysis(df, "USDTWD")
        assert fig is not None
        # Should have 2 rows (no volume), not 3
        volume_traces = [t for t in fig.data if t.name == "Volume"]
        assert len(volume_traces) == 0

    def test_save_html(self, visualizer, sample_ohlcv, tmp_path):
        fig = visualizer.plot_technical_analysis(
            sample_ohlcv, "USDTWD", save_path="ta_chart.html"
        )
        assert (tmp_path / "figures" / "ta_chart.html").exists()

    def test_save_adds_suffix(self, visualizer, sample_ohlcv, tmp_path):
        visualizer.plot_technical_analysis(
            sample_ohlcv, "USDTWD", save_path="ta_chart"
        )
        assert (tmp_path / "figures" / "ta_chart.html").exists()

    def test_dark_theme(self, tmp_path, sample_ohlcv):
        viz = InteractiveVisualizer(theme="dark", output_dir=str(tmp_path))
        fig = viz.plot_technical_analysis(sample_ohlcv, "USDTWD")
        assert fig.layout.template.layout.paper_bgcolor is not None


# ------------------------------------------------------------------
# Prediction Comparison Chart
# ------------------------------------------------------------------


class TestPlotPredictionComparison:
    def test_single_model(self, visualizer, sample_ohlcv):
        actual = sample_ohlcv["Close"][-10:]
        preds = {"ModelA": actual.values + np.random.randn(10) * 0.01}
        fig = visualizer.plot_prediction_comparison(actual, preds, "USDTWD")
        assert fig is not None
        # 1 actual trace + 1 prediction trace
        assert len(fig.data) == 2

    def test_multi_model(self, visualizer, sample_ohlcv):
        actual = sample_ohlcv["Close"][-10:]
        preds = {
            "ModelA": actual.values + 0.01,
            "ModelB": actual.values - 0.01,
        }
        fig = visualizer.plot_prediction_comparison(actual, preds, "USDTWD")
        assert len(fig.data) == 3  # actual + 2 models

    def test_with_metrics(self, visualizer, sample_ohlcv):
        actual = sample_ohlcv["Close"][-10:]
        preds = {"ModelA": actual.values + 0.01}
        metrics = {"ModelA": {"rmse": 0.01, "mae": 0.008, "mase": 0.95, "mda": 0.65}}
        fig = visualizer.plot_prediction_comparison(
            actual, preds, "USDTWD", metrics=metrics
        )
        # Label should include RMSE and MASE/MDA
        assert "RMSE" in fig.data[1].name
        assert "MASE" in fig.data[1].name
        assert "MDA" in fig.data[1].name

    def test_save_html(self, visualizer, sample_ohlcv, tmp_path):
        actual = sample_ohlcv["Close"][-5:]
        preds = {"M": actual.values}
        visualizer.plot_prediction_comparison(
            actual, preds, "T", save_path="pred.html"
        )
        assert (tmp_path / "figures" / "pred.html").exists()
