"""
Interactive Visualization Module (Plotly)

Provides interactive charts with technical indicators, prediction overlays,
and HTML export. Requires ``plotly`` (install via ``uv sync --extra interactive``).
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .themes import ThemeManager

logger = logging.getLogger(__name__)


def _require_plotly() -> None:
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install with: uv sync --extra interactive"
        )


class InteractiveVisualizer:
    """Interactive chart builder backed by Plotly."""

    def __init__(
        self,
        theme: str = "light",
        output_dir: str = "results/figures",
    ):
        _require_plotly()
        self.theme = ThemeManager(theme)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_theme(self, theme: str) -> None:
        self.theme.set_theme(theme)

    # ------------------------------------------------------------------
    # Technical Analysis Chart
    # ------------------------------------------------------------------

    def plot_technical_analysis(
        self,
        df: pd.DataFrame,
        symbol: str,
        save_path: Optional[str] = None,
    ) -> "go.Figure":
        """
        Multi-subplot interactive chart: Candlestick + Volume + RSI + MACD + BB.

        Args:
            df: OHLCV DataFrame (needs Open/High/Low/Close; Volume optional)
            symbol: Symbol name for title
            save_path: If set, export HTML to this path (relative to output_dir)

        Returns:
            plotly Figure
        """
        has_volume = "Volume" in df.columns and df["Volume"].max() > 0
        row_count = 3 if has_volume else 2
        row_heights = [0.5, 0.2, 0.3] if has_volume else [0.6, 0.4]
        subplot_titles = ["Price & Bollinger Bands", "Volume", "RSI / MACD"] if has_volume else ["Price & Bollinger Bands", "RSI / MACD"]

        fig = make_subplots(
            rows=row_count, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )

        idx = df.index

        # --- Row 1: Candlestick + Bollinger Bands ---
        fig.add_trace(go.Candlestick(
            x=idx, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color=self.theme.up_color,
            decreasing_line_color=self.theme.down_color,
            name="OHLC",
        ), row=1, col=1)

        # Bollinger Bands
        bb_window = min(15, len(df) // 4) if len(df) > 20 else 5
        if bb_window >= 3:
            sma = df["Close"].rolling(bb_window).mean()
            std = df["Close"].rolling(bb_window).std()
            fig.add_trace(go.Scatter(
                x=idx, y=sma, mode="lines", name=f"SMA({bb_window})",
                line=dict(color=self.theme.line_color(0), width=1),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=idx, y=(sma + 2 * std), mode="lines", name="BB Upper",
                line=dict(color=self.theme.line_color(1), width=1, dash="dot"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=idx, y=(sma - 2 * std), mode="lines", name="BB Lower",
                line=dict(color=self.theme.line_color(1), width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(100,100,200,0.1)",
            ), row=1, col=1)

        # --- Row 2: Volume (if available) ---
        vol_row = 2 if has_volume else None
        indicator_row = 3 if has_volume else 2

        if has_volume:
            colors = [
                self.theme.up_color if c >= o else self.theme.down_color
                for c, o in zip(df["Close"], df["Open"])
            ]
            fig.add_trace(go.Bar(
                x=idx, y=df["Volume"], marker_color=colors,
                name="Volume", opacity=0.6,
            ), row=vol_row, col=1)

        # --- Last Row: RSI + MACD ---
        # RSI
        rsi_window = min(10, len(df) // 4) if len(df) > 12 else 3
        if rsi_window >= 3:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(rsi_window).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            fig.add_trace(go.Scatter(
                x=idx, y=rsi, mode="lines", name=f"RSI({rsi_window})",
                line=dict(color=self.theme.line_color(2), width=1.5),
            ), row=indicator_row, col=1)
            # Overbought / oversold
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          opacity=0.4, row=indicator_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          opacity=0.4, row=indicator_row, col=1)

        # MACD
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        fig.add_trace(go.Scatter(
            x=idx, y=macd, mode="lines", name="MACD",
            line=dict(color=self.theme.line_color(3), width=1),
            yaxis="y" + str(indicator_row * 2),  # secondary y on same row
            visible="legendonly",
        ), row=indicator_row, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=signal, mode="lines", name="Signal",
            line=dict(color=self.theme.line_color(4), width=1, dash="dot"),
            visible="legendonly",
        ), row=indicator_row, col=1)

        # Layout
        layout_kw = self.theme.layout_defaults()
        layout_kw.update(
            title=f"{symbol} Technical Analysis",
            height=200 * row_count + 200,
            showlegend=True,
            xaxis_rangeslider_visible=False,
        )
        fig.update_layout(**layout_kw)

        if save_path:
            self._save_html(fig, save_path)

        return fig

    # ------------------------------------------------------------------
    # Prediction Comparison Chart
    # ------------------------------------------------------------------

    def plot_prediction_comparison(
        self,
        actual: pd.Series,
        model_predictions: Dict[str, np.ndarray],
        symbol: str,
        metrics: Optional[Dict[str, Dict[str, float]]] = None,
        save_path: Optional[str] = None,
    ) -> "go.Figure":
        """
        Interactive multi-model prediction comparison.

        Args:
            actual: Actual values with DatetimeIndex
            model_predictions: {model_name: prediction_array}
            symbol: Symbol name
            metrics: {model_name: {rmse, mae, ...}} for annotation
            save_path: HTML export path

        Returns:
            plotly Figure
        """
        fig = go.Figure()

        # Actual
        fig.add_trace(go.Scatter(
            x=actual.index, y=actual.values,
            mode="lines", name="Actual",
            line=dict(color=self.theme.text_color, width=2),
        ))

        # Model predictions
        pred_start = actual.index[-1]
        for i, (name, preds) in enumerate(model_predictions.items()):
            pred_idx = pd.date_range(
                start=pred_start, periods=len(preds) + 1, freq="D",
            )[1:]
            label = name
            if metrics and name in metrics:
                rmse = metrics[name].get("rmse", 0)
                label = f"{name} (RMSE={rmse:.4f})"

            fig.add_trace(go.Scatter(
                x=pred_idx, y=preds,
                mode="lines+markers", name=label,
                line=dict(color=self.theme.line_color(i), width=2, dash="dash"),
                marker=dict(size=5),
            ))

        layout_kw = self.theme.layout_defaults()
        layout_kw.update(
            title=f"{symbol} — Multi-Model Prediction Comparison",
            yaxis_title="Price",
            height=500,
            hovermode="x unified",
        )
        fig.update_layout(**layout_kw)

        if save_path:
            self._save_html(fig, save_path)

        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_html(self, fig: "go.Figure", filename: str) -> Path:
        path = self.output_dir / filename
        if not str(path).endswith(".html"):
            path = path.with_suffix(".html")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn")
        logger.info(f"Interactive chart saved to {path}")
        return path
