"""Backtesting result dataclasses."""

from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    """Result of a single backtest fold."""

    fold_index: int
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    train_size: int
    test_size: int

    # Prediction metrics (from compute_unified_metrics)
    prediction_metrics: Dict[str, float]

    # Financial metrics
    financial_metrics: Dict[str, float]

    # Raw data for visualization
    actual_prices: np.ndarray
    predicted_prices: np.ndarray
    equity_curve: np.ndarray
    training_time: float  # seconds

    # Rolling evaluation (optional, backwards compatible)
    per_horizon_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    n_origins: int = 0


@dataclass
class BacktestResult:
    """Aggregate result of a complete backtest run."""

    symbol: str
    model_name: str
    strategy: str  # "rolling" or "expanding"
    n_folds: int

    # Per-fold results
    fold_results: List[FoldResult]

    # Aggregate metrics (mean across folds)
    avg_prediction_metrics: Dict[str, float]
    avg_financial_metrics: Dict[str, float]
    std_prediction_metrics: Dict[str, float]

    # Overall equity curve (concatenated test periods)
    overall_equity_curve: np.ndarray
    overall_actual_prices: np.ndarray
    overall_predicted_prices: np.ndarray
    overall_dates: List[pd.Timestamp]

    # Metadata
    total_training_time: float
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Rolling evaluation (optional, backwards compatible)
    avg_per_horizon_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
