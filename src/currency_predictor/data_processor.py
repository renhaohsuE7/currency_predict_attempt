"""Data Processing Module

This module provides functionality to clean and engineer features from
currency exchange rate data for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and prepares currency data for machine learning."""

    def __init__(self):
        self.scaler = None
        self.feature_columns = []

    @staticmethod
    def to_log_returns(close: pd.Series) -> pd.Series:
        """Convert a price series to log returns: r_t = ln(P_t) - ln(P_{t-1}).

        Args:
            close: Price series (must be positive).

        Returns:
            Log-return series, same index; the first element is NaN.
        """
        return np.log(close.astype(float)).diff()

    @staticmethod
    def align_feature_columns(
        frames: List[pd.DataFrame],
        explicit: Optional[List[str]] = None,
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """Align multiple DataFrames to a common feature-column set.

        Needed for panel training: per-symbol feature engineering can yield
        slightly different columns (adaptive indicators), but pooled sequences
        require an identical schema.

        Args:
            frames: Per-symbol feature DataFrames.
            explicit: Explicit column list to enforce; when None, use the
                intersection of all frames' columns (ordered by the first frame).

        Returns:
            Tuple of (frames reindexed to the common columns, common column list).
        """
        if not frames:
            return [], []

        if explicit is not None:
            common = list(explicit)
        else:
            common_set = set(frames[0].columns)
            for f in frames[1:]:
                common_set &= set(f.columns)
            # Preserve the first frame's column order
            common = [c for c in frames[0].columns if c in common_set]

        aligned = [f[common] for f in frames]
        return aligned, common

    @staticmethod
    def from_log_returns(last_price: float, log_returns: np.ndarray) -> np.ndarray:
        """Reconstruct a forward price path from log returns.

        P_i = last_price * exp(cumsum(log_returns)_i).

        Args:
            last_price: Last known price before the forecast horizon.
            log_returns: Predicted log returns for each future step.

        Returns:
            Reconstructed price levels (same length as ``log_returns``).
        """
        returns = np.asarray(log_returns, dtype=float)
        prices: np.ndarray = float(last_price) * np.exp(np.cumsum(returns))
        return prices

    def _compute_adaptive_windows(self, data_length: int) -> dict:
        """Compute adaptive rolling window sizes based on data length.

        Each rolling indicator will produce at most ~15% NaN before filling.
        """
        max_rolling = max(3, int(data_length * 0.15))
        return {
            "ma_windows": [w for w in [5, 10, 20] if w <= max_rolling],
            "rsi_window": min(10, max_rolling),
            "bb_window": min(15, max_rolling),
            "volatility_window": min(15, max_rolling),
            "price_change_periods": [p for p in [1, 5, 10] if p <= max_rolling],
            "volume_window": min(15, max_rolling),
        }

    def _compute_adaptive_lags(self, data_length: int) -> list:
        """Compute adaptive lag periods based on data length.

        Each lag will produce at most ~10% NaN before filling.
        """
        max_lag = max(1, int(data_length * 0.1))
        return [lag for lag in [1, 2, 3, 5] if lag <= max_lag]

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw currency data.

        Args:
            data: Raw currency data DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original data
        df = data.copy()

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df = df.dropna()

        # Ensure datetime index and handle timezone
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone information if present to avoid conversion issues
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Sort by date
        df = df.sort_index()

        logger.info(f"Data cleaned: {len(df)} records remaining")
        return df

    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators for currency prediction.

        Window sizes are dynamically adjusted based on data length to avoid
        excessive NaN values in short datasets.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with additional technical indicators
        """
        df = data.copy()
        windows = self._compute_adaptive_windows(len(df))
        logger.debug(f"Adaptive windows for {len(df)} rows: {windows}")

        # Moving averages (adaptive — skip windows larger than threshold)
        for w in windows["ma_windows"]:
            df[f"MA_{w}"] = df["Close"].rolling(window=w).mean()

        # Exponential moving averages (always safe — ewm produces no NaN)
        df["EMA_12"] = df["Close"].ewm(span=12).mean()
        df["EMA_26"] = df["Close"].ewm(span=26).mean()

        # MACD (derived from EMA, always safe)
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

        # RSI (adaptive window)
        rsi_w = windows["rsi_window"]
        if rsi_w >= 3:
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_w).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (adaptive window, need >=5 for meaningful std)
        bb_w = windows["bb_window"]
        if bb_w >= 5:
            df["BB_Middle"] = df["Close"].rolling(window=bb_w).mean()
            bb_std = df["Close"].rolling(window=bb_w).std()
            df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
            df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
            df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
            df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / df["BB_Width"]

        # Volatility (adaptive window)
        vol_w = windows["volatility_window"]
        if vol_w >= 3:
            df["Volatility"] = df["Close"].rolling(window=vol_w).std()

        # Price changes (adaptive periods)
        for p in windows["price_change_periods"]:
            if p == 1:
                df["Price_Change"] = df["Close"].pct_change()
            else:
                df[f"Price_Change_{p}"] = df["Close"].pct_change(periods=p)

        # Volume indicators (adaptive window)
        vol_ma_w = windows["volume_window"]
        if (
            "Volume" in df.columns
            and vol_ma_w >= 3
            and df["Volume"].notna().sum() > vol_ma_w
        ):
            df["Volume_MA"] = df["Volume"].rolling(window=vol_ma_w).mean()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]

        # Fill NaN values created by rolling windows
        df = df.bfill()
        df = df.ffill()

        logger.info(f"Technical indicators created: {len(df.columns)} total features")
        return df

    def create_lagged_features(
        self, data: pd.DataFrame, lags: Optional[list] = None, auto_lags: bool = True
    ) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.

        Args:
            data: DataFrame with features
            lags: Explicit list of lag periods (overrides auto_lags if provided)
            auto_lags: If True and lags is None, compute adaptive lags based on
                data length. If False and lags is None, use legacy default [1,2,3,5].

        Returns:
            DataFrame with lagged features
        """
        df = data.copy()

        # Determine effective lags
        if lags is not None:
            effective_lags = lags
        elif auto_lags:
            effective_lags = self._compute_adaptive_lags(len(df))
        else:
            effective_lags = [1, 2, 3, 5]
        logger.debug(f"Lagged features: using lags={effective_lags} for {len(df)} rows")

        # Create lagged features for Close price
        for lag in effective_lags:
            df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

            # Only create Volume lag if Volume column exists
            if "Volume" in df.columns:
                df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)

            # Only create Price_Change lag if Price_Change exists
            if "Price_Change" in df.columns:
                df[f"Price_Change_lag_{lag}"] = df["Price_Change"].shift(lag)

        # 檢查NaN值
        nan_count_before = df.isnull().sum().sum()
        records_before = len(df)

        # Drop columns that are entirely NaN (these can't be filled)
        df = df.dropna(axis=1, how="all")

        # Fill NaN values instead of dropping rows
        # Use forward fill first
        df = df.ffill()

        # Then backward fill any remaining NaN at the start
        df = df.bfill()

        # Only drop rows if there are still NaN after filling
        # (this should rarely happen now)
        rows_with_nan = df.isnull().any(axis=1).sum()
        if rows_with_nan > 0:
            logger.warning(
                f"Still have {rows_with_nan} rows with NaN after filling, dropping them"
            )
            df = df.dropna()

        nan_count_after = df.isnull().sum().sum()
        logger.info(
            f"Lagged features created: {len(df)} records (started with {records_before}, had {nan_count_before} NaN, filled to {nan_count_after})"
        )
        return df

    def prepare_features_target(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        prediction_horizon: int = 1,
        feature_columns: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for machine learning.

        Args:
            data: Processed DataFrame
            target_column: Column to predict
            prediction_horizon: How many periods ahead to predict
            feature_columns: Specific columns to use as features

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = data.copy()

        # Create target variable (future price)
        df["target"] = df[target_column].shift(-prediction_horizon)

        # Remove rows where target is NaN
        df = df.dropna()

        # Select feature columns
        if feature_columns is None:
            # Use all numeric columns except target and OHLCV
            exclude_cols = ["target", "Open", "High", "Low", "Close", "Volume"]
            feature_columns = [
                col
                for col in df.columns
                if col not in exclude_cols and df[col].dtype in ["float64", "int64"]
            ]

        self.feature_columns = feature_columns

        X = df[feature_columns]
        y = df["target"]

        logger.info(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        return X, y

    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type: str = "standard"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler or MinMaxScaler.

        Args:
            X_train: Training features
            X_test: Testing features
            scaler_type: 'standard' or 'minmax'

        Returns:
            Tuple of (scaled train features, scaled test features)
        """
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")

        # Fit scaler on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        logger.info(f"Features scaled using {scaler_type} scaler")
        return X_train_scaled, X_test_scaled

    def train_test_split(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test sets chronologically.

        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def create_capm_features(
        self,
        data: pd.DataFrame,
        market_data: pd.DataFrame,
        risk_free_rate: float = 0.0,
        rolling_window: int = 252,
    ) -> pd.DataFrame:
        """Add CAPM-related features for stock analysis.

        Only intended for stock symbols. Forex/crypto should skip this.

        Features added:
        - Daily_Return: daily percentage change of Close
        - Market_Return: market index daily return (aligned by date)
        - Excess_Return: Daily_Return minus daily risk-free rate
        - Rolling_Beta: rolling covariance(asset, market) / variance(market)
        - Rolling_Alpha: Jensen's Alpha (excess return − beta × market excess)
        - Sharpe_Ratio: rolling mean(excess return) / std(daily return)

        Args:
            data: Asset OHLCV DataFrame (datetime index)
            market_data: Market index OHLCV DataFrame (datetime index)
            risk_free_rate: Annualized risk-free rate (e.g. 0.04 for 4%)
            rolling_window: Window size in trading days

        Returns:
            DataFrame with additional CAPM feature columns
        """
        df = data.copy()

        # Daily returns
        df["Daily_Return"] = df["Close"].pct_change()

        # Market returns (aligned by date)
        # Strip timezone from market_data index to match cleaned asset data
        market_close = market_data["Close"].copy()
        if hasattr(market_close.index, "tz") and market_close.index.tz is not None:
            market_close.index = market_close.index.tz_localize(None)
        market_returns = market_close.pct_change()
        df["Market_Return"] = market_returns.reindex(df.index).ffill()

        # Daily risk-free rate (continuous compounding approximation)
        rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1

        # Excess returns
        df["Excess_Return"] = df["Daily_Return"] - rf_daily
        market_excess = df["Market_Return"] - rf_daily

        # Adaptive rolling window (same idea as _compute_adaptive_windows)
        effective_window = min(rolling_window, max(3, int(len(df) * 0.5)))

        # Rolling Beta = Cov(Ri, Rm) / Var(Rm)
        # Guard against zero market variance (flat/holiday-filled windows) → NaN, not inf.
        cov = df["Daily_Return"].rolling(effective_window).cov(df["Market_Return"])
        var = df["Market_Return"].rolling(effective_window).var()
        df["Rolling_Beta"] = cov / var.replace(0.0, np.nan)

        # Rolling Alpha (Jensen's Alpha)
        df["Rolling_Alpha"] = (
            df["Excess_Return"].rolling(effective_window).mean()
            - df["Rolling_Beta"] * market_excess.rolling(effective_window).mean()
        )

        # Sharpe Ratio = mean(excess return) / std(daily return)
        # Guard against zero return volatility → NaN, not inf.
        rolling_mean = df["Excess_Return"].rolling(effective_window).mean()
        rolling_std = df["Daily_Return"].rolling(effective_window).std()
        df["Sharpe_Ratio"] = rolling_mean / rolling_std.replace(0.0, np.nan)

        # Replace any residual inf, then fill NaN from rolling calculations
        df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill()

        logger.info(
            f"CAPM features created: window={effective_window}, "
            f"rf={risk_free_rate:.4f}, {len(df)} records"
        )
        return df
