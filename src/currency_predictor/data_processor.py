"""Data Processing Module

This module provides functionality to clean, preprocess, and engineer features
from currency exchange rate data for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and prepares currency data for machine learning."""
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = []
    
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
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        logger.info(f"Data cleaned: {len(df)} records remaining")
        return df
    
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators for currency prediction.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = data.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        df['Price_Change_10'] = df['Close'].pct_change(periods=10)
        
        # Volume indicators (if volume data available)
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        logger.info(f"Technical indicators created: {len(df.columns)} total features")
        return df
    
    def create_lagged_features(self, data: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.
        
        Args:
            data: DataFrame with features
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lagged features
        """
        df = data.copy()
        
        # Create lagged features for Close price
        for lag in lags:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else np.nan
            df[f'Price_Change_lag_{lag}'] = df['Price_Change'].shift(lag)
        
        # Drop rows with NaN values created by lagging
        df = df.dropna()
        
        logger.info(f"Lagged features created: {len(df)} records after removing NaN")
        return df
    
    def prepare_features_target(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'Close',
        prediction_horizon: int = 1,
        feature_columns: Optional[list] = None
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
        df['target'] = df[target_column].shift(-prediction_horizon)
        
        # Remove rows where target is NaN
        df = df.dropna()
        
        # Select feature columns
        if feature_columns is None:
            # Use all numeric columns except target and OHLCV
            exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume']
            feature_columns = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        self.feature_columns = feature_columns
        
        X = df[feature_columns]
        y = df['target']
        
        logger.info(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        return X, y
    
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        scaler_type: str = 'standard'
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
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        # Fit scaler on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info(f"Features scaled using {scaler_type} scaler")
        return X_train_scaled, X_test_scaled
    
    def train_test_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2
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