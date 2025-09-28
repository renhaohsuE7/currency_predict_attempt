"""Data Collection Module

This module provides functionality to collect currency exchange rate data
from various APIs like Yahoo Finance, Alpha Vantage, etc.
"""

import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurrencyDataCollector:
    """Collects currency exchange rate data from various sources."""
    
    def __init__(self):
        self.supported_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'AUDUSD=X', 
            'CADUSD=X', 'CHFUSD=X', 'NZDUSD=X', 'USDSEK=X'
        ]
    
    def get_yahoo_finance_data(
        self, 
        currency_pair: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch currency data from Yahoo Finance.
        
        Args:
            currency_pair: Currency pair symbol (e.g., 'EURUSD=X')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(currency_pair)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {currency_pair}")
                return pd.DataFrame()
            
            logger.info(f"Successfully fetched {len(data)} records for {currency_pair}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {currency_pair}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_currencies(
        self, 
        currency_pairs: List[str], 
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple currency pairs.
        
        Args:
            currency_pairs: List of currency pair symbols
            period: Time period for data collection
            
        Returns:
            Dictionary mapping currency pairs to their DataFrames
        """
        results = {}
        
        for pair in currency_pairs:
            logger.info(f"Fetching data for {pair}...")
            data = self.get_yahoo_finance_data(pair, period)
            if not data.empty:
                results[pair] = data
                
        return results
    
    def get_supported_pairs_data(self, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch data for all supported currency pairs."""
        return self.get_multiple_currencies(self.supported_pairs, period)
    
    def save_data(self, data: pd.DataFrame, filepath: str) -> bool:
        """
        Save DataFrame to CSV file.
        
        Args:
            data: DataFrame to save
            filepath: Path where to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
            return False