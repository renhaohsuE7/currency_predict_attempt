"""Utility Functions Module

This module contains utility functions for configuration management,
model persistence, and other helper functions.
"""

import json
import pickle
import os
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return dict(config)
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_path} not found. Using defaults.")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return get_default_config()


def save_config(config: Dict[str, Any], config_path: str = "config.json") -> bool:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration settings.
    
    Returns:
        Dictionary containing default configuration
    """
    return {
        "data": {
            "default_period": "1y",
            "default_interval": "1d",
            "currency_pairs": [
                "EURUSD=X", "GBPUSD=X", "JPYUSD=X", "AUDUSD=X",
                "CADUSD=X", "CHFUSD=X", "NZDUSD=X", "USDSEK=X"
            ]
        },
        "preprocessing": {
            "test_size": 0.2,
            "scaler_type": "standard",
            "prediction_horizon": 1,
            "lag_periods": [1, 2, 3, 5, 10]
        },
        "models": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42
            }
        },
        "paths": {
            "data_dir": "data",
            "models_dir": "models",
            "results_dir": "results"
        }
    }


def save_model(model: Any, filepath: str) -> bool:
    """
    Save a model to disk using pickle.
    
    Args:
        model: Model object to save
        filepath: Path where to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {str(e)}")
        return False


def load_model(filepath: str) -> Optional[Any]:
    """
    Load a model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object or None if failed
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {str(e)}")
        return None


def ensure_directories(paths: list) -> None:
    """
    Ensure that directories exist, create them if they don't.
    
    Args:
        paths: List of directory paths to ensure exist
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")


def setup_logging(
    level: str = "INFO", 
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path to write logs to
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    logger.info(f"Logging configured at {level} level")


def calculate_returns(prices: Any) -> Any:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series (pandas Series or numpy array)
        
    Returns:
        Returns series
    """
    try:
        if hasattr(prices, 'pct_change'):
            return prices.pct_change().dropna()
        else:
            # For numpy arrays
            import numpy as np
            return np.diff(prices) / prices[:-1]
    except Exception as e:
        logger.error(f"Error calculating returns: {str(e)}")
        return None


def format_currency_pair(pair: str) -> str:
    """
    Format currency pair string for Yahoo Finance.
    
    Args:
        pair: Currency pair (e.g., 'EURUSD' or 'EUR/USD')
        
    Returns:
        Formatted pair for Yahoo Finance (e.g., 'EURUSD=X')
    """
    # Remove any slashes or spaces
    clean_pair = pair.replace('/', '').replace(' ', '').upper()
    
    # Add =X suffix if not present
    if not clean_pair.endswith('=X'):
        clean_pair += '=X'
    
    return clean_pair


def validate_currency_pair(pair: str) -> bool:
    """
    Validate if currency pair format is correct.
    
    Args:
        pair: Currency pair string
        
    Returns:
        True if valid format, False otherwise
    """
    # Basic validation - should be 6 characters + =X
    formatted_pair = format_currency_pair(pair)
    
    if len(formatted_pair) != 8:  # 6 chars + =X
        return False
    
    if not formatted_pair.endswith('=X'):
        return False
    
    # Check if base and quote currencies are 3 characters each
    base_quote = formatted_pair[:-2]  # Remove =X
    if len(base_quote) != 6:
        return False
    
    return True