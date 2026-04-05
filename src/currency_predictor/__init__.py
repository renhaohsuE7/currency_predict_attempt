"""Currency Predictor Package

A machine learning package for predicting currency exchange rates.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data.collectors import YahooFinanceCollector
from .data_processor import DataProcessor

# 向後兼容別名
CurrencyDataCollector = YahooFinanceCollector
from .prediction import CurrencyPredictor, PredictionPipeline
from .utils import (
    load_config, 
    save_model, 
    load_model, 
    setup_logging, 
    ensure_directories, 
    get_default_config,
    validate_currency_pair,
    format_currency_pair
)

__all__ = [
    "CurrencyDataCollector",
    "DataProcessor", 
    "CurrencyPredictor",
    "PredictionPipeline",
    "load_config",
    "save_model",
    "load_model",
    "setup_logging",
    "ensure_directories",
    "get_default_config",
    "validate_currency_pair",
    "format_currency_pair"
]