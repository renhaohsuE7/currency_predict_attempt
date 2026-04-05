"""
測試工具模組

測試 utils.py 中的所有工具函數
"""

import pytest
import json
import logging
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from currency_predictor.utils import (
    setup_logging,
    load_config,
    save_config,
    get_default_config,
    save_model,
    load_model,
    ensure_directories,
    calculate_returns,
    format_currency_pair,
    validate_currency_pair,
)


class TestLogging:
    """測試日誌設置功能"""

    def test_setup_logging_default(self):
        setup_logging()
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO

    def test_setup_logging_custom_level(self):
        setup_logging(level="DEBUG")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_warning_level(self):
        setup_logging(level="WARNING")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_setup_logging_with_file(self, tmp_path):
        log_file = tmp_path / "subdir" / "test.log"
        setup_logging(level="INFO", log_file=str(log_file))
        logger = logging.getLogger("test_file_logger")
        logger.info("Test message")
        assert log_file.exists()

    def test_setup_logging_custom_format(self):
        setup_logging(format_string="%(message)s")
        root_logger = logging.getLogger()
        assert root_logger.level is not None

    def test_logging_output(self, caplog):
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("test")
            logger.info("Test info message")
        assert "Test info message" in caplog.text


class TestConfig:
    """測試配置讀寫"""

    def test_load_config_file_not_found(self):
        result = load_config("/nonexistent/path/config.json")
        assert isinstance(result, dict)
        assert 'data' in result  # returns default config

    def test_load_config_valid_file(self, tmp_path):
        config = {"key": "value", "number": 42}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))
        result = load_config(str(config_file))
        assert result == config

    def test_load_config_invalid_json(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not valid json {{{")
        result = load_config(str(config_file))
        assert isinstance(result, dict)  # returns default

    def test_save_config(self, tmp_path):
        config = {"model": "test", "value": 123}
        config_file = tmp_path / "output.json"
        result = save_config(config, str(config_file))
        assert result is True
        assert config_file.exists()
        loaded = json.loads(config_file.read_text())
        assert loaded == config

    def test_save_config_failure(self):
        result = save_config({"a": 1}, "/nonexistent/dir/config.json")
        assert result is False

    def test_get_default_config(self):
        config = get_default_config()
        assert isinstance(config, dict)
        assert 'data' in config
        assert 'models' in config
        assert 'paths' in config
        assert 'preprocessing' in config


class TestModelPersistence:
    """測試模型存取"""

    def test_save_and_load_model(self, tmp_path):
        model = {"type": "test", "weights": [1, 2, 3]}
        filepath = tmp_path / "models" / "test.pkl"
        assert save_model(model, str(filepath))
        loaded = load_model(str(filepath))
        assert loaded == model

    def test_save_model_creates_directory(self, tmp_path):
        filepath = tmp_path / "deep" / "nested" / "model.pkl"
        assert save_model({"x": 1}, str(filepath))
        assert filepath.exists()

    def test_load_model_not_found(self):
        result = load_model("/nonexistent/model.pkl")
        assert result is None


class TestEnsureDirectories:
    """測試目錄確認"""

    def test_creates_directories(self, tmp_path):
        dirs = [
            str(tmp_path / "dir_a"),
            str(tmp_path / "dir_b" / "nested"),
        ]
        ensure_directories(dirs)
        for d in dirs:
            assert Path(d).exists()

    def test_existing_directories(self, tmp_path):
        d = str(tmp_path / "existing")
        Path(d).mkdir()
        ensure_directories([d])  # should not raise
        assert Path(d).exists()


class TestCalculateReturns:
    """測試報酬率計算"""

    def test_pandas_series(self):
        prices = pd.Series([100, 110, 105, 115])
        returns = calculate_returns(prices)
        assert len(returns) == 3
        assert abs(returns.iloc[0] - 0.1) < 1e-6

    def test_numpy_array(self):
        prices = np.array([100.0, 110.0, 105.0])
        returns = calculate_returns(prices)
        assert len(returns) == 2
        assert abs(returns[0] - 0.1) < 1e-6


class TestCurrencyPairFormatting:
    """測試貨幣對格式化"""

    def test_format_plain(self):
        assert format_currency_pair("EURUSD") == "EURUSD=X"

    def test_format_with_slash(self):
        assert format_currency_pair("EUR/USD") == "EURUSD=X"

    def test_format_already_formatted(self):
        assert format_currency_pair("EURUSD=X") == "EURUSD=X"

    def test_format_lowercase(self):
        assert format_currency_pair("eurusd") == "EURUSD=X"

    def test_validate_valid_pair(self):
        assert validate_currency_pair("EURUSD=X") is True
        assert validate_currency_pair("EUR/USD") is True
        assert validate_currency_pair("eurusd") is True

    def test_validate_invalid_pair(self):
        assert validate_currency_pair("EU") is False
        assert validate_currency_pair("TOOLONGCURRENCY") is False
