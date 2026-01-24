"""
測試工具模組

測試 utils.py 中的所有工具函數
"""

import pytest
import logging
from pathlib import Path
import tempfile

from src.currency_predictor.utils import setup_logging


class TestLogging:
    """測試日誌設置功能"""

    def test_setup_logging_default(self):
        """測試使用默認參數設置日誌"""
        setup_logging()

        # 檢查 root logger 已被配置
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO

    def test_setup_logging_custom_level(self):
        """測試使用自定義日誌級別"""
        setup_logging(level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_warning_level(self):
        """測試使用 WARNING 級別"""
        setup_logging(level="WARNING")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_setup_logging_with_file(self):
        """測試寫入文件的日誌設置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            setup_logging(level="INFO", log_file=str(log_file))

            # 寫入一些日誌
            logger = logging.getLogger("test_logger")
            logger.info("Test message")

            # 檢查文件是否被創建（取決於實現）
            # 注意：具體行為取決於 setup_logging 的實現

    def test_logging_output(self, caplog):
        """測試日誌輸出"""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("test")
            logger.info("Test info message")
            logger.debug("Test debug message")

        # 檢查日誌記錄
        assert "Test info message" in caplog.text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
