"""
配置管理器

基於 Pydantic Settings 提供配置的載入、驗證和管理功能
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import ValidationError

from .settings import AppSettings, load_settings_from_json, save_settings_to_json

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    配置驗證器

    提供向後兼容的驗證介面
    """

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        驗證配置的有效性

        Args:
            config: 配置字典

        Returns:
            (是否有效, 錯誤訊息列表)
        """
        try:
            AppSettings(**config)
            return True, []
        except ValidationError as e:
            errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            return False, errors


class ConfigManager:
    """
    配置管理器

    基於 Pydantic Settings 的配置管理，提供：
    - 從 JSON 文件載入配置
    - 從環境變數載入配置
    - 配置驗證
    - 配置更新和儲存
    """

    def __init__(self, config_path: Optional[str] = None, validate: bool = True):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路徑，如果為 None 則使用默認配置
            validate: 是否驗證配置（總是為 True，保持向後兼容）
        """
        self.config_path = Path(config_path) if config_path else Path("config.json")
        self.validate = validate  # 保持向後兼容，實際上總是驗證
        self._settings: AppSettings = AppSettings()
        self._load()

    def _load(self):
        """載入配置"""
        try:
            if self.config_path.exists():
                logger.info(f"從文件載入配置: {self.config_path}")
                self._settings = load_settings_from_json(self.config_path)
            else:
                logger.info("配置文件不存在，使用默認配置")
                # 嘗試從環境變數載入，否則使用默認值
                self._settings = AppSettings()

            logger.info("配置載入成功")

        except ValidationError as e:
            logger.error(f"配置驗證失敗:\n{e}")
            logger.info("使用默認配置")
            self._settings = AppSettings()

        except Exception as e:
            logger.error(f"載入配置文件失敗: {e}")
            logger.info("使用默認配置")
            self._settings = AppSettings()

    def get_config(self) -> Dict[str, Any]:
        """
        取得配置字典

        Returns:
            配置字典
        """
        return self._settings.model_dump()

    def get(self, key: str, default: Any = None) -> Any:
        """
        取得配置值

        Args:
            key: 配置鍵（支援點號分隔的巢狀鍵，如 'model_params.seq_len'）
            default: 默認值

        Returns:
            配置值
        """
        try:
            # 支援巢狀鍵
            keys = key.split('.')
            value = self._settings

            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default

            # 如果值是 Pydantic 模型，轉換為字典
            if hasattr(value, 'model_dump'):
                return value.model_dump()

            return value

        except Exception:
            return default

    def update(self, updates: Dict[str, Any], save: bool = False):
        """
        更新配置

        Args:
            updates: 要更新的配置字典
            save: 是否儲存到文件
        """
        try:
            # 合併現有配置和更新
            current_config = self._settings.model_dump()
            current_config.update(updates)

            # 創建新的 Settings 實例（會自動驗證）
            self._settings = AppSettings(**current_config)

            logger.info("配置更新成功")

            if save:
                self.save()

        except ValidationError as e:
            logger.error(f"配置更新失敗: {e}")
            raise

    def save(self, file_path: Optional[Path] = None):
        """
        儲存配置到文件

        Args:
            file_path: 儲存路徑，如果為 None 則使用原路徑
        """
        save_path = file_path or self.config_path

        try:
            save_settings_to_json(self._settings, save_path)
            logger.info(f"配置已儲存至: {save_path}")
        except Exception as e:
            logger.error(f"儲存配置失敗: {e}")

    def get_model_config(self) -> Dict[str, Any]:
        """取得模型配置"""
        return self._settings.get_model_config()

    def get_symbols(self) -> List[str]:
        """取得貨幣符號列表"""
        return self._settings.get_symbols()

    def get_prediction_horizon(self) -> int:
        """取得預測範圍"""
        return self._settings.get_prediction_horizon()

    def get_model_names(self) -> Optional[List[str]]:
        """取得多模型比較的模型名稱列表（None 表示未設定）"""
        return self._settings.model_names

    @property
    def settings(self) -> AppSettings:
        """取得 Pydantic Settings 實例"""
        return self._settings

    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}')"


# 向後兼容：保留 DEFAULT_CONFIG
# 現在從 AppSettings 生成
def get_default_config() -> Dict[str, Any]:
    """取得默認配置字典（向後兼容）"""
    return AppSettings().model_dump()


# 為 ConfigManager 類添加 DEFAULT_CONFIG 屬性
ConfigManager.DEFAULT_CONFIG = get_default_config()  # type: ignore[attr-defined]
