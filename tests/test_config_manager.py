"""
測試配置管理器

測試 ConfigManager 和 ConfigValidator 的所有功能
"""

import pytest
import json
import tempfile
from pathlib import Path

from currency_predictor.config.manager import ConfigManager, ConfigValidator


@pytest.fixture
def temp_config_file(tmp_path):
    """創建臨時配置文件"""
    config = {
        "model_name": "PatchTST",
        "model_params": {
            "seq_len": 100,
            "pred_len": 10
        },
        "symbols": ["USDTWD=X", "EURUSD=X"],
        "prediction_horizon": 5
    }

    config_file = tmp_path / "test_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f)

    return config_file


@pytest.fixture
def invalid_config_file(tmp_path):
    """創建無效的配置文件"""
    invalid_config = {
        "model_name": "PatchTST"
        # 缺少必要的鍵
    }

    config_file = tmp_path / "invalid_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_config, f)

    return config_file


class TestConfigValidator:
    """測試 ConfigValidator 類別"""

    def test_validate_valid_config(self):
        """測試驗證有效配置"""
        valid_config = {
            "model_name": "PatchTST",
            "model_params": {
                "seq_len": 100,
                "pred_len": 10
            },
            "symbols": ["USDTWD=X"],
            "prediction_horizon": 7
        }

        is_valid, errors = ConfigValidator.validate_config(valid_config)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_required_keys(self):
        """測試缺少必要鍵的配置（Pydantic 會使用默認值）"""
        config_with_minimal_keys = {
            "model_name": "PatchTST"
            # 缺少 model_params 和 symbols，但 Pydantic 會使用默認值
        }

        is_valid, errors = ConfigValidator.validate_config(config_with_minimal_keys)

        # Pydantic Settings 會自動填充默認值，所以這是有效的
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_model_params(self):
        """測試 model_params 缺少必要參數（Pydantic 會使用默認值）"""
        config = {
            "model_name": "PatchTST",
            "model_params": {
                "seq_len": 100
                # 缺少 pred_len，但 Pydantic 會使用默認值
            },
            "symbols": ["USDTWD=X"]
        }

        is_valid, errors = ConfigValidator.validate_config(config)

        # Pydantic 會為 pred_len 使用默認值，所以這是有效的
        assert is_valid is True

    def test_validate_invalid_seq_len(self):
        """測試無效的 seq_len 值"""
        config = {
            "model_name": "PatchTST",
            "model_params": {
                "seq_len": -10,  # 無效值
                "pred_len": 10
            },
            "symbols": ["USDTWD=X"]
        }

        is_valid, errors = ConfigValidator.validate_config(config)

        assert is_valid is False
        # Pydantic 的錯誤訊息格式為 "model_params: Input should be greater than 0"
        assert any("model_params" in err or "greater than 0" in err for err in errors)

    def test_validate_empty_symbols(self):
        """測試空的 symbols 列表"""
        config = {
            "model_name": "PatchTST",
            "model_params": {
                "seq_len": 100,
                "pred_len": 10
            },
            "symbols": []  # 空列表
        }

        is_valid, errors = ConfigValidator.validate_config(config)

        assert is_valid is False
        assert any("symbols" in err for err in errors)


class TestConfigManager:
    """測試 ConfigManager 類別"""

    def test_init_with_existing_file(self, temp_config_file):
        """測試從現有文件初始化"""
        manager = ConfigManager(config_path=str(temp_config_file))

        assert manager._settings is not None
        assert manager.get('model_name') == 'PatchTST'
        assert manager.get('symbols') == ["USDTWD=X", "EURUSD=X"]

    def test_init_without_file(self, tmp_path):
        """測試從不存在的文件初始化（使用默認配置）"""
        non_existent_file = tmp_path / "non_existent.json"
        manager = ConfigManager(config_path=str(non_existent_file), validate=False)

        assert manager._settings is not None
        # 應該使用默認配置
        assert manager.get('model_name') == 'patchtst_sklearn'
        assert len(manager.get('symbols', [])) > 0

    def test_get_config(self, temp_config_file):
        """測試取得配置"""
        manager = ConfigManager(config_path=str(temp_config_file))
        config = manager.get_config()

        assert isinstance(config, dict)
        assert 'model_name' in config
        assert 'symbols' in config

    def test_get_method(self, temp_config_file):
        """測試 get 方法"""
        manager = ConfigManager(config_path=str(temp_config_file))

        # 取得存在的鍵
        assert manager.get('model_name') == 'PatchTST'

        # 取得不存在的鍵（使用默認值）
        assert manager.get('non_existent_key', 'default_value') == 'default_value'

    def test_update_config(self, temp_config_file):
        """測試更新配置"""
        manager = ConfigManager(config_path=str(temp_config_file), validate=False)

        # 更新配置
        manager.update({'new_key': 'new_value'})

        assert manager.get('new_key') == 'new_value'

    def test_save_config(self, temp_config_file):
        """測試儲存配置"""
        manager = ConfigManager(config_path=str(temp_config_file), validate=False)

        # 更新並儲存
        manager.update({'new_key': 'new_value'})
        manager.save()

        # 重新載入驗證
        new_manager = ConfigManager(config_path=str(temp_config_file), validate=False)
        assert new_manager.get('new_key') == 'new_value'

    def test_get_model_config(self, temp_config_file):
        """測試取得模型配置"""
        manager = ConfigManager(config_path=str(temp_config_file))
        model_config = manager.get_model_config()

        assert 'model_name' in model_config
        assert 'model_params' in model_config
        assert model_config['model_name'] == 'PatchTST'

    def test_get_symbols(self, temp_config_file):
        """測試取得貨幣符號列表"""
        manager = ConfigManager(config_path=str(temp_config_file))
        symbols = manager.get_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "USDTWD=X" in symbols

    def test_get_prediction_horizon(self, temp_config_file):
        """測試取得預測範圍"""
        manager = ConfigManager(config_path=str(temp_config_file))
        horizon = manager.get_prediction_horizon()

        assert isinstance(horizon, int)
        assert horizon > 0

    def test_validation_on_load(self, invalid_config_file):
        """測試載入時的驗證"""
        # 使用驗證（默認）
        manager = ConfigManager(config_path=str(invalid_config_file), validate=True)

        # 應該自動修正配置
        assert manager.get('symbols') is not None
        assert manager.get('model_params') is not None

    def test_default_config_values(self):
        """測試默認配置值"""
        assert 'model_name' in ConfigManager.DEFAULT_CONFIG
        assert 'model_params' in ConfigManager.DEFAULT_CONFIG
        assert 'symbols' in ConfigManager.DEFAULT_CONFIG

    def test_config_deep_copy(self, temp_config_file):
        """測試配置返回深拷貝"""
        manager = ConfigManager(config_path=str(temp_config_file))

        config1 = manager.get_config()
        config2 = manager.get_config()

        # 修改一個不應影響另一個
        config1['new_key'] = 'value'

        assert 'new_key' not in config2


class TestDataCollectionConfig:
    """Test that data_collection config fields load correctly."""

    def test_data_collection_period_loads_from_config(self, tmp_path):
        """data_collection.period from JSON maps to AppSettings correctly."""
        config = {
            "model_name": "patchtst_sklearn",
            "symbols": ["USDTWD=X"],
            "data_collection": {"period": "3y", "interval": "1d"},
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        manager = ConfigManager(config_path=str(config_file))
        assert manager.get("data_collection.period") == "3y"
        assert manager.get("data_collection.interval") == "1d"

    def test_data_collection_propagates_to_get_config(self, tmp_path):
        """get_config() output contains data_collection with correct values."""
        config = {
            "model_name": "patchtst_sklearn",
            "symbols": ["USDTWD=X"],
            "data_collection": {"period": "5y", "interval": "1wk"},
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        manager = ConfigManager(config_path=str(config_file))
        full_config = manager.get_config()
        assert full_config["data_collection"]["period"] == "5y"
        assert full_config["data_collection"]["interval"] == "1wk"

    def test_missing_data_collection_uses_pydantic_defaults(self, tmp_path):
        """No data_collection section falls back to Pydantic defaults."""
        config = {
            "model_name": "patchtst_sklearn",
            "symbols": ["USDTWD=X"],
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        manager = ConfigManager(config_path=str(config_file))
        assert manager.get("data_collection.period") == "2y"
        assert manager.get("data_collection.interval") == "1d"

    def test_custom_config_file_loads_stock_symbols(self, tmp_path):
        """Config with stock ticker loads symbols correctly."""
        config = {
            "model_name": "patchtst_sklearn",
            "symbols": ["2330.TW"],
            "prediction_horizon": 7,
            "data_collection": {"period": "2y", "interval": "1d"},
        }
        config_file = tmp_path / "tw2330_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        manager = ConfigManager(config_path=str(config_file))
        assert manager.get_symbols() == ["2330.TW"]
        assert manager.get_prediction_horizon() == 7
        assert manager.get("data_collection.period") == "2y"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
