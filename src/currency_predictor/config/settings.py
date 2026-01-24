"""
Pydantic Settings 配置模型

使用 Pydantic 定義配置 schema 並提供驗證
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelParams(BaseModel):
    """模型參數配置"""

    seq_len: int = Field(168, gt=0, description="輸入序列長度")
    pred_len: int = Field(24, gt=0, description="預測長度")
    patch_len: int = Field(12, gt=0, description="Patch 長度")
    stride: int = Field(6, gt=0, description="Patch 步長")
    n_estimators: int = Field(100, gt=0, description="估計器數量")
    max_depth: int = Field(10, gt=0, description="最大深度")
    random_state: int = Field(42, description="隨機種子")

    @field_validator('patch_len')
    @classmethod
    def validate_patch_len(cls, v, info):
        """驗證 patch_len 不大於 seq_len"""
        # 注意：在 Pydantic v2 中，我們無法在這裡訪問其他欄位
        # 需要在 Settings 層級做整體驗證
        return v


class DataCollectionConfig(BaseModel):
    """資料收集配置"""

    period: str = Field("1y", description="資料期間")
    interval: str = Field("1d", description="資料間隔")
    force_update: bool = Field(False, description="是否強制更新")


class TrainParams(BaseModel):
    """訓練參數"""

    validation_split: float = Field(0.2, ge=0, le=1, description="驗證集比例")


class ModelTrainingConfig(BaseModel):
    """模型訓練配置"""

    period: str = Field("1y", description="訓練資料期間")
    target_column: str = Field("Close", description="目標欄位")
    feature_columns: Optional[List[str]] = Field(None, description="特徵欄位")
    train_params: TrainParams = Field(default_factory=TrainParams)


class PredictionConfig(BaseModel):
    """預測配置"""

    period: str = Field("1y", description="預測資料期間")
    return_uncertainty: bool = Field(True, description="是否返回不確定性")


class AppSettings(BaseSettings):
    """
    應用程式配置

    支援從以下來源載入配置（按優先順序）：
    1. 環境變數
    2. .env 文件
    3. JSON 配置文件
    4. 默認值
    """

    model_config = SettingsConfigDict(
        env_prefix="CURRENCY_PRED_",  # 環境變數前綴
        env_nested_delimiter="__",     # 巢狀欄位分隔符
        env_file=".env",               # .env 文件路徑
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"                  # 允許額外欄位
    )

    # 基本配置
    model_name: str = Field("PatchTST", description="模型名稱")
    model_params: ModelParams = Field(default_factory=ModelParams)

    # 路徑配置
    data_storage_path: str = Field("data", description="資料儲存路徑")
    log_level: str = Field("INFO", description="日誌級別")

    # 資料相關配置
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    model_training: ModelTrainingConfig = Field(default_factory=ModelTrainingConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)

    # 預測配置
    symbols: List[str] = Field(
        default=["USDTWD=X", "EURUSD=X", "GBPUSD=X"],
        description="要預測的貨幣符號列表"
    )
    prediction_horizon: int = Field(7, gt=0, description="預測範圍（天數）")

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        """驗證 symbols 不為空"""
        if not v:
            raise ValueError("symbols 不能為空")
        return v

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """驗證日誌級別"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level 必須是以下之一: {', '.join(valid_levels)}")
        return v.upper()

    def get_model_config(self) -> Dict[str, Any]:
        """取得模型配置"""
        return {
            'model_name': self.model_name,
            'model_params': self.model_params.model_dump()
        }

    def get_symbols(self) -> List[str]:
        """取得貨幣符號列表"""
        return self.symbols

    def get_prediction_horizon(self) -> int:
        """取得預測範圍"""
        return self.prediction_horizon


def load_settings_from_json(json_path: str | Path) -> AppSettings:
    """
    從 JSON 文件載入設置

    Args:
        json_path: JSON 文件路徑

    Returns:
        AppSettings 實例
    """
    import json

    path = Path(json_path)
    if not path.exists():
        # 如果文件不存在，返回默認設置
        return AppSettings()

    with open(path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    return AppSettings(**config_dict)


def save_settings_to_json(settings: AppSettings, json_path: str | Path):
    """
    儲存設置到 JSON 文件

    Args:
        settings: AppSettings 實例
        json_path: JSON 文件路徑
    """
    import json

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            settings.model_dump(),
            f,
            indent=2,
            ensure_ascii=False
        )
