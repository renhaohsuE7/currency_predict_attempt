"""
Pydantic Settings 配置模型

使用 Pydantic 定義配置 schema 並提供驗證
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelParams(BaseModel):
    """模型參數配置 (支援 sklearn 和 transformer 版本)"""

    # 通用參數 (兩版本共用)
    seq_len: int = Field(64, gt=0, description="輸入序列長度 (sklearn/transformer)")
    pred_len: int = Field(7, gt=0, description="預測長度 (sklearn/transformer)")
    patch_len: int = Field(8, gt=0, description="Patch 長度")
    stride: int = Field(4, gt=0, description="Patch 步長")
    random_state: int = Field(42, description="隨機種子")

    # sklearn 專用參數
    n_estimators: int = Field(100, gt=0, description="估計器數量 (sklearn)")
    max_depth: int = Field(10, gt=0, description="最大深度 (sklearn)")
    estimator: str = Field(
        "gradient_boosting",
        description="sklearn 底層回歸器: 'gradient_boosting' | 'hist_gradient_boosting'"
        "（hist 對大量樣本如 panel 訓練快很多）",
    )

    # transformer 專用參數
    d_model: int = Field(64, gt=0, description="Transformer 隱藏層維度")
    num_attention_heads: int = Field(4, gt=0, description="注意力頭數量")
    num_hidden_layers: int = Field(2, gt=0, description="Transformer 層數")
    ffn_dim: int = Field(256, gt=0, description="前饋網路維度")
    dropout: float = Field(0.1, ge=0, le=1, description="Dropout 率")
    num_parallel_samples: int = Field(100, gt=0, description="並行採樣數量")

    # 預訓練模型參數
    pretrained_model_name_or_path: Optional[str] = Field(
        None,
        description="預訓練模型名稱或路徑 (例如 'ibm-granite/granite-timeseries-patchtst')",
    )
    fine_tune_mode: str = Field(
        "from_scratch",
        description="Fine-tune 模式: 'from_scratch', 'full', 'linear_probe'",
    )

    @field_validator("patch_len")
    @classmethod
    def validate_patch_len(cls, v, info):
        """驗證 patch_len 不大於 seq_len"""
        # 注意：在 Pydantic v2 中，我們無法在這裡訪問其他欄位
        # 需要在 Settings 層級做整體驗證
        return v


class DataCollectionConfig(BaseModel):
    """資料收集配置"""

    period: str = Field("2y", description="資料期間")
    interval: str = Field("1d", description="資料間隔")
    force_update: bool = Field(False, description="是否強制更新")


class TrainParams(BaseModel):
    """訓練參數"""

    validation_split: float = Field(0.2, ge=0, le=1, description="驗證集比例")


class ModelTrainingConfig(BaseModel):
    """模型訓練配置"""

    period: str = Field("1y", description="訓練資料期間")
    target_column: str = Field("Close", description="目標欄位")
    target_transform: str = Field(
        "price",
        description='目標轉換："price"（直接預測價格）或 "log_return"（預測對數報酬，輸出時還原為價格）',
    )
    feature_columns: Optional[List[str]] = Field(None, description="特徵欄位")
    train_params: TrainParams = Field(default_factory=TrainParams)
    test_days: Optional[int] = Field(
        None,
        ge=1,
        description="Test set 固定天數。None = max(2 * prediction_horizon, 30)",
    )

    @field_validator("target_transform")
    @classmethod
    def _validate_target_transform(cls, v: str) -> str:
        allowed = {"price", "log_return"}
        if v not in allowed:
            raise ValueError(f"target_transform 必須是 {allowed}，得到 '{v}'")
        return v


class PredictionConfig(BaseModel):
    """預測配置"""

    period: str = Field("1y", description="預測資料期間")
    return_uncertainty: bool = Field(True, description="是否返回不確定性")


class CAPMConfig(BaseModel):
    """Optional CAPM configuration for stock analysis.

    When enabled, the pipeline computes CAPM-related features
    (Beta, Alpha, Sharpe Ratio) for stock symbols.
    Forex and crypto symbols are unaffected.
    """

    enabled: bool = Field(False, description="啟用 CAPM 特徵（僅對股票有效）")
    market_index: str = Field("^GSPC", description="市場基準指數（預設 S&P 500）")
    risk_free_rate_symbol: str = Field(
        "^IRX", description="無風險利率符號（13-week T-Bill）"
    )
    rolling_window: int = Field(252, gt=0, description="滾動窗口大小（交易日）")


class BacktestConfig(BaseModel):
    """Backtesting 配置"""

    enabled: bool = Field(False, description="啟用 walk-forward backtesting")
    strategy: str = Field("rolling", description="策略: rolling | expanding")
    initial_train_days: int = Field(252, gt=0, description="初始訓練窗口（交易日）")
    test_step_days: int = Field(30, gt=0, description="每次前進步數（交易日）")
    test_window_days: int = Field(30, gt=0, description="測試窗口大小（交易日）")
    data_period: str = Field("3y", description="Backtest 資料收集期間")


class PanelConfig(BaseModel):
    """多股 Panel 訓練配置.

    啟用時,對 ``symbols`` 列出的所有股票各自抽取序列,concat 成單一訓練集,
    訓練一個全域模型,再對每檔預測。需搭配 ``model_training.target_transform="log_return"``
    （報酬空間才跨股可比）。
    """

    enabled: bool = Field(False, description="啟用多股 panel 訓練")
    symbols: List[str] = Field(
        default_factory=list,
        description="Panel universe（股票 ticker 列表）",
    )
    feature_columns: Optional[List[str]] = Field(
        None,
        description="固定特徵欄位；None = 自動取所有股票的欄位交集",
    )
    min_history_days: int = Field(
        500, gt=0, description="每檔最少歷史交易日，不足者跳過"
    )


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
        env_nested_delimiter="__",  # 巢狀欄位分隔符
        env_file=".env",  # .env 文件路徑
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",  # 允許額外欄位
    )

    # 基本配置
    model_name: str = Field("patchtst_sklearn", description="模型名稱")
    model_params: ModelParams = Field(default_factory=ModelParams)

    # 路徑配置
    data_storage_path: str = Field("data", description="資料儲存路徑")
    log_level: str = Field("INFO", description="日誌級別")

    # 資料相關配置
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    model_training: ModelTrainingConfig = Field(default_factory=ModelTrainingConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)

    # CAPM 配置（預設關閉）
    capm: CAPMConfig = Field(default_factory=CAPMConfig)

    # Backtesting 配置（預設關閉）
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # 多股 Panel 訓練配置（預設關閉）
    panel: PanelConfig = Field(default_factory=PanelConfig)

    # 預測配置
    symbols: List[str] = Field(
        default=["USDTWD=X", "EURUSD=X", "GBPUSD=X"],
        description="要預測的符號列表（貨幣對如 USDTWD=X 或股票 ticker 如 AAPL）",
    )
    prediction_horizon: int = Field(7, gt=0, description="預測範圍（天數）")

    # 多模型比較
    model_names: Optional[List[str]] = Field(
        default=None,
        description="多模型比較時使用的模型名稱列表（如 ['sklearn', 'huggingface']）",
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v):
        """驗證 symbols 不為空"""
        if not v:
            raise ValueError("symbols 不能為空")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """驗證日誌級別"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level 必須是以下之一: {', '.join(valid_levels)}")
        return v.upper()

    def get_model_config(self) -> Dict[str, Any]:
        """取得模型配置"""
        return {
            "model_name": self.model_name,
            "model_params": self.model_params.model_dump(),
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

    with open(path, "r", encoding="utf-8") as f:
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

    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2, ensure_ascii=False)
