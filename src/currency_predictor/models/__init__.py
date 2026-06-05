"""
模型模組

提供各種時間序列預測模型
"""

from .base import (
    BaseModel,
    TimeSeriesModel,
    SklearnBasedModel,
    TransformerBasedModel,
    ModelType,
)

# 從 patchtst 模組導入
from .patchtst import PatchTSTSklearn, PatchTSTConfig
from .patchtst import PatchTST  # 向後兼容別名

# Naive baseline
from .naive import NaiveModel

# 條件導入 HuggingFace 版本
try:
    from .patchtst import PatchTSTHuggingFace, PatchTSTTransformer

    _has_transformers = True
except (ImportError, TypeError):
    PatchTSTHuggingFace = None  # type: ignore[assignment,misc]
    PatchTSTTransformer = None  # type: ignore[assignment,misc]
    _has_transformers = False

# 從 factory 導入
from .factory import ModelFactory, create_patchtst_model

__all__ = [
    # 基礎類別
    "BaseModel",
    "TimeSeriesModel",
    "SklearnBasedModel",
    "TransformerBasedModel",
    "ModelType",
    # 配置
    "PatchTSTConfig",
    # sklearn 版本
    "PatchTSTSklearn",
    "PatchTST",  # 向後兼容別名
    # Naive baseline
    "NaiveModel",
    # 工廠
    "ModelFactory",
    "create_patchtst_model",
]

if _has_transformers:
    __all__.extend(
        [
            "PatchTSTHuggingFace",
            "PatchTSTTransformer",  # 向後兼容別名
        ]
    )
