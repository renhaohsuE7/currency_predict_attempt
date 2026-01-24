"""
模型模組

提供各種時間序列預測模型
"""

from .base import BaseModel, TimeSeriesModel, SklearnBasedModel, TransformerBasedModel, ModelType
from .patchtst import PatchTST

# 條件導入 transformer 模型
try:
    from .patchtst_transformer import PatchTSTTransformer
    _has_transformers = True
except ImportError:
    PatchTSTTransformer = None
    _has_transformers = False

__all__ = [
    'BaseModel',
    'TimeSeriesModel', 
    'SklearnBasedModel',
    'TransformerBasedModel',
    'ModelType',
    'PatchTST'
]

if _has_transformers:
    __all__.append('PatchTSTTransformer')