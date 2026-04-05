"""
PatchTST 模型模組

提供多種 PatchTST 實作:
- sklearn: 基於 sklearn 的簡化版本 (快速，適合原型開發)
- huggingface: 基於 HuggingFace Transformers (成熟，有預訓練支援)
- lightning: 基於 PyTorch Lightning (靈活，適合研究和自定義)

使用範例:
    # 直接導入
    from currency_predictor.models.patchtst import PatchTSTSklearn
    from currency_predictor.models.patchtst import PatchTSTHuggingFace
    from currency_predictor.models.patchtst import PatchTSTLightningWrapper

    # 使用配置
    from currency_predictor.models.patchtst import PatchTSTConfig
    config = PatchTSTConfig(context_length=64, prediction_length=7)
    model = PatchTSTSklearn(config=config)

    # 向後兼容
    from currency_predictor.models.patchtst import PatchTST  # 等同於 PatchTSTSklearn
    from currency_predictor.models.patchtst import PatchTSTTransformer  # 等同於 PatchTSTHuggingFace
"""

from .config import PatchTSTConfig, TrainingConfig
from .sklearn import PatchTSTSklearn, PatchTST

# 條件導入 HuggingFace 版本
try:
    from .huggingface import PatchTSTHuggingFace, PatchTSTTransformer
    _has_huggingface = True
except ImportError:
    PatchTSTHuggingFace = None  # type: ignore[assignment,misc]
    PatchTSTTransformer = None  # type: ignore[assignment,misc]
    _has_huggingface = False

# 條件導入 Lightning 版本
try:
    from .lightning import PatchTSTLightningWrapper
    _has_lightning = True
except ImportError:
    PatchTSTLightningWrapper = None  # type: ignore[assignment,misc]
    _has_lightning = False

__all__ = [
    # 配置
    'PatchTSTConfig',
    'TrainingConfig',
    # sklearn 版本
    'PatchTSTSklearn',
    'PatchTST',  # 向後兼容別名
]

if _has_huggingface:
    __all__.extend([
        'PatchTSTHuggingFace',
        'PatchTSTTransformer',  # 向後兼容別名
    ])

if _has_lightning:
    __all__.append('PatchTSTLightningWrapper')
