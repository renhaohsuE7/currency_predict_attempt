"""
PatchTST PyTorch Lightning 實作

基於 PyTorch Lightning 的 PatchTST 模型，提供:
- 完整的訓練控制
- 豐富的回調系統
- 易於調試和擴展
- 支援分散式訓練

使用範例:
    from currency_predictor.models.patchtst.lightning import PatchTSTLightningWrapper

    model = PatchTSTLightningWrapper(seq_len=64, pred_len=7)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

import logging

logger = logging.getLogger(__name__)

# 嘗試導入 Lightning 模組
try:
    from .modules import (
        PatchEmbedding,
        TransformerEncoder,
        TransformerEncoderLayer,
        PatchTSTModel,
        PositionalEncoding,
        PredictionHead
    )
    from .lightning_module import PatchTSTLightning, get_lightning_callbacks, HAS_LIGHTNING as _HAS_PL
    from .wrapper import PatchTSTLightningWrapper

    # 只有在 pytorch_lightning 實際可用時才設為 True
    HAS_LIGHTNING = _HAS_PL

    if HAS_LIGHTNING:
        __all__ = [
            # 核心模組
            'PatchEmbedding',
            'TransformerEncoder',
            'TransformerEncoderLayer',
            'PatchTSTModel',
            'PositionalEncoding',
            'PredictionHead',
            # Lightning 模組
            'PatchTSTLightning',
            'get_lightning_callbacks',
            # 封裝
            'PatchTSTLightningWrapper',
        ]
    else:
        logger.warning("pytorch_lightning 未安裝，PatchTSTLightningWrapper 不可用")
        PatchTSTLightningWrapper = None
        PatchTSTLightning = None
        __all__ = []

except ImportError as e:
    logger.warning(f"無法導入 Lightning 模組: {e}")
    HAS_LIGHTNING = False
    __all__ = []

    # 提供空的佔位符
    PatchTSTLightningWrapper = None
    PatchTSTLightning = None
