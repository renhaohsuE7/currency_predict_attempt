"""
PatchTST HuggingFace 實作

基於 HuggingFace Transformers 的 PatchTST 模型
"""

try:
    from .model import PatchTSTHuggingFace, PatchTSTTransformer
    __all__ = ['PatchTSTHuggingFace', 'PatchTSTTransformer']
except ImportError:
    # transformers 未安裝
    PatchTSTHuggingFace = None  # type: ignore[assignment,misc]
    PatchTSTTransformer = None  # type: ignore[assignment,misc]
    __all__ = []
