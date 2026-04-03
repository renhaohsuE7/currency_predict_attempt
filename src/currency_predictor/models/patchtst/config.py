"""
PatchTST 統一配置

定義所有 PatchTST 實作共用的配置類別
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class PatchTSTConfig:
    """
    PatchTST 模型配置

    支援三種實作版本:
    - sklearn: 基於 sklearn 的簡化版本
    - huggingface: 基於 HuggingFace Transformers 的版本
    - lightning: 基於 PyTorch Lightning 的版本 (開發中)

    Attributes:
        context_length: 輸入序列長度 (舊名: seq_len)
        prediction_length: 預測序列長度 (舊名: pred_len)
        patch_length: Patch 大小 (舊名: patch_len)
        patch_stride: Patch 步長 (舊名: stride)
        d_model: Transformer 隱藏層維度
        n_heads: 注意力頭數量
        n_layers: Transformer 層數
        d_ff: 前饋網路維度
        dropout: Dropout 率
        activation: 激活函數
        scaling: 標準化方式
        random_state: 隨機種子
    """

    # 序列參數
    context_length: int = 64
    prediction_length: int = 7
    patch_length: int = 8
    patch_stride: int = 4

    # Transformer 架構參數
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: Literal['gelu', 'relu'] = 'gelu'

    # 標準化
    scaling: Optional[Literal['std', 'mean', 'none']] = 'std'

    # sklearn 專用參數
    n_estimators: int = 100
    max_depth: int = 10

    # 訓練參數
    random_state: int = 42

    # 參數別名 (sklearn 風格)
    @classmethod
    def from_sklearn_params(
        cls,
        seq_len: Optional[int] = None,
        pred_len: Optional[int] = None,
        patch_len: Optional[int] = None,
        stride: Optional[int] = None,
        **kwargs
    ) -> 'PatchTSTConfig':
        """
        從 sklearn 風格參數創建配置

        Args:
            seq_len: 輸入序列長度 (映射到 context_length)
            pred_len: 預測長度 (映射到 prediction_length)
            patch_len: Patch 大小 (映射到 patch_length)
            stride: Patch 步長 (映射到 patch_stride)
            **kwargs: 其他參數

        Returns:
            PatchTSTConfig 實例
        """
        config_kwargs = kwargs.copy()

        if seq_len is not None:
            config_kwargs['context_length'] = seq_len
        if pred_len is not None:
            config_kwargs['prediction_length'] = pred_len
        if patch_len is not None:
            config_kwargs['patch_length'] = patch_len
        if stride is not None:
            config_kwargs['patch_stride'] = stride

        return cls(**config_kwargs)

    @property
    def num_patches(self) -> int:
        """計算 patch 數量"""
        return (self.context_length - self.patch_length) // self.patch_stride + 1

    # sklearn 風格別名屬性
    @property
    def seq_len(self) -> int:
        return self.context_length

    @property
    def pred_len(self) -> int:
        return self.prediction_length

    @property
    def patch_len(self) -> int:
        return self.patch_length

    @property
    def stride(self) -> int:
        return self.patch_stride

    def validate(self) -> None:
        """驗證配置參數"""
        if self.context_length <= 0:
            raise ValueError(f"context_length 必須大於 0, 得到 {self.context_length}")
        if self.prediction_length <= 0:
            raise ValueError(f"prediction_length 必須大於 0, 得到 {self.prediction_length}")
        if self.patch_length <= 0:
            raise ValueError(f"patch_length 必須大於 0, 得到 {self.patch_length}")
        if self.patch_stride <= 0:
            raise ValueError(f"patch_stride 必須大於 0, 得到 {self.patch_stride}")
        if self.patch_length > self.context_length:
            raise ValueError(
                f"patch_length ({self.patch_length}) 不能大於 "
                f"context_length ({self.context_length})"
            )
        if self.d_model <= 0:
            raise ValueError(f"d_model 必須大於 0, 得到 {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads 必須大於 0, 得到 {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) 必須能被 n_heads ({self.n_heads}) 整除"
            )
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout 必須在 [0, 1] 範圍內, 得到 {self.dropout}")

    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'patch_length': self.patch_length,
            'patch_stride': self.patch_stride,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'activation': self.activation,
            'scaling': self.scaling,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
        }


@dataclass
class TrainingConfig:
    """
    訓練配置

    統一管理不同實作的訓練參數
    """

    # 通用參數
    validation_split: float = 0.2
    random_state: int = 42

    # 深度學習參數
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 早停參數
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001

    # 學習率調度
    lr_scheduler: Literal['cosine', 'linear', 'constant'] = 'cosine'

    # 其他
    gradient_clip_val: Optional[float] = 1.0
    accumulate_grad_batches: int = 1
