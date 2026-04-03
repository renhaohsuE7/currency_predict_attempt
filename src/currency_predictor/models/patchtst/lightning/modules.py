"""
PatchTST PyTorch Lightning 核心模組

實作 PatchTST 的核心組件:
- PatchEmbedding: Patch 嵌入層
- TransformerEncoderLayer: Transformer 編碼器層
- PatchTSTModel: 完整的 PatchTST 模型
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from ..config import PatchTSTConfig


class PositionalEncoding(nn.Module):
    """
    位置編碼

    使用可學習的位置嵌入而非固定的正弦位置編碼
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 可學習的位置嵌入
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch 嵌入層

    將時間序列分割成 patches 並投影到隱藏空間

    Args:
        config: PatchTST 配置
        num_features: 輸入特徵數量 (通道數)
    """

    def __init__(self, config: PatchTSTConfig, num_features: int = 1):
        super().__init__()
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride
        self.d_model = config.d_model
        self.num_features = num_features

        # 計算 patch 數量
        self.num_patches = config.num_patches

        # Patch 線性投影
        # 每個 patch 從 (patch_length,) 投影到 (d_model,)
        self.projection = nn.Linear(self.patch_length, self.d_model)

        # 位置嵌入
        self.position_encoding = PositionalEncoding(
            self.d_model,
            max_len=self.num_patches,
            dropout=config.dropout
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, context_length, num_features)

        Returns:
            patches: (batch, num_features, num_patches, d_model)
        """
        batch_size, seq_len, n_features = x.shape

        # 使用 unfold 創建 patches
        # x: (batch, context_length, num_features)
        # 先轉置使 unfold 作用在時間維度
        x = x.permute(0, 2, 1)  # (batch, num_features, context_length)

        # 創建 patches
        patches = x.unfold(
            dimension=2,
            size=self.patch_length,
            step=self.patch_stride
        )  # (batch, num_features, num_patches, patch_length)

        # 投影到隱藏空間
        patches = self.projection(patches)  # (batch, num_features, num_patches, d_model)

        # 對每個特徵通道應用位置編碼
        # 重塑以應用位置編碼
        b, c, n, d = patches.shape
        patches = patches.reshape(b * c, n, d)  # (batch * num_features, num_patches, d_model)
        patches = self.position_encoding(patches)
        patches = self.norm(patches)
        patches = patches.reshape(b, c, n, d)  # (batch, num_features, num_patches, d_model)

        return patches


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 編碼器層

    包含:
    - 多頭自注意力
    - 前饋網路
    - 殘差連接和層正規化
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads

        # 多頭自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # 前饋網路
        if config.activation == 'gelu':
            activation = nn.GELU()
        else:
            activation = nn.ReLU()

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            activation,
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

        # 層正規化
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: 可選的注意力遮罩

        Returns:
            (batch, seq_len, d_model)
        """
        # 自注意力 (Pre-LN)
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attention(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(attn_output)

        # 前饋網路 (Pre-LN)
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + ffn_output

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer 編碼器

    堆疊多個 TransformerEncoderLayer
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.n_layers)
        ])

        # 最終層正規化
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: 可選的注意力遮罩

        Returns:
            (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.final_norm(x)
        return x


class PredictionHead(nn.Module):
    """
    預測頭

    將編碼器輸出映射到預測值
    """

    def __init__(
        self,
        config: PatchTSTConfig,
        num_features: int = 1
    ):
        super().__init__()

        self.num_patches = config.num_patches
        self.d_model = config.d_model
        self.prediction_length = config.prediction_length
        self.num_features = num_features

        # 展平後的輸入維度
        flatten_dim = self.num_patches * self.d_model

        # 預測投影
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),  # (batch, num_patches * d_model)
            nn.Linear(flatten_dim, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, self.prediction_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_patches, d_model)

        Returns:
            predictions: (batch, prediction_length)
        """
        return self.head(x)


class PatchTSTModel(nn.Module):
    """
    完整的 PatchTST 模型

    組合:
    - Patch 嵌入層
    - Transformer 編碼器
    - 預測頭

    支援通道獨立 (Channel Independence) 模式
    """

    def __init__(
        self,
        config: PatchTSTConfig,
        num_features: int = 1
    ):
        super().__init__()

        self.config = config
        self.num_features = num_features
        self.context_length = config.context_length
        self.prediction_length = config.prediction_length

        # Patch 嵌入
        self.patch_embedding = PatchEmbedding(config, num_features)

        # Transformer 編碼器
        self.encoder = TransformerEncoder(config)

        # 預測頭 (每個特徵通道一個)
        self.prediction_heads = nn.ModuleList([
            PredictionHead(config, 1)
            for _ in range(num_features)
        ])

        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        """初始化模型權重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, context_length, num_features)

        Returns:
            predictions: (batch, prediction_length, num_features)
        """
        batch_size = x.shape[0]

        # Patch 嵌入
        # patches: (batch, num_features, num_patches, d_model)
        patches = self.patch_embedding(x)

        # 對每個特徵通道獨立處理 (Channel Independence)
        outputs = []
        for i in range(self.num_features):
            # 取出單一通道
            channel_patches = patches[:, i, :, :]  # (batch, num_patches, d_model)

            # 通過編碼器
            encoded = self.encoder(channel_patches)  # (batch, num_patches, d_model)

            # 通過預測頭
            pred = self.prediction_heads[i](encoded)  # (batch, prediction_length)
            outputs.append(pred)

        # 堆疊所有通道的預測
        predictions = torch.stack(outputs, dim=-1)  # (batch, prediction_length, num_features)

        return predictions

    def get_num_parameters(self) -> int:
        """返回模型參數數量"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self) -> int:
        """返回可訓練參數數量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
