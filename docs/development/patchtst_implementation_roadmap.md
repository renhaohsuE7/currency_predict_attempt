# PatchTST 實作路線圖

## 1. 現有實作分析

### 1.1 目前 PatchTST 實作分類

| 檔案 | 名稱 | 底層框架 | 特點 |
|------|------|----------|------|
| `patchtst.py` | `PatchTST` | NumPy + sklearn | 模擬 patching 概念，使用 GradientBoostingRegressor |
| `patchtst_transformer.py` | `PatchTSTTransformer` | PyTorch + HuggingFace | 真正的 Transformer 架構 |

### 1.2 依賴關係圖

```
patchtst.py (sklearn 版)
├── numpy                    # 數據處理
├── pandas                   # DataFrame 操作
├── sklearn.ensemble         # GradientBoostingRegressor
├── sklearn.preprocessing    # StandardScaler
└── sklearn.metrics          # mean_squared_error, mean_absolute_error

patchtst_transformer.py (HuggingFace 版)
├── numpy                    # 數據處理
├── pandas                   # DataFrame 操作
├── torch                    # PyTorch 張量操作
├── transformers             # PatchTSTConfig, PatchTSTForPrediction, Trainer
├── sklearn.preprocessing    # StandardScaler
└── scipy.stats              # 信賴區間計算
```

---

## 2. 問題識別

### 2.1 sklearn 版 PatchTST 問題

**問題**: 這不是真正的 PatchTST，只是一個模擬概念的實作

```python
# 目前實作 (patchtst.py)
class PatchTST(SklearnBasedModel):
    def __init__(self):
        # 使用 GradientBoostingRegressor 代替 Transformer
        self.ensemble_model = GradientBoostingRegressor(...)

    def _create_patches(self, data):
        # 創建滑動窗口 (正確)

    def _extract_patch_features(self, patches):
        # 提取統計特徵 (均值、標準差等) 代替 Transformer 的注意力機制
        # 這不是 PatchTST 的核心概念
```

**真正的 PatchTST 應該**:
1. 將時間序列分割成 patches
2. 使用 Transformer encoder 處理 patches
3. 使用自注意力機制學習 patch 間的關係

### 2.2 HuggingFace 版問題

**優點**: 使用真正的 Transformer 架構
**問題**: 依賴 HuggingFace Trainer，訓練流程不夠靈活

---

## 3. 建議的模型分類架構

### 3.1 新的模組結構

```
src/currency_predictor/models/
├── base.py                          # 基礎抽象類別
├── factory.py                       # 模型工廠
│
├── classical/                       # 傳統 ML 模型
│   ├── __init__.py
│   ├── gradient_boosting.py        # GradientBoosting 時間序列
│   └── random_forest.py            # RandomForest 時間序列
│
├── patchtst/                        # PatchTST 專用模組
│   ├── __init__.py
│   ├── config.py                   # PatchTST 配置
│   ├── patchtst_sklearn.py         # sklearn 版 (重新命名)
│   ├── patchtst_huggingface.py     # HuggingFace 版
│   └── patchtst_lightning.py       # PyTorch Lightning 版 (新增)
│
└── transformer/                     # 其他 Transformer 模型
    ├── __init__.py
    └── informer.py                 # 未來擴展
```

### 3.2 模型類別層次

```
BaseModel (ABC)
├── ClassicalModel                   # 傳統 ML 模型基類
│   ├── GradientBoostingTS          # 時間序列用的 GB
│   └── RandomForestTS              # 時間序列用的 RF
│
├── PatchTSTBase (ABC)               # PatchTST 專用基類
│   ├── PatchTSTSklearn             # sklearn 特徵工程版
│   ├── PatchTSTHuggingFace         # HuggingFace 版
│   └── PatchTSTLightning           # PyTorch Lightning 版
│
└── TransformerModel                 # 其他 Transformer 模型
    └── Informer                    # 未來擴展
```

---

## 4. PyTorch Lightning PatchTST 實作計畫

### 4.1 為什麼選擇 PyTorch Lightning?

| 特點 | HuggingFace Trainer | PyTorch Lightning |
|------|---------------------|-------------------|
| 靈活性 | 低 (黑盒) | 高 (完全控制) |
| 學習曲線 | 低 | 中 |
| 自定義訓練循環 | 困難 | 容易 |
| 分散式訓練 | 自動 | 自動 |
| 回調系統 | 有限 | 豐富 |
| 調試 | 困難 | 容易 |

### 4.2 實作架構

```python
# patchtst_lightning.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class PatchTSTConfig:
    """PatchTST 配置類"""
    context_length: int = 64
    prediction_length: int = 7
    patch_length: int = 8
    patch_stride: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = 'gelu'


class PatchEmbedding(nn.Module):
    """Patch 嵌入層"""
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.patch_length = config.patch_length
        self.stride = config.patch_stride
        self.d_model = config.d_model

        # Linear projection of patches
        self.projection = nn.Linear(config.patch_length, config.d_model)

        # Position embedding
        num_patches = (config.context_length - config.patch_length) // config.stride + 1
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, config.d_model))

    def forward(self, x):
        # x: (batch, context_length, n_features)
        batch_size, seq_len, n_features = x.shape

        # Create patches
        patches = x.unfold(1, self.patch_length, self.stride)  # (batch, n_patches, n_features, patch_len)
        patches = patches.permute(0, 2, 1, 3)  # (batch, n_features, n_patches, patch_len)

        # Project patches
        patches = self.projection(patches)  # (batch, n_features, n_patches, d_model)

        # Add position embedding
        patches = patches + self.position_embedding

        return patches  # (batch, n_features, n_patches, d_model)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class PatchTSTModel(nn.Module):
    """PatchTST 核心模型"""
    def __init__(self, config: PatchTSTConfig, n_features: int = 1):
        super().__init__()
        self.config = config
        self.n_features = n_features

        # Patch embedding
        self.patch_embedding = PatchEmbedding(config)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.n_layers)
        ])

        # Prediction head
        num_patches = (config.context_length - config.patch_length) // config.stride + 1
        self.head = nn.Linear(num_patches * config.d_model, config.prediction_length)

    def forward(self, x):
        # x: (batch, context_length, n_features)
        batch_size = x.shape[0]

        # Patch embedding
        patches = self.patch_embedding(x)  # (batch, n_features, n_patches, d_model)

        # Process each feature channel independently
        outputs = []
        for i in range(self.n_features):
            channel_patches = patches[:, i]  # (batch, n_patches, d_model)

            # Apply transformer encoder
            for layer in self.encoder_layers:
                channel_patches = layer(channel_patches)

            # Flatten and predict
            channel_patches = channel_patches.flatten(1)  # (batch, n_patches * d_model)
            channel_pred = self.head(channel_patches)  # (batch, prediction_length)
            outputs.append(channel_pred)

        # Stack outputs
        output = torch.stack(outputs, dim=-1)  # (batch, prediction_length, n_features)

        return output


class PatchTSTLightning(pl.LightningModule):
    """PyTorch Lightning 封裝的 PatchTST"""

    def __init__(
        self,
        config: PatchTSTConfig,
        n_features: int = 1,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.learning_rate = learning_rate

        # 核心模型
        self.model = PatchTSTModel(config, n_features)

        # 損失函數
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['past_values'], batch['future_values']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['past_values'], batch['future_values']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        return [optimizer], [scheduler]
```

### 4.3 與現有介面整合

```python
# patchtst/patchtst_lightning_wrapper.py

from ..base import TransformerBasedModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class PatchTSTLightningWrapper(TransformerBasedModel):
    """
    PyTorch Lightning PatchTST 的 sklearn-compatible 封裝

    提供與其他模型一致的介面:
    - fit(X, y, validation_data=None)
    - predict(X, horizon=None)
    - predict_with_uncertainty(X, horizon=None, confidence_level=0.95)
    """

    def __init__(
        self,
        # sklearn 風格參數
        seq_len: int = 64,
        pred_len: int = 7,
        patch_len: int = 8,
        stride: int = 4,
        # 模型架構參數
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        # 訓練參數
        max_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 10,
        **kwargs
    ):
        super().__init__(model_name="PatchTST_Lightning")

        # 配置
        self.config = PatchTSTConfig(
            context_length=seq_len,
            prediction_length=pred_len,
            patch_length=patch_len,
            patch_stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )

        # 訓練參數
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        # 模型和訓練器
        self.lightning_model = None
        self.trainer = None
        self.scaler = StandardScaler()

    def fit(self, X, y, validation_data=None, **kwargs):
        """訓練模型"""
        # 準備資料
        train_dataset = self._prepare_dataset(X, y, fit_scaler=True)

        val_dataset = None
        if validation_data:
            X_val, y_val = validation_data
            val_dataset = self._prepare_dataset(X_val, y_val, fit_scaler=False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size) if val_dataset else None

        # 創建 Lightning 模型
        self.lightning_model = PatchTSTLightning(
            config=self.config,
            n_features=1,
            learning_rate=self.learning_rate
        )

        # 回調
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                mode='min'
            ),
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1
            )
        ]

        # 訓練器
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            accelerator='auto',
            enable_progress_bar=True
        )

        # 訓練
        self.trainer.fit(
            self.lightning_model,
            train_loader,
            val_loader
        )

        self.is_fitted = True
        return self

    def predict(self, X, horizon=None):
        """預測"""
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")

        horizon = horizon or self.config.prediction_length
        # ... 實作預測邏輯

    def predict_with_uncertainty(self, X, horizon=None, confidence_level=0.95):
        """帶不確定性的預測"""
        # 使用 Monte Carlo Dropout 估計不確定性
        pass
```

---

## 5. 實作優先順序

### Phase 1: 重構現有程式碼 (1-2 天)

| 任務 | 說明 | 優先級 |
|------|------|--------|
| 1.1 | 重新命名 `PatchTST` → `PatchTSTSklearn` | 高 |
| 1.2 | 重新命名 `PatchTSTTransformer` → `PatchTSTHuggingFace` | 高 |
| 1.3 | 創建 `patchtst/` 子模組 | 高 |
| 1.4 | 更新 factory.py 和 __init__.py | 高 |

### Phase 2: 實作 PyTorch Lightning 版本 (3-5 天)

| 任務 | 說明 | 優先級 |
|------|------|--------|
| 2.1 | 實作 `PatchTSTConfig` | 高 |
| 2.2 | 實作 `PatchEmbedding` | 高 |
| 2.3 | 實作 `TransformerEncoderLayer` | 高 |
| 2.4 | 實作 `PatchTSTModel` | 高 |
| 2.5 | 實作 `PatchTSTLightning` | 高 |
| 2.6 | 實作 `PatchTSTLightningWrapper` | 高 |
| 2.7 | 更新 ModelFactory | 中 |

### Phase 3: 測試與驗證 (2-3 天)

| 任務 | 說明 | 優先級 |
|------|------|--------|
| 3.1 | 單元測試 | 高 |
| 3.2 | 整合測試 | 高 |
| 3.3 | 效能比較 (sklearn vs HuggingFace vs Lightning) | 中 |
| 3.4 | 文件更新 | 中 |

---

## 6. 依賴更新

### 6.1 新增依賴

```toml
# pyproject.toml

[project]
dependencies = [
    # ... 現有依賴
    "pytorch-lightning>=2.0.0",
    "torchmetrics>=1.0.0",
]
```

### 6.2 可選依賴群組

```toml
[project.optional-dependencies]
lightning = [
    "pytorch-lightning>=2.0.0",
    "torchmetrics>=1.0.0",
]

huggingface = [
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
    "datasets>=2.12.0",
]

all = [
    "currency-predict-attempt[lightning,huggingface]",
]
```

---

## 7. 預期的最終架構

### 7.1 模組結構

```
src/currency_predictor/models/
├── base.py
├── factory.py
├── __init__.py
│
└── patchtst/
    ├── __init__.py
    ├── config.py                    # 統一配置
    ├── sklearn/
    │   ├── __init__.py
    │   └── model.py                 # PatchTSTSklearn
    ├── huggingface/
    │   ├── __init__.py
    │   └── model.py                 # PatchTSTHuggingFace
    └── lightning/
        ├── __init__.py
        ├── modules.py               # PatchEmbedding, TransformerEncoder
        ├── model.py                 # PatchTSTLightning
        └── wrapper.py               # PatchTSTLightningWrapper
```

### 7.2 使用範例

```python
from currency_predictor.models import ModelFactory

# 選項 1: sklearn 版 (快速，適合原型開發)
model = ModelFactory.create_model('patchtst_sklearn', seq_len=64, pred_len=7)

# 選項 2: HuggingFace 版 (成熟，有預訓練權重)
model = ModelFactory.create_model('patchtst_huggingface', seq_len=64, pred_len=7)

# 選項 3: PyTorch Lightning 版 (靈活，適合研究和自定義)
model = ModelFactory.create_model('patchtst_lightning', seq_len=64, pred_len=7)

# 統一介面
model.fit(X_train, y_train, validation_data=(X_val, y_val))
predictions = model.predict(X_test, horizon=7)
results = model.predict_with_uncertainty(X_test, confidence_level=0.95)
```

---

## 8. 風險與考量

### 8.1 向後兼容性

- 保持現有 `patchtst_sklearn` 和 `patchtst_transformer` 名稱可用
- 使用別名映射到新的類別名稱

### 8.2 效能考量

| 版本 | 訓練速度 | 推論速度 | GPU 支援 | 準確度 |
|------|----------|----------|----------|--------|
| sklearn | 快 | 快 | 否 | 低-中 |
| HuggingFace | 中 | 中 | 是 | 高 |
| Lightning | 中 | 中-快 | 是 | 高 |

### 8.3 維護成本

- PyTorch Lightning 版需要更多程式碼維護
- 但提供更好的可測試性和可擴展性

---

## 9. 下一步行動

1. **立即**: 審核並確認此計畫
2. **Phase 1**: 開始重構現有程式碼
3. **Phase 2**: 實作 PyTorch Lightning 版本
4. **Phase 3**: 測試與文件

---

## 附錄 A: PatchTST 論文參考

**論文**: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

**核心概念**:
1. **Patching**: 將時間序列分割成固定長度的 patches
2. **Channel Independence**: 每個變量獨立處理
3. **Transformer Encoder**: 使用標準 Transformer encoder
4. **Instance Normalization**: 對每個樣本進行標準化

**關鍵公式**:

```
Number of patches = (context_length - patch_length) / stride + 1

Patch embedding: R^patch_length -> R^d_model

Self-attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```
