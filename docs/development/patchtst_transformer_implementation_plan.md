# PatchTST Transformer 實作計畫

## 文件資訊

- **版本**: 1.0
- **建立日期**: 2026-01-25
- **目標**: 整合 HuggingFace Transformers 的 PatchTST 模型進行貨幣預測

---

## 1. 現況分析

### 1.1 現有架構

```
src/currency_predictor/models/
├── base.py                  # 基礎模型介面 (BaseModel, TransformerBasedModel)
├── factory.py               # 模型工廠 (支援 patchtst_sklearn, patchtst_transformer)
├── patchtst.py              # sklearn 版本 (RandomForest-based, 目前預設)
├── patchtst_transformer.py  # Transformer 版本 (已有框架，需完善)
└── __init__.py
```

### 1.2 現有問題

| 問題 | 描述 | 影響 |
|------|------|------|
| 資料格式不匹配 | `PatchTSTForPrediction` 期望 `(batch, seq_len, channels)` 格式 | 預測失敗 |
| 訓練資料不足 | 目前 260 筆資料不足以有效訓練 Transformer | 模型效能差 |
| 特徵處理問題 | 現有 DataProcessor 產生過多 NaN | 訓練資料損失 |
| 缺乏 TimeSeriesPreprocessor | HuggingFace 建議使用 `tsfm_public.toolkit` | 資料準備不完整 |

### 1.3 依賴狀態

```toml
# pyproject.toml 已包含
transformers>=4.56.2
torch>=2.8.0
accelerate>=1.10.1
datasets>=4.1.1
```

---

## 2. HuggingFace PatchTST 概覽

### 2.1 模型架構

```
輸入時間序列 → Patch 分割 → Transformer Encoder → 預測頭 → 輸出預測
     ↓              ↓              ↓              ↓
(batch, seq_len, channels) → (batch, n_patches, d_model) → (batch, pred_len, channels)
```

### 2.2 關鍵配置參數

| 參數 | 預設值 | 描述 | 建議值 (貨幣預測) |
|------|--------|------|-------------------|
| `context_length` | 32 | 輸入序列長度 | 64-256 |
| `prediction_length` | 24 | 預測長度 | 7-30 |
| `patch_length` | 1 | Patch 大小 | 8-16 |
| `patch_stride` | 1 | Patch 步長 | patch_length/2 |
| `num_input_channels` | 1 | 輸入通道數 | 1 (單變量: Close) 或 5+ (多變量) |
| `d_model` | 128 | Transformer 維度 | 64-256 |
| `num_attention_heads` | 4 | 注意力頭數 | 4-8 |
| `num_hidden_layers` | 3 | Encoder 層數 | 2-6 |
| `ffn_dim` | 512 | FFN 維度 | 256-512 |
| `dropout` | 0.0 | Dropout 率 | 0.1-0.3 |

### 2.3 輸入資料格式

```python
# past_values: 過去時間序列
shape: (batch_size, context_length, num_input_channels)
dtype: torch.float32

# future_values: 未來目標值 (訓練時)
shape: (batch_size, prediction_length, num_input_channels)
dtype: torch.float32

# past_observed_mask: 觀測遮罩 (可選)
shape: (batch_size, context_length, num_input_channels)
dtype: torch.bool
```

### 2.4 模型變體

| 類別 | 用途 | 輸出 |
|------|------|------|
| `PatchTSTModel` | 特徵提取 | 隱藏狀態 |
| `PatchTSTForPrediction` | **時間序列預測** | 預測值 |
| `PatchTSTForClassification` | 分類任務 | 類別 logits |
| `PatchTSTForRegression` | 回歸任務 | 回歸值 |
| `PatchTSTForPretraining` | 預訓練 | 重建損失 |

---

## 3. 最小可行方案 (MVP)

### 3.1 MVP 目標

1. **基本功能**: 能夠使用真正的 PatchTST 進行貨幣價格預測
2. **資料兼容**: 與現有 DataProcessor 輸出兼容
3. **API 一致**: 維持 `fit()`, `predict()`, `predict_with_uncertainty()` 介面
4. **錯誤處理**: 優雅降級到 sklearn 版本

### 3.2 MVP 範圍

#### 包含 (In Scope)
- [x] 單變量預測 (Close 價格)
- [x] 基本訓練流程
- [x] 模型儲存/載入
- [x] 不確定性估計
- [ ] 與現有 pipeline 整合

#### 不包含 (Out of Scope)
- 多變量預測
- 遷移學習/預訓練
- 分類/回歸任務
- 分散式訓練

### 3.3 MVP 資料需求

```
最小訓練資料量: 500+ 筆 (建議 1000+)
資料格式: 日頻或更高頻率
必要欄位: Date, Close
可選欄位: Open, High, Low, Volume
```

---

## 4. 實作計畫

### 4.1 階段一: 資料準備層 (Phase 1)

**目標**: 建立符合 PatchTST 輸入格式的資料處理管道

#### 4.1.1 新增 TimeSeriesDataModule

```python
# src/currency_predictor/data/time_series_dataset.py

class CurrencyTimeSeriesDataset(Dataset):
    """貨幣時間序列資料集"""

    def __init__(
        self,
        data: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        target_column: str = 'Close',
        feature_columns: List[str] = None,
        scaling: bool = True
    ):
        ...

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'past_values': past_values,      # (context_length, num_channels)
            'future_values': future_values,  # (prediction_length, num_channels)
        }
```

#### 4.1.2 修改 DataProcessor

```python
# 新增方法
def prepare_for_transformer(
    self,
    df: pd.DataFrame,
    context_length: int = 64,
    prediction_length: int = 7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """準備 Transformer 格式資料"""
    ...
```

### 4.2 階段二: 模型層改進 (Phase 2)

**目標**: 完善 PatchTSTTransformer 實作

#### 4.2.1 修改 patchtst_transformer.py

**關鍵修改點**:

```python
class PatchTSTTransformer(TransformerBasedModel):

    def __init__(
        self,
        context_length: int = 64,       # 調整預設值
        prediction_length: int = 7,
        patch_length: int = 8,          # 調整預設值
        stride: int = 4,
        num_input_channels: int = 1,    # 新增參數
        d_model: int = 64,              # 減小以適應小資料集
        num_attention_heads: int = 4,
        num_hidden_layers: int = 2,     # 減少層數
        **kwargs
    ):
        ...

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None,  # 改為 tuple
        **kwargs
    ) -> 'PatchTSTTransformer':
        """訓練模型"""
        # 1. 資料驗證
        self._validate_data(X)

        # 2. 準備資料集
        train_dataset, val_dataset = self._prepare_datasets(X, validation_data)

        # 3. 配置訓練器
        trainer = self._setup_trainer(train_dataset, val_dataset, **kwargs)

        # 4. 訓練
        trainer.train()

        return self
```

#### 4.2.2 資料格式轉換

```python
def _prepare_datasets(
    self,
    X: pd.DataFrame,
    validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
) -> Tuple[Dataset, Optional[Dataset]]:
    """準備訓練和驗證資料集"""

    # 選擇目標欄位
    if 'Close' in X.columns:
        values = X['Close'].values.reshape(-1, 1)
    else:
        values = X.values

    # 標準化
    self.scaler = StandardScaler()
    values_scaled = self.scaler.fit_transform(values)

    # 創建序列
    sequences, targets = self._create_sequences(
        values_scaled,
        self.context_length,
        self.prediction_length
    )

    # 轉換為 PyTorch 張量
    # Shape: (num_samples, seq_len, num_channels)
    past_values = torch.FloatTensor(sequences)
    future_values = torch.FloatTensor(targets)

    train_dataset = TimeSeriesDataset(past_values, future_values)

    # 處理驗證資料...

    return train_dataset, val_dataset
```

### 4.3 階段三: 整合與測試 (Phase 3)

**目標**: 與現有 pipeline 整合並驗證

#### 4.3.1 修改 predictor.py

```python
# src/currency_predictor/prediction/predictor.py

def _create_model(self, model_name: str, model_params: Dict) -> BaseModel:
    """創建模型實例"""

    # 參數映射
    if model_name == 'patchtst_transformer':
        # 將 seq_len 映射到 context_length
        if 'seq_len' in model_params:
            model_params['context_length'] = model_params.pop('seq_len')
        if 'pred_len' in model_params:
            model_params['prediction_length'] = model_params.pop('pred_len')

    return ModelFactory.create_model(model_name, **model_params)
```

#### 4.3.2 測試案例

```python
# tests/test_patchtst_transformer.py

class TestPatchTSTTransformer:

    def test_model_initialization(self):
        """測試模型初始化"""
        model = PatchTSTTransformer(
            context_length=64,
            prediction_length=7
        )
        assert model is not None

    def test_data_preparation(self):
        """測試資料準備"""
        # 生成測試資料
        df = pd.DataFrame({
            'Close': np.random.randn(500).cumsum() + 30
        })

        model = PatchTSTTransformer()
        data_dict = model.prepare_data_for_transformer(df)

        assert 'past_values' in data_dict
        assert data_dict['past_values'].shape[1] == model.context_length

    def test_training(self):
        """測試訓練流程"""
        df = pd.DataFrame({
            'Close': np.random.randn(1000).cumsum() + 30
        })

        model = PatchTSTTransformer(
            context_length=32,
            prediction_length=7,
            num_hidden_layers=1  # 減少層數加快測試
        )

        model.fit(df, num_epochs=2, batch_size=16)

        assert model.is_fitted

    def test_prediction(self):
        """測試預測流程"""
        # ... 完整訓練後測試預測
```

---

## 5. 資料流程圖

### 5.1 訓練流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        訓練資料流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Yahoo Finance API                                              │
│       ↓                                                         │
│  DataStorage (CSV)                                              │
│       ↓                                                         │
│  DataProcessor.load_and_process()                               │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ DataFrame                                               │   │
│  │ columns: [Close, (Open, High, Low, Volume)]            │   │
│  │ shape: (N, 1~5)                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  PatchTSTTransformer._prepare_datasets()                        │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ TimeSeriesDataset                                       │   │
│  │ past_values: (batch, context_length, num_channels)     │   │
│  │ future_values: (batch, prediction_length, num_channels)│   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  HuggingFace Trainer                                            │
│       ↓                                                         │
│  PatchTSTForPrediction (訓練)                                   │
│       ↓                                                         │
│  模型儲存 (save_pretrained)                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 預測流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        預測資料流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  歷史資料 (最近 context_length 筆)                              │
│       ↓                                                         │
│  標準化 (使用訓練時的 scaler)                                   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ past_values: (1, context_length, num_channels)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  PatchTSTForPrediction.forward(past_values)                     │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ outputs.prediction_outputs                              │   │
│  │ shape: (1, num_samples, prediction_length, num_channels)│   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  反標準化 (inverse_transform)                                   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ predictions: (prediction_length,)                       │   │
│  │ uncertainty: (prediction_length,)                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. API 兼容性

### 6.1 介面對照表

| sklearn 版本 | Transformer 版本 | 說明 |
|-------------|-----------------|------|
| `seq_len` | `context_length` | 輸入序列長度 |
| `pred_len` | `prediction_length` | 預測長度 |
| `patch_len` | `patch_length` | Patch 大小 |
| `stride` | `patch_stride` | Patch 步長 |
| `n_estimators` | `num_hidden_layers` | 模型複雜度 |
| `max_depth` | `d_model` | 模型容量 |

### 6.2 統一介面設計

```python
# 統一的模型配置接口
class ModelConfig:
    # 通用參數 (兩版本共用)
    seq_len: int = 168           # 會映射到 context_length
    pred_len: int = 24           # 會映射到 prediction_length

    # sklearn 專用
    n_estimators: int = 100
    max_depth: int = 10

    # transformer 專用
    d_model: int = 64
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    dropout: float = 0.1
```

---

## 7. 修改清單

### 7.1 需要修改的檔案

| 檔案 | 修改類型 | 優先級 | 說明 |
|------|---------|--------|------|
| `patchtst_transformer.py` | 重構 | P0 | 核心模型實作 |
| `data_processor.py` | 新增方法 | P0 | 增加 Transformer 資料準備 |
| `predictor.py` | 修改 | P1 | 參數映射與模型選擇 |
| `settings.py` | 新增 | P1 | Transformer 配置參數 |
| `factory.py` | 修改 | P2 | 模型工廠更新 |
| `pipeline.py` | 修改 | P2 | 整合新模型 |

### 7.2 需要新增的檔案

| 檔案 | 說明 |
|------|------|
| `src/currency_predictor/data/time_series_dataset.py` | PyTorch Dataset 類別 |
| `tests/test_patchtst_transformer.py` | 單元測試 |
| `examples/transformer_prediction.py` | 使用範例 |

---

## 8. 風險評估

### 8.1 技術風險

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| 訓練資料不足 | 高 | 高 | 增加資料收集期間至 2-5 年 |
| GPU 記憶體不足 | 中 | 中 | 減小 batch_size 和 d_model |
| 訓練時間過長 | 中 | 低 | 使用 early stopping |
| 模型過擬合 | 高 | 高 | 增加 dropout，減少層數 |

### 8.2 相容性風險

| 風險 | 說明 | 緩解措施 |
|------|------|----------|
| API 不兼容 | 現有程式碼使用 sklearn 版本參數 | 參數映射層 |
| 模型格式不同 | Transformer 使用目錄儲存 | 統一儲存介面 |
| 依賴衝突 | torch 與其他庫版本問題 | 固定版本號 |

---

## 9. 實作順序建議

### Phase 1: 基礎設施 (Week 1)

1. [ ] 建立 `TimeSeriesDataset` 類別
2. [ ] 修改 `DataProcessor` 新增 Transformer 資料準備方法
3. [ ] 撰寫基本單元測試

### Phase 2: 模型完善 (Week 1-2)

4. [ ] 重構 `PatchTSTTransformer.fit()` 方法
5. [ ] 重構 `PatchTSTTransformer.predict()` 方法
6. [ ] 實作 `predict_with_uncertainty()`
7. [ ] 完善模型儲存/載入

### Phase 3: 整合測試 (Week 2)

8. [ ] 修改 `predictor.py` 參數映射
9. [ ] 更新 `settings.py` 配置
10. [ ] 整合測試 pipeline
11. [ ] 建立使用範例

### Phase 4: 驗證與文檔 (Week 3)

12. [ ] 效能基準測試
13. [ ] 與 sklearn 版本比較
14. [ ] 撰寫使用文檔
15. [ ] 更新 README

---

## 10. 成功指標

### 10.1 功能指標

- [ ] 能夠成功完成訓練 (無錯誤)
- [ ] 能夠進行預測 (輸出正確格式)
- [ ] 模型可儲存和載入
- [ ] 與現有 pipeline 整合

### 10.2 效能指標

| 指標 | 目標 | 說明 |
|------|------|------|
| MSE | < sklearn 版本 | 預測誤差 |
| 訓練時間 | < 30 分鐘 | 1 年日頻資料 |
| 推論延遲 | < 1 秒 | 單次預測 |
| 記憶體使用 | < 4 GB | GPU/CPU |

---

## 11. 參考資源

### 官方文檔

- [HuggingFace PatchTST Documentation](https://huggingface.co/docs/transformers/model_doc/patchtst)
- [PatchTST Blog Post](https://huggingface.co/blog/patchtst)
- [IBM tsfm Toolkit](https://github.com/IBM/tsfm)

### 論文

- [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730) (ICLR 2023)

### 預訓練模型

- `namctin/patchtst_etth1_pretrain` - ETTh1 預訓練模型
- `namctin/patchtst_etth1_forecast` - ETTh1 預測模型
- `ibm/patchtst` - IBM 官方模型

---

## 附錄 A: 最小可執行程式碼範例

```python
"""
PatchTST Transformer 最小可執行範例
"""
from transformers import PatchTSTConfig, PatchTSTForPrediction
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. 準備資料
df = pd.DataFrame({
    'Close': np.random.randn(500).cumsum() + 30
})

# 2. 參數設定
context_length = 64
prediction_length = 7
num_samples = len(df) - context_length - prediction_length + 1

# 3. 創建序列
scaler = StandardScaler()
values = scaler.fit_transform(df[['Close']].values)

past_values = []
future_values = []

for i in range(num_samples):
    past_values.append(values[i:i+context_length])
    future_values.append(values[i+context_length:i+context_length+prediction_length])

past_values = torch.FloatTensor(np.array(past_values))      # (N, 64, 1)
future_values = torch.FloatTensor(np.array(future_values))  # (N, 7, 1)

# 4. 創建模型
config = PatchTSTConfig(
    num_input_channels=1,
    context_length=context_length,
    prediction_length=prediction_length,
    patch_length=8,
    patch_stride=4,
    d_model=64,
    num_attention_heads=4,
    num_hidden_layers=2,
    dropout=0.1,
)

model = PatchTSTForPrediction(config)

# 5. 訓練 (簡化版)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    outputs = model(past_values=past_values, future_values=future_values)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 6. 預測
model.eval()
with torch.no_grad():
    # 使用最後一個序列進行預測
    test_input = past_values[-1:].clone()
    outputs = model(past_values=test_input)
    prediction = outputs.prediction_outputs  # (1, num_samples, 7, 1)

    # 取平均
    mean_pred = prediction.mean(dim=1).squeeze().numpy()  # (7,)

    # 反標準化
    mean_pred_rescaled = scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()

    print(f"Predictions: {mean_pred_rescaled}")
```

---

## 附錄 B: 配置參數對照表

| 用途 | sklearn 參數 | transformer 參數 | 建議值 |
|------|-------------|-----------------|--------|
| 輸入長度 | `seq_len` | `context_length` | 64-256 |
| 預測長度 | `pred_len` | `prediction_length` | 7-30 |
| Patch 大小 | `patch_len` | `patch_length` | 8-16 |
| Patch 步長 | `stride` | `patch_stride` | 4-8 |
| 模型維度 | - | `d_model` | 64-128 |
| 注意力頭 | - | `num_attention_heads` | 4-8 |
| 層數 | - | `num_hidden_layers` | 2-4 |
| Dropout | - | `dropout` | 0.1-0.3 |
| 學習率 | - | `learning_rate` | 1e-4 |
| 批次大小 | - | `batch_size` | 16-64 |
| 訓練輪數 | - | `num_epochs` | 50-100 |
