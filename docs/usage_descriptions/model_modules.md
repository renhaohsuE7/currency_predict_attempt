# 模型模組使用說明

本文檔說明如何使用模型相關模組進行貨幣預測模型的訓練和預測。

## 目錄

- [模型架構概覽](#模型架構概覽)
- [PatchTST 模型](#patchtst-模型)
  - [sklearn 版本](#sklearn-版本-patchtstsklearn)
  - [HuggingFace 版本](#huggingface-版本-patchtsthuggingface)
  - [Lightning 版本](#lightning-版本-patchtstlightningwrapper)
- [模型工廠 (ModelFactory)](#模型工廠-modelfactory)
- [基礎模型介面](#基礎模型介面)

---

## 模型架構概覽

本專案提供多種 PatchTST 實作，按照不同的後端框架分類：

```
src/currency_predictor/models/
├── base.py                          # 基礎抽象類別
├── factory.py                       # 模型工廠
│
└── patchtst/                        # PatchTST 模組
    ├── config.py                    # 統一配置類別
    ├── sklearn/                     # sklearn 版本
    │   └── model.py                 # PatchTSTSklearn
    ├── huggingface/                 # HuggingFace 版本
    │   └── model.py                 # PatchTSTHuggingFace
    └── lightning/                   # PyTorch Lightning 版本
```

### 可用模型比較

| 模型名稱 | 類別名稱 | 框架 | 特點 | 適用場景 |
|----------|----------|------|------|----------|
| `patchtst_sklearn` | `PatchTSTSklearn` | sklearn | 快速、輕量 | 原型開發、CPU 環境 |
| `patchtst_huggingface` | `PatchTSTHuggingFace` | HuggingFace | 完整 Transformer | 生產環境、GPU 加速 |
| `patchtst_lightning` | `PatchTSTLightningWrapper` | PyTorch Lightning | 靈活、可擴展 | 研究、自定義訓練（需 `uv sync --extra lightning`） |

---

## PatchTST 模型

### 統一配置 (PatchTSTConfig)

所有 PatchTST 版本共用相同的配置類別：

```python
from currency_predictor.models.patchtst import PatchTSTConfig

# 創建配置
config = PatchTSTConfig(
    context_length=64,      # 輸入序列長度 (舊名: seq_len)
    prediction_length=7,    # 預測長度 (舊名: pred_len)
    patch_length=8,         # Patch 大小
    patch_stride=4,         # Patch 步長
    d_model=64,             # Transformer 隱藏層維度
    n_heads=4,              # 注意力頭數
    n_layers=2,             # Transformer 層數
)

# 預訓練模型配置
config = PatchTSTConfig(
    pretrained_model_name_or_path="ibm-granite/granite-timeseries-patchtst",
    fine_tune_mode="full",          # 'from_scratch' | 'full' | 'linear_probe'
    prediction_length=7,
)
print(config.is_pretrained)  # True

# 或使用 sklearn 風格參數
config = PatchTSTConfig.from_sklearn_params(
    seq_len=64,
    pred_len=7,
    patch_len=8,
    stride=4
)
```

#### TrainingConfig — Fine-Tune 模式預設值

`TrainingConfig.for_fine_tune_mode()` 根據模式回傳推薦訓練超參數：

```python
from currency_predictor.models.patchtst.config import TrainingConfig

tc = TrainingConfig.for_fine_tune_mode("full")
# tc.num_epochs=20, tc.learning_rate=1e-5, tc.early_stopping_patience=5
```

| Mode | num_epochs | learning_rate | early_stopping_patience |
|------|-----------|---------------|------------------------|
| `from_scratch` | 50 | 1e-4 | 10 |
| `full` | 20 | 1e-5 | 5 |
| `linear_probe` | 10 | 1e-3 | 5 |

### sklearn 版本 (PatchTSTSklearn)

基於 sklearn GradientBoosting 的簡化版本，使用 patching 和統計特徵提取。

#### 導入

```python
# 推薦方式
from src.currency_predictor.models import PatchTSTSklearn

# 或使用向後兼容的別名
from src.currency_predictor.models import PatchTST  # 等同於 PatchTSTSklearn
```

#### 基本用法

```python
from src.currency_predictor.models import PatchTSTSklearn

# 創建模型 (sklearn 風格參數)
model = PatchTSTSklearn(
    seq_len=64,           # 輸入序列長度
    pred_len=7,           # 預測長度
    patch_len=8,          # Patch 大小
    stride=4,             # Patch 步長
    n_estimators=100,     # GradientBoosting 估計器數量
    max_depth=10,         # 決策樹最大深度
    random_state=42
)

# 訓練
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)

# 帶不確定性的預測
result = model.predict_with_uncertainty(X_test, confidence_level=0.95)
print(f"預測值: {result['predictions']}")
print(f"下界: {result['lower_bound']}")
print(f"上界: {result['upper_bound']}")
```

#### 使用配置物件

```python
from src.currency_predictor.models import PatchTSTSklearn, PatchTSTConfig

config = PatchTSTConfig(
    context_length=64,
    prediction_length=7,
    n_estimators=150,
    max_depth=12
)

model = PatchTSTSklearn(config=config)
```

### HuggingFace 版本 (PatchTSTHuggingFace)

基於 HuggingFace Transformers 的完整 PatchTST 實作，支援從頭訓練或載入預訓練模型 fine-tune。

#### 導入

```python
# 推薦方式
from currency_predictor.models import PatchTSTHuggingFace

# 或使用向後兼容的別名
from currency_predictor.models import PatchTSTTransformer
```

#### 從頭訓練（預設，向後相容）

```python
from currency_predictor.models import PatchTSTHuggingFace

model = PatchTSTHuggingFace(
    context_length=64,        # 輸入序列長度
    prediction_length=7,      # 預測長度
    d_model=64,               # Transformer 隱藏層維度
    num_attention_heads=4,    # 注意力頭數
    num_hidden_layers=2,      # Transformer 層數
    dropout=0.1               # Dropout 率
)

model.fit(X_train, y_train, num_epochs=50, learning_rate=1e-4)
predictions = model.predict(X_test)
```

#### 預訓練模型 + Fine-Tune

從 HuggingFace Hub 載入預訓練模型並 fine-tune。架構參數（`d_model`, `n_heads`, `n_layers`）由預訓練模型決定，用戶只需指定任務參數。

```python
# Full fine-tune — 所有參數可訓練，較低 learning rate
model = PatchTSTHuggingFace(
    pretrained_model_name_or_path="ibm-granite/granite-timeseries-patchtst",
    fine_tune_mode="full",
    prediction_length=7,
)
model.fit(X_train, y_train)  # 預設 num_epochs=20, lr=1e-5

# Linear probe — 凍結 backbone，只訓練 prediction head
model = PatchTSTHuggingFace(
    pretrained_model_name_or_path="ibm-granite/granite-timeseries-patchtst",
    fine_tune_mode="linear_probe",
    prediction_length=7,
)
model.fit(X_train, y_train)  # 預設 num_epochs=10, lr=1e-3
```

使用者明確傳入的訓練參數優先於模式預設值：

```python
model.fit(X_train, y_train, num_epochs=5, learning_rate=2e-5)  # 覆蓋預設
```

#### 支援的預訓練模型

| Model | Parameters | 說明 |
| --- | --- | --- |
| [ibm-granite/granite-timeseries-patchtst](https://huggingface.co/ibm-granite/granite-timeseries-patchtst) | 616K | 基礎版，ETTh1 dataset，context=512 |
| [ibm-granite/granite-timeseries-patchtst-fm-r1](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r1) | ~260M | Foundation model，context=8192 |

> **注意**: IBM Granite 預設 `context_length=512`，需要 >=519 筆資料。使用 pretrained 時需確保資料量足夠。

#### 不確定性預測 (Monte Carlo Dropout)

透過多次 forward pass (dropout enabled) 估算預測不確定性：

```python
result = model.predict_with_uncertainty(X_test, confidence_level=0.95)
# result keys: predictions, std, lower_bound, upper_bound, confidence_level
```

#### GPU 加速

HuggingFace 版本自動檢測並使用 GPU：

```python
model = PatchTSTHuggingFace(context_length=64, prediction_length=7)
print(f"使用設備: {model.device}")  # cuda 或 cpu
```

#### config.json 設定

也可透過 `config.json` 的 `model_params` 指定預訓練參數：

```json
{
    "model_name": "patchtst_transformer",
    "model_params": {
        "pretrained_model_name_or_path": "ibm-granite/granite-timeseries-patchtst",
        "fine_tune_mode": "full",
        "pred_len": 7
    }
}
```

`ModelFactory` 透過 `**kwargs` 傳遞，不需額外改動。

### Lightning 版本 (PatchTSTLightningWrapper)

基於 PyTorch Lightning 的實作，提供更靈活的訓練控制。需安裝 optional dependency：`uv sync --extra lightning`。

#### 導入

```python
from currency_predictor.models import PatchTSTLightningWrapper
```

#### 基本用法

```python
from currency_predictor.models import PatchTSTLightningWrapper

model = PatchTSTLightningWrapper(
    context_length=64,
    prediction_length=7,
    d_model=64,
    n_heads=4,
    n_layers=2,
    max_epochs=50,
    accelerator='auto',   # 'cpu', 'gpu', 'auto'
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 帶不確定性的預測 (Monte Carlo Dropout)
result = model.predict_with_uncertainty(X_test, confidence_level=0.95)
```

#### CLI 使用

```bash
uv run main.py --models lightning --symbols USDTWD=X
```

---

## 模型工廠 (ModelFactory)

### 概述

`ModelFactory` 提供統一的模型創建介面，支援多種模型類型。

### 導入

```python
from src.currency_predictor.models import ModelFactory, create_patchtst_model
```

### 使用工廠創建模型

```python
from src.currency_predictor.models import ModelFactory

# 創建 sklearn 版本
sklearn_model = ModelFactory.create_model('patchtst_sklearn', seq_len=64, pred_len=7)

# 創建 HuggingFace 版本
hf_model = ModelFactory.create_model('patchtst_huggingface', seq_len=64, pred_len=7)

# 使用舊名稱 (向後兼容)
model = ModelFactory.create_model('patchtst_transformer', seq_len=64, pred_len=7)
```

### 便捷函數

```python
from src.currency_predictor.models import create_patchtst_model

# 自動選擇最佳版本
model = create_patchtst_model(seq_len=64, pred_len=7)

# 指定實作類型
model = create_patchtst_model(implementation='sklearn', seq_len=64, pred_len=7)
model = create_patchtst_model(implementation='huggingface', seq_len=64, pred_len=7)

# 向後兼容的 use_transformer 參數
model = create_patchtst_model(use_transformer=True, seq_len=64, pred_len=7)
```

### 支援的模型類型

| 模型名稱 | 實作 | 說明 |
|----------|------|------|
| `patchtst_sklearn` | sklearn | sklearn 版本 |
| `patchtst_huggingface` | huggingface | HuggingFace 版本 |
| `patchtst_transformer` | huggingface | HuggingFace 版本 (別名) |
| `patchtst_lightning` | lightning | Lightning 版本（需 `--extra lightning`） |

### 查看可用模型

```python
from src.currency_predictor.models import ModelFactory

# 列印所有可用模型
ModelFactory.print_model_info()

# 取得推薦模型
recommended = ModelFactory.get_recommended_model(prefer_accuracy=True)
print(f"推薦模型: {recommended}")
```

---

## 基礎模型介面

### 概述

所有模型都繼承自 `BaseModel` 抽象基類，提供統一的介面。

### 類別階層

```
BaseModel (ABC)
├── TimeSeriesModel
│   ├── SklearnBasedModel
│   │   └── PatchTSTSklearn
│   └── TransformerBasedModel
│       ├── PatchTSTHuggingFace
│       └── PatchTSTLightningWrapper (optional)
```

### 必須實現的方法

```python
from src.currency_predictor.models.base import BaseModel
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, validation_data=None, **kwargs):
        """訓練模型"""
        pass

    @abstractmethod
    def predict(self, X, horizon=1, **kwargs):
        """進行預測"""
        pass

    @abstractmethod
    def predict_with_uncertainty(self, X, horizon=1, confidence_level=0.95):
        """帶不確定性的預測"""
        pass
```

### 創建自定義模型

```python
from src.currency_predictor.models.base import SklearnBasedModel

class MyCustomModel(SklearnBasedModel):
    def __init__(self, **kwargs):
        super().__init__("MyCustomModel")

    def fit(self, X, y, validation_data=None, **kwargs):
        # 實作訓練邏輯
        self.is_fitted = True
        return self

    def predict(self, X, horizon=1, **kwargs):
        # 實作預測邏輯
        return predictions

    def predict_with_uncertainty(self, X, horizon=1, confidence_level=0.95):
        # 實作不確定性預測
        return {'predictions': ..., 'std': ..., 'lower_bound': ..., 'upper_bound': ...}
```

---

## 完整範例

### 使用 sklearn 版本

```python
from src.currency_predictor.data.storage import DataStorage
from src.currency_predictor.data_processor import DataProcessor
from src.currency_predictor.models import PatchTSTSklearn

# 1. 載入和處理資料
storage = DataStorage()
processor = DataProcessor()

raw_data = storage.load_raw_data('USDTWD', '1y')
cleaned = processor.clean_data(raw_data)
with_indicators = processor.create_technical_indicators(cleaned)

# 2. 準備訓練資料
X = with_indicators.drop('Close', axis=1)
y = with_indicators['Close']

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 3. 創建並訓練模型
model = PatchTSTSklearn(seq_len=64, pred_len=7)
model.fit(X_train, y_train)

# 4. 預測
predictions = model.predict(X_test)
print(f"預測結果: {predictions}")

# 5. 評估
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.4f}")

# 6. 儲存模型
model.save_model('models/patchtst_sklearn.joblib')
```

### 使用 HuggingFace 版本（從頭訓練）

```python
from currency_predictor.models import PatchTSTHuggingFace

model = PatchTSTHuggingFace(
    context_length=64, prediction_length=7,
    d_model=64, num_attention_heads=4, num_hidden_layers=2,
)

model.fit(X_train, y_train, num_epochs=50, learning_rate=1e-4)
result = model.predict_with_uncertainty(X_test, confidence_level=0.95)
model.save_model('models/patchtst_huggingface/')
```

### 使用 HuggingFace 版本（預訓練 + Fine-Tune）

```python
from currency_predictor.models import PatchTSTHuggingFace

model = PatchTSTHuggingFace(
    pretrained_model_name_or_path="ibm-granite/granite-timeseries-patchtst",
    fine_tune_mode="full",
    prediction_length=7,
)

model.fit(X_train, y_train)  # 使用 full 模式預設值
predictions = model.predict(X_test)

# 模型資訊
info = model.get_model_info()
print(info["pretrained_model_name_or_path"])  # ibm-granite/...
print(info["fine_tune_mode"])                 # full
print(info["trainable_parameters"])           # 所有參數數量

# 儲存 — metadata 包含 pretrained 資訊
model.save_model('models/patchtst_pretrained/')
```

---

## 向後兼容性

為確保向後兼容，提供以下別名：

| 舊名稱 | 新名稱 | 說明 |
|--------|--------|------|
| `PatchTST` | `PatchTSTSklearn` | sklearn 版本別名 |
| `PatchTSTTransformer` | `PatchTSTHuggingFace` | HuggingFace 版本別名 |
| `patchtst_transformer` | `patchtst_huggingface` | 模型工廠名稱別名 |

舊的導入方式仍然有效：

```python
# 這些都可以正常工作
from src.currency_predictor.models import PatchTST
from src.currency_predictor.models import PatchTSTTransformer
model = ModelFactory.create_model('patchtst_transformer', ...)
```

---

## 相關文檔

- [資料模組使用說明](./data_modules.md)
- [配置管理說明](./config_management.md)
- [系統架構總覽](../architectures/system_architecture.md)
- [PatchTST Pretrained + Fine-Tune 計畫](../plans/2026-04-03-0050-patchtst-pretrained-finetune.md)
- [架構分析報告](../development/architecture_analysis_report.md)
- [PatchTST 實作路線圖](../development/patchtst_implementation_roadmap.md)
