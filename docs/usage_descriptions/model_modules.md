# 模型模組使用說明

本文檔說明如何使用模型相關模組進行貨幣預測模型的訓練和預測。

## 目錄

- [模型架構概覽](#模型架構概覽)
- [PatchTST 模型](#patchtst-模型)
  - [sklearn 版本](#sklearn-版本-patchtstsklearn)
  - [HuggingFace 版本](#huggingface-版本-patchtsthuggingface)
  - [Lightning 版本](#lightning-版本-patchtstlightning-開發中)
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
    └── lightning/                   # PyTorch Lightning 版本 (開發中)
```

### 可用模型比較

| 模型名稱 | 類別名稱 | 框架 | 特點 | 適用場景 |
|----------|----------|------|------|----------|
| `patchtst_sklearn` | `PatchTSTSklearn` | sklearn | 快速、輕量 | 原型開發、CPU 環境 |
| `patchtst_huggingface` | `PatchTSTHuggingFace` | HuggingFace | 完整 Transformer | 生產環境、GPU 加速 |
| `patchtst_lightning` | `PatchTSTLightning` | PyTorch Lightning | 靈活、可擴展 | 研究、自定義訓練 (開發中) |

---

## PatchTST 模型

### 統一配置 (PatchTSTConfig)

所有 PatchTST 版本共用相同的配置類別：

```python
from src.currency_predictor.models.patchtst import PatchTSTConfig

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

# 或使用 sklearn 風格參數
config = PatchTSTConfig.from_sklearn_params(
    seq_len=64,
    pred_len=7,
    patch_len=8,
    stride=4
)
```

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

基於 HuggingFace Transformers 的完整 PatchTST 實作，使用真正的 Transformer 架構。

#### 導入

```python
# 推薦方式
from src.currency_predictor.models import PatchTSTHuggingFace

# 或使用向後兼容的別名
from src.currency_predictor.models import PatchTSTTransformer
```

#### 基本用法

```python
from src.currency_predictor.models import PatchTSTHuggingFace

# 創建模型
model = PatchTSTHuggingFace(
    seq_len=64,               # 輸入序列長度
    pred_len=7,               # 預測長度
    d_model=64,               # Transformer 隱藏層維度
    num_attention_heads=4,    # 注意力頭數
    num_hidden_layers=2,      # Transformer 層數
    dropout=0.1               # Dropout 率
)

# 訓練 (支援更多參數)
model.fit(
    X_train, y_train,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    early_stopping_patience=10
)

# 預測
predictions = model.predict(X_test)

# 帶不確定性的預測 (使用 Monte Carlo sampling)
result = model.predict_with_uncertainty(X_test, confidence_level=0.95)
```

#### GPU 加速

HuggingFace 版本自動檢測並使用 GPU：

```python
model = PatchTSTHuggingFace(seq_len=64, pred_len=7)
print(f"使用設備: {model.device}")  # cuda 或 cpu
```

### Lightning 版本 (PatchTSTLightning) - 開發中

基於 PyTorch Lightning 的實作，提供更靈活的訓練控制。

```python
# 開發中，尚未可用
# from src.currency_predictor.models.patchtst.lightning import PatchTSTLightning
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
| `patchtst_lightning` | lightning | Lightning 版本 (開發中) |

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
│       └── PatchTSTHuggingFace
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

### 使用 HuggingFace 版本

```python
from src.currency_predictor.models import PatchTSTHuggingFace

# 創建模型
model = PatchTSTHuggingFace(
    seq_len=64,
    pred_len=7,
    d_model=64,
    num_attention_heads=4,
    num_hidden_layers=2
)

# 訓練 (會自動使用 GPU)
model.fit(
    X_train, y_train,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    early_stopping_patience=10
)

# 帶不確定性的預測
result = model.predict_with_uncertainty(X_test, confidence_level=0.95)

# 儲存模型 (目錄格式)
model.save_model('models/patchtst_huggingface/')
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
- [架構分析報告](../development/architecture_analysis_report.md)
- [PatchTST 實作路線圖](../development/patchtst_implementation_roadmap.md)
