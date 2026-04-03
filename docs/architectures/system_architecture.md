# Currency Prediction Project Architecture

## 專案概述

本專案是一個模組化的貨幣匯率預測系統，使用 PatchTST 模型進行時間序列預測。

## 設計原則

1. **模組化設計** - 資料下載、處理、預測完全分離
2. **可擴展性** - 支援多種貨幣對和預測模型實作
3. **可重現性** - 完整的配置管理和日誌記錄
4. **測試友好** - 每個模組都可獨立測試
5. **向後兼容** - 新版本保持與舊版本 API 相容

## 當前架構

### 模組結構

```text
src/currency_predictor/
├── __init__.py               # 套件初始化
│
├── config/                   # 配置管理
│   ├── settings.py           # Pydantic 配置類別
│   └── manager.py            # 配置管理器
│
├── data/                     # 資料相關模組
│   ├── collectors.py         # 資料收集器 (Yahoo Finance)
│   ├── storage.py            # 資料儲存和載入
│   └── validators.py         # 資料驗證
│
├── models/                   # 模型相關模組
│   ├── base.py               # 基礎模型介面 (ABC)
│   ├── factory.py            # 模型工廠
│   └── patchtst/             # PatchTST 模型模組
│       ├── config.py         # 統一配置類別
│       ├── sklearn/          # sklearn 版本
│       │   └── model.py      # PatchTSTSklearn
│       ├── huggingface/      # HuggingFace 版本
│       │   └── model.py      # PatchTSTHuggingFace
│       └── lightning/        # Lightning 版本 (開發中)
│
├── prediction/               # 預測相關模組
│   ├── predictor.py          # 預測執行器
│   └── pipeline.py           # 預測管道
│
├── reporting/                # 報告模組
│   └── formatter.py          # 結果格式化
│
├── visualization/            # 視覺化模組
│   └── visualizer.py         # 圖表生成
│
├── data_processor.py         # 資料處理器
└── utils.py                  # 工具函數
```

### 類別階層

```text
BaseModel (ABC)
├── TimeSeriesModel
│   ├── SklearnBasedModel
│   │   └── PatchTSTSklearn
│   └── TransformerBasedModel
│       └── PatchTSTHuggingFace
```

## 資料流程

```text
1. 資料收集 (data/collectors.py)
   ↓
2. 資料儲存 (data/storage.py)
   ↓
3. 資料處理 (data_processor.py)
   ↓
4. 特徵工程 (data_processor.py)
   ↓
5. 模型訓練 (models/patchtst/*)
   ↓
6. 模型評估 (prediction/predictor.py)
   ↓
7. 預測執行 (prediction/pipeline.py)
   ↓
8. 結果輸出 (reporting/formatter.py)
```

## 模型系統

### 可用模型

| 模型名稱 | 類別 | 框架 | 狀態 |
| --- | --- | --- | --- |
| patchtst_sklearn | PatchTSTSklearn | sklearn | 可用 |
| patchtst_huggingface | PatchTSTHuggingFace | HuggingFace | 可用 |
| patchtst_transformer | PatchTSTHuggingFace | HuggingFace | 可用 (別名) |
| patchtst_lightning | PatchTSTLightning | PyTorch Lightning | 開發中 |

### 模型工廠

```python
from src.currency_predictor.models import ModelFactory

# 創建模型
model = ModelFactory.create_model('patchtst_sklearn', seq_len=64, pred_len=7)
model = ModelFactory.create_model('patchtst_huggingface', seq_len=64, pred_len=7)

# 取得推薦模型
recommended = ModelFactory.get_recommended_model(prefer_accuracy=True)
```

## 配置系統

### 配置類別

使用 Pydantic 進行配置管理和驗證：

```python
from src.currency_predictor.config import ConfigManager

config = ConfigManager()
model_config = config.get_model_config()
```

### 配置文件 (config.json)

```json
{
    "model_name": "patchtst_huggingface",
    "model_params": {
        "seq_len": 64,
        "pred_len": 7,
        "d_model": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2
    },
    "symbols": ["USDTWD=X"],
    "prediction_horizon": 7
}
```

## 使用流程

### 1. 資料收集階段

```python
from src.currency_predictor.data import YahooFinanceCollector, DataStorage

collector = YahooFinanceCollector()
storage = DataStorage()

# 收集資料
data = collector.get_currency_data('USDTWD=X', period='1y')

# 儲存原始資料
storage.save_raw_data(data, 'USDTWD', '1y')
```

### 2. 模型訓練階段

```python
from src.currency_predictor.models import PatchTSTSklearn

# 創建並訓練模型
model = PatchTSTSklearn(seq_len=64, pred_len=7)
model.fit(X_train, y_train)
```

### 3. 預測階段

```python
from src.currency_predictor.prediction import PredictionPipeline

pipeline = PredictionPipeline(config)
results = pipeline.run_full_pipeline(symbols=['USDTWD=X'])
```

## 目錄結構

```text
currency_predict_attempt/
├── src/currency_predictor/   # 主要程式碼
├── data/                     # 資料儲存
│   ├── raw/                  # 原始資料
│   └── processed/            # 處理後資料
├── models/                   # 訓練好的模型
├── results/                  # 預測結果
├── logs/                     # 日誌檔案
├── docs/                     # 文件
├── notebooks/                # Jupyter notebooks
├── tests/                    # 測試檔案
├── examples/                 # 使用範例
├── config.json               # 主配置檔
├── main.py                   # 主入口點
└── pyproject.toml            # 套件配置
```

## 設計模式

### 使用的模式

1. **Factory Pattern** - ModelFactory 用於動態創建模型
2. **Abstract Factory** - BaseModel 定義統一介面
3. **Pipeline Pattern** - PredictionPipeline 編排處理流程
4. **Strategy Pattern** - 不同的模型實作可互換
5. **Facade Pattern** - CurrencyPredictor 簡化複雜操作

## 擴展指南

### 新增模型實作

1. 在 `models/patchtst/` 下創建新的子目錄
2. 繼承 `SklearnBasedModel` 或 `TransformerBasedModel`
3. 實作 `fit()`, `predict()`, `predict_with_uncertainty()` 方法
4. 在 `factory.py` 中註冊新模型

### 新增資料來源

1. 在 `data/collectors.py` 中創建新的收集器類別
2. 繼承或實作相同的介面
3. 在工廠中註冊

## 相關文件

- [模型模組使用說明](../usage_descriptions/model_modules.md)
- [資料模組使用說明](../usage_descriptions/data_modules.md)
- [配置管理說明](../usage_descriptions/config_management.md)
- [架構分析報告](../development/architecture_analysis_report.md)
- [PatchTST 實作路線圖](../development/patchtst_implementation_roadmap.md)
