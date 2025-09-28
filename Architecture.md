# Currency Prediction Project Architecture

## 專案概述

本專案旨在建立一個模組化的貨幣匯率預測系統，使用 PatchTST 模型進行時間序列預測。

## 設計原則

1. **模組化設計** - 資料下載、處理、預測完全分離
2. **可擴展性** - 支援多種貨幣對和預測模型
3. **可重現性** - 完整的配置管理和日誌記錄
4. **測試友好** - 每個模組都可獨立測試

## 當前架構分析

### 現有模組結構
```
src/currency_predictor/
├── __init__.py           # 套件初始化
├── data_collector.py     # 資料收集 ✓
├── data_processor.py     # 資料處理 ✓  
├── models.py            # 模型定義 (需重構)
└── utils.py             # 工具函數 ✓
```

### 架構問題
1. **模組職責不清晰** - `models.py` 包含了太多功能
2. **缺少預測模組** - 沒有獨立的預測執行模組
3. **配置管理不足** - 配置散落在各處
4. **缺少資料管理** - 沒有專門的資料儲存和版本控制

## 建議的新架構

### 核心模組設計

```
src/currency_predictor/
├── __init__.py                    # 套件初始化
│
├── data/                          # 資料相關模組
│   ├── __init__.py
│   ├── collectors.py              # 資料收集器 (重構 data_collector.py)
│   ├── processors.py              # 資料處理器 (重構 data_processor.py)  
│   ├── storage.py                 # 資料儲存和載入
│   └── validators.py              # 資料驗證
│
├── models/                        # 模型相關模組
│   ├── __init__.py
│   ├── base.py                    # 基礎模型介面
│   ├── patchtst.py                # PatchTST 模型實作
│   ├── traditional.py             # 傳統模型 (RF, LSTM等)
│   └── ensemble.py                # 集成模型
│
├── prediction/                    # 預測相關模組
│   ├── __init__.py
│   ├── predictor.py               # 預測執行器
│   ├── evaluator.py               # 模型評估
│   └── pipeline.py                # 預測管道
│
├── config/                        # 配置管理
│   ├── __init__.py
│   ├── settings.py                # 配置類別
│   └── defaults.py                # 默認配置
│
└── utils/                         # 工具模組
    ├── __init__.py
    ├── logging.py                 # 日誌工具
    ├── visualization.py           # 視覺化工具
    └── helpers.py                 # 其他工具函數
```

### 資料流程設計

```
1. 資料收集 (data/collectors.py)
   ↓
2. 資料儲存 (data/storage.py)
   ↓  
3. 資料處理 (data/processors.py)
   ↓
4. 特徵工程 (data/processors.py)
   ↓
5. 模型訓練 (models/*.py)
   ↓
6. 模型評估 (prediction/evaluator.py)
   ↓
7. 預測執行 (prediction/predictor.py)
```

## 詳細模組職責

### 1. 資料模組 (`data/`)

#### `collectors.py` - 資料收集器
```python
class YahooFinanceCollector:
    """Yahoo Finance 資料收集器"""
    
class CentralBankCollector:
    """央行資料收集器 (備用)"""
    
class DataCollectorFactory:
    """資料收集器工廠"""
```

#### `processors.py` - 資料處理器  
```python
class TimeSeriesProcessor:
    """時間序列資料處理"""
    
class FeatureEngineer:
    """特徵工程"""
    
class TechnicalIndicators:
    """技術指標計算"""
```

#### `storage.py` - 資料儲存
```python
class DataStorage:
    """資料儲存管理"""
    # - 原始資料儲存
    # - 處理後資料儲存  
    # - 資料版本控制
    # - 資料載入
```

### 2. 模型模組 (`models/`)

#### `base.py` - 基礎模型介面
```python
class BaseModel(ABC):
    """所有模型的基礎類別"""
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod  
    def predict(self, X):
        pass
```

#### `patchtst.py` - PatchTST 模型
```python
class PatchTST(BaseModel):
    """PatchTST 時間序列預測模型"""
```

### 3. 預測模組 (`prediction/`)

#### `predictor.py` - 預測執行器
```python
class CurrencyPredictor:
    """貨幣預測執行器"""
    # - 載入訓練好的模型
    # - 執行預測
    # - 結果後處理
```

#### `pipeline.py` - 預測管道
```python  
class PredictionPipeline:
    """完整的預測管道"""
    # - 資料收集 → 處理 → 預測 → 評估
```

### 4. 配置模組 (`config/`)

#### `settings.py` - 配置管理
```python
class DataConfig:
    """資料相關配置"""
    
class ModelConfig:  
    """模型相關配置"""
    
class PredictionConfig:
    """預測相關配置"""
```

## 使用流程

### 1. 資料收集階段
```python
from currency_predictor.data import YahooFinanceCollector, DataStorage

collector = YahooFinanceCollector()
storage = DataStorage()

# 收集資料
data = collector.get_currency_data('USDTWD=X', period='1M')

# 儲存原始資料
storage.save_raw_data(data, 'USDTWD', '2025-09')
```

### 2. 模型訓練階段
```python
from currency_predictor.data import DataStorage, TimeSeriesProcessor
from currency_predictor.models import PatchTST

# 載入和處理資料
storage = DataStorage()
processor = TimeSeriesProcessor()

raw_data = storage.load_raw_data('USDTWD', '2025-09')
processed_data = processor.process(raw_data)

# 訓練模型
model = PatchTST()
model.fit(processed_data)
```

### 3. 預測階段
```python
from currency_predictor.prediction import CurrencyPredictor

predictor = CurrencyPredictor()
predictions = predictor.predict('USDTWD', horizon=7)
```

## 配置文件結構

```json
{
  "data": {
    "sources": ["yahoo_finance", "central_bank"],
    "currency_pairs": ["USDTWD=X", "EURUSD=X"],
    "update_frequency": "daily"
  },
  "models": {
    "patchtst": {
      "patch_len": 16,
      "stride": 8,
      "d_model": 128
    }
  },
  "prediction": {
    "horizon": 7,
    "confidence_interval": 0.95
  }
}
```

## 目錄結構

```
currency_predict_attempt/
├── src/currency_predictor/          # 主要程式碼
├── data/                           # 資料儲存
│   ├── raw/                        # 原始資料
│   ├── processed/                  # 處理後資料
│   └── features/                   # 特徵資料
├── models/                         # 訓練好的模型
├── results/                        # 預測結果
├── logs/                          # 日誌檔案
├── configs/                       # 配置檔案
├── notebooks/                     # Jupyter notebooks
├── tests/                         # 測試檔案
└── scripts/                       # 執行腳本
    ├── collect_data.py            # 資料收集腳本
    ├── train_model.py             # 模型訓練腳本
    └── predict.py                 # 預測腳本
```

## 下一步實作計劃

1. **第一階段：重構資料模組**
   - 分離 collectors 和 processors
   - 實作 storage 模組
   
2. **第二階段：實作 PatchTST 模型**
   - 基礎模型介面
   - PatchTST 實作
   
3. **第三階段：建立預測管道**
   - 預測執行器
   - 評估模組

4. **第四階段：整合和測試**
   - 端到端測試
   - 性能優化

## 優勢分析

### 相對於現有架構的改進：
1. **職責分離** - 每個模組職責單一明確
2. **可測試性** - 每個模組可獨立測試  
3. **可擴展性** - 新增模型或資料源容易
4. **維護性** - 修改一個功能不影響其他模組
5. **重用性** - 模組可在不同專案中重用

你覺得這個架構設計如何？有什麼需要調整或補充的地方嗎？