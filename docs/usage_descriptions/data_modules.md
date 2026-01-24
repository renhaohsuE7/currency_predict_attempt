# 資料模組使用說明

本文檔說明如何使用資料相關的模組進行貨幣資料的收集、儲存和處理。

## 目錄

- [資料收集 (DataCollector)](#資料收集-datacollector)
- [資料儲存 (DataStorage)](#資料儲存-datastorage)
- [資料處理 (DataProcessor)](#資料處理-dataprocessor)

---

## 資料收集 (DataCollector)

### 概述

`YahooFinanceCollector` 類別用於從 Yahoo Finance API 收集貨幣匯率資料。

### 導入

```python
from src.currency_predictor.data.collectors import YahooFinanceCollector
```

### 基本用法

#### 創建收集器實例

```python
collector = YahooFinanceCollector()
```

#### 收集單一貨幣對資料

```python
# 收集美元對台幣的匯率資料
data = collector.get_currency_data(
    symbol='USDTWD=X',
    period='1y',      # 1年的歷史資料
    interval='1d'     # 每日資料
)

if data is not None:
    print(f"成功收集 {len(data)} 筆資料")
    print(data.head())
else:
    print("資料收集失敗")
```

### 參數說明

#### `get_currency_data()` 方法

| 參數 | 類型 | 說明 | 可選值 |
|------|------|------|--------|
| `symbol` | str | 貨幣對符號 | `'USDTWD=X'`, `'EURUSD=X'`, `'GBPUSD=X'` 等 |
| `period` | str | 時間範圍 | `'1d'`, `'5d'`, `'1mo'`, `'3mo'`, `'6mo'`, `'1y'`, `'2y'`, `'5y'`, `'10y'`, `'ytd'`, `'max'` |
| `interval` | str | 資料間隔 | `'1m'`, `'5m'`, `'1h'`, `'1d'`, `'1wk'`, `'1mo'` |

### 完整範例

```python
from src.currency_predictor.data.collectors import YahooFinanceCollector

# 創建收集器
collector = YahooFinanceCollector()

# 收集多個貨幣對
currency_pairs = ['USDTWD=X', 'EURUSD=X', 'GBPUSD=X']

for pair in currency_pairs:
    print(f"正在收集 {pair}...")
    data = collector.get_currency_data(
        symbol=pair,
        period='6mo',
        interval='1d'
    )

    if data is not None:
        print(f"  成功：{len(data)} 筆資料")
        print(f"  日期範圍：{data.index[0]} 至 {data.index[-1]}")
        print(f"  欄位：{list(data.columns)}")
    else:
        print(f"  失敗：無法取得資料")
    print()
```

### 返回的資料格式

返回的 DataFrame 包含以下欄位：

| 欄位 | 說明 |
|------|------|
| `Open` | 開盤價 |
| `High` | 最高價 |
| `Low` | 最低價 |
| `Close` | 收盤價 |
| `Volume` | 交易量 |

索引為日期時間 (DatetimeIndex)。

---

## 資料儲存 (DataStorage)

### 概述

`DataStorage` 類別提供資料的儲存、載入和管理功能。

### 導入

```python
from src.currency_predictor.data.storage import DataStorage
```

### 基本用法

#### 創建儲存管理器

```python
storage = DataStorage(base_dir='data')
```

#### 儲存原始資料

```python
# 假設已經收集了資料
success = storage.save_raw_data(
    data=currency_data,
    symbol='USDTWD',
    period='1y'
)

if success:
    print("資料儲存成功")
```

#### 載入原始資料

```python
data = storage.load_raw_data(
    symbol='USDTWD',
    period='1y'
)

if data is not None:
    print(f"載入了 {len(data)} 筆資料")
```

### 主要方法

#### `save_raw_data(data, symbol, period)`

儲存原始資料到 `data/raw/` 目錄。

**參數：**
- `data`: pd.DataFrame - 要儲存的資料
- `symbol`: str - 貨幣對符號（去除 =X）
- `period`: str - 資料期間

**返回：** bool - 是否儲存成功

#### `load_raw_data(symbol, period)`

從 `data/raw/` 載入原始資料。

**參數：**
- `symbol`: str - 貨幣對符號
- `period`: str - 資料期間

**返回：** pd.DataFrame 或 None

#### `save_processed_data(data, symbol, period)`

儲存處理後的資料到 `data/processed/` 目錄。

#### `load_processed_data(symbol, period)`

從 `data/processed/` 載入處理後的資料。

#### `list_available_data()`

列出所有可用的資料文件。

**返回：** List[Dict] - 資料文件資訊列表

#### `data_exists(symbol, period, data_type='raw')`

檢查資料是否存在。

**參數：**
- `symbol`: str
- `period`: str
- `data_type`: str - `'raw'` 或 `'processed'`

**返回：** bool

### 完整範例

```python
from src.currency_predictor.data.collectors import YahooFinanceCollector
from src.currency_predictor.data.storage import DataStorage

# 創建實例
collector = YahooFinanceCollector()
storage = DataStorage(base_dir='data')

# 收集並儲存資料
symbol = 'USDTWD=X'
data = collector.get_currency_data(symbol, period='1y')

if data is not None:
    # 儲存原始資料
    success = storage.save_raw_data(data, 'USDTWD', '1y')
    print(f"儲存成功：{success}")

    # 檢查資料是否存在
    exists = storage.data_exists('USDTWD', '1y', 'raw')
    print(f"資料存在：{exists}")

    # 載入資料
    loaded_data = storage.load_raw_data('USDTWD', '1y')
    print(f"載入了 {len(loaded_data)} 筆資料")

    # 列出所有可用資料
    available = storage.list_available_data()
    print(f"可用資料：{len(available)} 個文件")
```

---

## 資料處理 (DataProcessor)

### 概述

`DataProcessor` 類別提供資料清理、特徵工程和預處理功能。

### 導入

```python
from src.currency_predictor.data_processor import DataProcessor
```

### 基本用法

#### 創建處理器

```python
processor = DataProcessor()
```

#### 清理資料

```python
# 清理原始資料
cleaned_data = processor.clean_data(raw_data)
```

#### 創建技術指標

```python
# 添加技術指標
data_with_indicators = processor.create_technical_indicators(cleaned_data)

# 新增的指標包括：
# - MA_5, MA_10, MA_20, MA_50 (移動平均)
# - EMA_12, EMA_26 (指數移動平均)
# - MACD, MACD_Signal, MACD_Histogram
# - RSI (相對強弱指標)
# - Bollinger Bands (布林通道)
```

### 主要方法

#### `clean_data(data)`

清理資料，包括：
- 移除重複值
- 處理缺失值
- 轉換日期索引
- 排序資料

**參數：** data: pd.DataFrame
**返回：** pd.DataFrame

#### `create_technical_indicators(data)`

創建技術指標。

**參數：** data: pd.DataFrame
**返回：** pd.DataFrame

#### `create_lagged_features(data, target_column, lags)`

創建滯後特徵。

**參數：**
- `data`: pd.DataFrame
- `target_column`: str - 目標欄位名稱
- `lags`: List[int] - 滯後期數列表

**返回：** pd.DataFrame

#### `prepare_features_target(data, target_column, prediction_horizon)`

準備特徵和目標變數用於訓練。

**參數：**
- `data`: pd.DataFrame
- `target_column`: str - 目標欄位
- `prediction_horizon`: int - 預測範圍

**返回：** Tuple[pd.DataFrame, pd.Series] - (X, y)

#### `scale_features(features, scaler_type='standard')`

縮放特徵。

**參數：**
- `features`: pd.DataFrame
- `scaler_type`: str - `'standard'` 或 `'minmax'`

**返回：** pd.DataFrame

### 完整範例

```python
from src.currency_predictor.data.storage import DataStorage
from src.currency_predictor.data_processor import DataProcessor

# 載入資料
storage = DataStorage()
raw_data = storage.load_raw_data('USDTWD', '1y')

# 創建處理器
processor = DataProcessor()

# 1. 清理資料
cleaned_data = processor.clean_data(raw_data)
print(f"清理後資料：{len(cleaned_data)} 筆")

# 2. 創建技術指標
data_with_indicators = processor.create_technical_indicators(cleaned_data)
print(f"特徵數：{data_with_indicators.shape[1]}")

# 3. 創建滯後特徵
data_with_lags = processor.create_lagged_features(
    data_with_indicators,
    target_column='Close',
    lags=[1, 2, 3, 5, 10]
)

# 4. 準備訓練資料
X, y = processor.prepare_features_target(
    data_with_lags,
    target_column='Close',
    prediction_horizon=1
)

print(f"特徵矩陣：{X.shape}")
print(f"目標變數：{y.shape}")

# 5. 分割訓練測試集
X_train, X_test, y_train, y_test = processor.train_test_split(
    X, y, test_size=0.2
)

print(f"訓練集：{len(X_train)} 筆")
print(f"測試集：{len(X_test)} 筆")
```

---

## 完整工作流程範例

以下是一個完整的資料處理工作流程：

```python
from src.currency_predictor.data.collectors import YahooFinanceCollector
from src.currency_predictor.data.storage import DataStorage
from src.currency_predictor.data_processor import DataProcessor

# 1. 收集資料
collector = YahooFinanceCollector()
data = collector.get_currency_data('USDTWD=X', period='1y')

# 2. 儲存原始資料
storage = DataStorage()
storage.save_raw_data(data, 'USDTWD', '1y')

# 3. 處理資料
processor = DataProcessor()
cleaned = processor.clean_data(data)
with_indicators = processor.create_technical_indicators(cleaned)
X, y = processor.prepare_features_target(with_indicators, 'Close', 1)

# 4. 儲存處理後的資料
storage.save_processed_data(with_indicators, 'USDTWD', '1y')

# 5. 分割資料集
X_train, X_test, y_train, y_test = processor.train_test_split(X, y, test_size=0.2)

print("資料準備完成！")
print(f"訓練集：{len(X_train)} 筆")
print(f"測試集：{len(X_test)} 筆")
print(f"特徵數：{X_train.shape[1]}")
```

---

## 常見問題

### Q: 如何處理資料收集失敗的情況？

A: 總是檢查返回值是否為 None：

```python
data = collector.get_currency_data('SYMBOL=X', period='1y')
if data is None:
    print("資料收集失敗，請檢查：")
    print("1. 網絡連接")
    print("2. 貨幣符號是否正確")
    print("3. Yahoo Finance API 是否可用")
```

### Q: 技術指標計算後有 NaN 值怎麼辦？

A: 某些技術指標需要一定的資料量才能計算。使用 `dropna()` 移除：

```python
data_with_indicators = processor.create_technical_indicators(data)
clean_data = data_with_indicators.dropna()
```

### Q: 如何自定義技術指標參數？

A: 目前技術指標使用默認參數。如需自定義，可以在資料處理後手動添加：

```python
data['MA_30'] = data['Close'].rolling(window=30).mean()
data['Custom_Indicator'] = # 您的計算邏輯
```

---

## 相關文檔

- [模型使用說明](./model_modules.md)
- [預測管道使用說明](./prediction_pipeline.md)
- [API 參考](./api_reference.md)
