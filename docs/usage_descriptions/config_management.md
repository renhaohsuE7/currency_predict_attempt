# 配置管理使用說明

本文檔說明如何使用 `ConfigManager` 和 `ConfigValidator` 管理應用程式配置。

## 目錄

- [概述](#概述)
- [Pydantic Settings 優勢](#pydantic-settings-優勢)
- [ConfigManager 使用](#configmanager-使用)
- [環境變數支援](#環境變數支援)
- [ConfigValidator 使用](#configvalidator-使用)
- [配置文件格式](#配置文件格式)
- [最佳實踐](#最佳實踐)

---

## 概述

`ConfigManager` 提供了基於 **Pydantic Settings** 的配置載入、驗證、更新和儲存功能。它確保應用程式始終使用有效的配置，並提供默認值作為後備。

### 主要特點

- ✅ **基於 Pydantic Settings** - 強大的型別驗證和自動轉換
- ✅ 自動載入配置文件
- ✅ **環境變數支援** - 可透過環境變數覆蓋配置
- ✅ 提供默認配置
- ✅ **型別安全** - 編譯時型別檢查
- ✅ 驗證配置有效性
- ✅ 自動修正無效配置
- ✅ 支援配置更新和儲存
- ✅ 深拷貝保護

---

## Pydantic Settings 優勢

### 為什麼使用 Pydantic Settings？

1. **型別安全**
   - 自動型別驗證和轉換
   - 編譯時型別檢查（與 mypy 配合）
   - 避免運行時型別錯誤

2. **環境變數支援**
   - 自動從環境變數載入配置
   - 支援 `.env` 文件
   - 適合容器化部署

3. **資料驗證**
   - 自動驗證數值範圍
   - 自定義驗證規則
   - 詳細的錯誤訊息

4. **預設值管理**
   - 集中管理默認值
   - 自動填充缺失的配置
   - 向後兼容性

### 架構

```
AppSettings (Pydantic BaseSettings)
├── ModelParams (BaseModel)
├── DataCollectionConfig (BaseModel)
├── ModelTrainingConfig (BaseModel)
└── PredictionConfig (BaseModel)
```

---

## ConfigManager 使用

### 基本用法

#### 1. 創建 ConfigManager

```python
from src.currency_predictor.config.manager import ConfigManager

# 使用默認配置文件 (config.json)
config_manager = ConfigManager()

# 使用自定義配置文件
config_manager = ConfigManager(config_path='my_config.json')

# 不進行驗證
config_manager = ConfigManager(validate=False)
```

#### 2. 取得配置

```python
# 取得完整配置字典
config = config_manager.get_config()

# 取得單一配置值
model_name = config_manager.get('model_name')

# 使用默認值
log_level = config_manager.get('log_level', 'INFO')
```

#### 3. 專用方法

```python
# 取得模型配置
model_config = config_manager.get_model_config()
# 返回: {'model_name': 'PatchTST', 'model_params': {...}}

# 取得貨幣符號列表
symbols = config_manager.get_symbols()
# 返回: ['USDTWD=X', 'EURUSD=X', ...]

# 取得預測範圍
horizon = config_manager.get_prediction_horizon()
# 返回: 7
```

---

## 環境變數支援

### 使用環境變數覆蓋配置

Pydantic Settings 支援透過環境變數覆蓋配置值。所有環境變數都使用 `CURRENCY_PRED_` 前綴。

#### 設置環境變數

**Linux/Mac:**

```bash
export CURRENCY_PRED_MODEL_NAME="CustomModel"
export CURRENCY_PRED_MODEL_PARAMS__SEQ_LEN=200
export CURRENCY_PRED_LOG_LEVEL="DEBUG"
```

**Windows:**

```cmd
set CURRENCY_PRED_MODEL_NAME=CustomModel
set CURRENCY_PRED_MODEL_PARAMS__SEQ_LEN=200
set CURRENCY_PRED_LOG_LEVEL=DEBUG
```

**巢狀配置使用雙底線 (`__`):**

```bash
# 設置 model_params.seq_len
export CURRENCY_PRED_MODEL_PARAMS__SEQ_LEN=200

# 設置 model_params.pred_len
export CURRENCY_PRED_MODEL_PARAMS__PRED_LEN=30

# 設置 data_collection.period
export CURRENCY_PRED_DATA_COLLECTION__PERIOD="2y"
```

#### 使用 .env 文件

創建 `.env` 文件在專案根目錄：

```env
# 模型配置
CURRENCY_PRED_MODEL_NAME=PatchTST
CURRENCY_PRED_MODEL_PARAMS__SEQ_LEN=200
CURRENCY_PRED_MODEL_PARAMS__PRED_LEN=30

# 日誌配置
CURRENCY_PRED_LOG_LEVEL=DEBUG

# 資料收集配置
CURRENCY_PRED_DATA_COLLECTION__PERIOD=2y
CURRENCY_PRED_DATA_COLLECTION__INTERVAL=1h

# 貨幣符號（使用 JSON 格式）
CURRENCY_PRED_SYMBOLS=["USDTWD=X", "EURUSD=X"]
```

ConfigManager 會自動載入 `.env` 文件中的配置。

#### 優先順序

配置的載入優先順序（由高到低）：

1. **環境變數** - `CURRENCY_PRED_*`
2. **JSON 配置文件** - `config.json`
3. **預設值** - Pydantic 模型中定義的預設值

```python
# 範例：環境變數覆蓋
import os

# 設置環境變數
os.environ['CURRENCY_PRED_MODEL_NAME'] = 'CustomModel'
os.environ['CURRENCY_PRED_MODEL_PARAMS__SEQ_LEN'] = '200'

# 載入配置（環境變數優先）
config_manager = ConfigManager()
print(config_manager.get('model_name'))  # 輸出: CustomModel
print(config_manager.get('model_params')['seq_len'])  # 輸出: 200
```

#### 直接使用 AppSettings

如果您需要更細緻的控制，可以直接使用 `AppSettings`：

```python
from currency_predictor.config import AppSettings
import os

# 設置環境變數
os.environ['CURRENCY_PRED_MODEL_NAME'] = 'CustomModel'

# 直接創建 Settings
settings = AppSettings()
print(settings.model_name)  # CustomModel

# 或從 JSON 載入
from currency_predictor.config import load_settings_from_json
settings = load_settings_from_json('config.json')
```

### 更新配置

```python
# 更新配置（不儲存）
config_manager.update({
    'symbols': ['USDTWD=X'],
    'prediction_horizon': 3
})

# 更新並儲存
config_manager.update(
    {'symbols': ['EURUSD=X']},
    save=True
)
```

### 儲存配置

```python
# 儲存到原路徑
config_manager.save()

# 儲存到新路徑
from pathlib import Path
config_manager.save(Path('new_config.json'))
```

---

## ConfigValidator 使用

### 驗證配置

`ConfigValidator` 使用 Pydantic 進行配置驗證，提供強大的型別檢查和資料驗證。

```python
from currency_predictor.config.manager import ConfigValidator

config = {
    "model_name": "PatchTST",
    "model_params": {
        "seq_len": 100,
        "pred_len": 10
    },
    "symbols": ["USDTWD=X"]
}

is_valid, errors = ConfigValidator.validate_config(config)

if is_valid:
    print("配置有效")
else:
    print("配置錯誤：")
    for error in errors:
        print(f"  - {error}")
```

### Pydantic 驗證特性

#### 自動填充預設值

Pydantic 會自動為缺失的欄位填充預設值：

```python
# 最小配置
minimal_config = {
    "model_name": "PatchTST"
}

is_valid, errors = ConfigValidator.validate_config(minimal_config)
# is_valid = True - Pydantic 自動填充其他欄位的預設值
```

#### 型別自動轉換

Pydantic 會自動進行型別轉換：

```python
config = {
    "model_name": "PatchTST",
    "model_params": {
        "seq_len": "100",  # 字串會自動轉換為整數
        "pred_len": "10"
    },
    "symbols": ["USDTWD=X"]
}

is_valid, errors = ConfigValidator.validate_config(config)
# is_valid = True - 型別自動轉換
```

#### 數值範圍驗證

Pydantic 會驗證數值範圍：

```python
invalid_config = {
    "model_name": "PatchTST",
    "model_params": {
        "seq_len": -10,  # 無效：必須 > 0
        "pred_len": 10
    },
    "symbols": ["USDTWD=X"]
}

is_valid, errors = ConfigValidator.validate_config(invalid_config)
# is_valid = False
# errors = ['model_params: Input should be greater than 0']
```

### 驗證規則

#### 必要的頂層鍵

所有欄位都有預設值，但建議提供：

- `model_name` - 模型名稱（預設：'PatchTST'）
- `model_params` - 模型參數字典
- `symbols` - 貨幣符號列表

#### model_params 驗證規則

- `seq_len` - 序列長度（必須 > 0，預設：168）
- `pred_len` - 預測長度（必須 > 0，預設：24）
- `patch_len` - Patch 長度（必須 > 0，預設：12）
- `stride` - 步長（必須 > 0，預設：6）
- `n_estimators` - 估計器數量（必須 > 0，預設：100）
- `max_depth` - 最大深度（必須 > 0，預設：10）
- `random_state` - 隨機種子（預設：42）

#### 其他驗證規則

- `symbols` - 必須是列表，至少包含一個元素
- `prediction_horizon` - 必須 > 0（預設：7）
- `log_level` - 必須是有效的日誌級別（預設：'INFO'）
- `data_storage_path` - 資料儲存路徑（預設：'data'）

---

## 配置文件格式

### 完整配置範例

```json
{
  "model_name": "PatchTST",
  "model_params": {
    "seq_len": 168,
    "pred_len": 24,
    "patch_len": 12,
    "stride": 6,
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "data_storage_path": "data",
  "log_level": "INFO",
  "data_collection": {
    "period": "1y",
    "interval": "1d",
    "force_update": false
  },
  "model_training": {
    "period": "1y",
    "target_column": "Close",
    "feature_columns": null,
    "train_params": {
      "validation_split": 0.2
    }
  },
  "prediction": {
    "period": "1y",
    "return_uncertainty": true
  },
  "symbols": [
    "USDTWD=X",
    "EURUSD=X",
    "GBPUSD=X"
  ],
  "prediction_horizon": 7
}
```

### 最小配置範例

```json
{
  "model_name": "PatchTST",
  "model_params": {
    "seq_len": 100,
    "pred_len": 10
  },
  "symbols": ["USDTWD=X"]
}
```

---

## 最佳實踐

### 1. 使用配置驗證

總是在生產環境中啟用配置驗證：

```python
# 好的做法
config_manager = ConfigManager(validate=True)  # 默認

# 不好的做法（除非有特殊原因）
config_manager = ConfigManager(validate=False)
```

### 2. 處理配置錯誤

```python
from src.currency_predictor.config.manager import ConfigManager

try:
    config_manager = ConfigManager('config.json')
    config = config_manager.get_config()
except Exception as e:
    print(f"配置載入失敗: {e}")
    # 使用默認配置或退出
```

### 3. 環境特定配置

```python
import os
from src.currency_predictor.config.manager import ConfigManager

# 根據環境載入不同配置
env = os.getenv('ENV', 'development')
config_file = f'config.{env}.json'

config_manager = ConfigManager(config_path=config_file)
```

### 4. 合併配置

```python
# 載入基礎配置
config_manager = ConfigManager('base_config.json')

# 覆蓋特定設定
custom_settings = {
    'symbols': ['USDTWD=X'],
    'prediction_horizon': 3
}
config_manager.update(custom_settings)
```

### 5. 配置版本控制

```python
# 在配置中添加版本號
config = {
    "version": "2.0",
    "model_name": "PatchTST",
    # ... 其他配置
}

# 檢查版本
version = config_manager.get('version', '1.0')
if version != '2.0':
    print("警告：配置版本不匹配")
```

---

## 完整範例

### 範例 1：基本使用

```python
from src.currency_predictor.config.manager import ConfigManager

# 創建配置管理器
config_manager = ConfigManager()

# 取得配置
model_name = config_manager.get('model_name')
symbols = config_manager.get_symbols()
horizon = config_manager.get_prediction_horizon()

print(f"Model: {model_name}")
print(f"Symbols: {symbols}")
print(f"Horizon: {horizon} days")
```

### 範例 2：自定義配置

```python
from src.currency_predictor.config.manager import ConfigManager

# 創建自定義配置
custom_config = {
    "model_name": "PatchTST",
    "model_params": {
        "seq_len": 50,
        "pred_len": 5
    },
    "symbols": ["EURUSD=X"],
    "prediction_horizon": 3
}

# 創建配置管理器（從空開始）
config_manager = ConfigManager(validate=False)
config_manager.update(custom_config)

# 儲存到文件
config_manager.save('my_custom_config.json')
```

### 範例 3：配置驗證

```python
from src.currency_predictor.config.manager import ConfigManager, ConfigValidator

# 載入可能有問題的配置
config_manager = ConfigManager('user_config.json', validate=True)

# ConfigManager 會自動修正問題並記錄警告
config = config_manager.get_config()

# 手動驗證
is_valid, errors = ConfigValidator.validate_config(config)
if not is_valid:
    print("配置已自動修正以下問題：")
    for error in errors:
        print(f"  - {error}")
```

---

## 相關文檔

- [模型模組使用說明](./model_modules.md)
- [範例使用指南](./examples_guide.md)
- [測試報告](../testing_reports/main_refactoring_report.md)

---

## 常見問題

### Q: 如果配置文件不存在會怎樣？

A: ConfigManager 會自動使用默認配置，並記錄資訊日誌。

### Q: 如何知道配置是否被修正過？

A: 如果配置驗證失敗，會記錄警告日誌，顯示所有被修正的問題。

### Q: 可以動態更新配置嗎？

A: 可以使用 `update()` 方法動態更新配置，但不會影響已創建的對象。

### Q: 默認配置在哪裡定義？

A: 默認配置在 `ConfigManager.DEFAULT_CONFIG` 中定義，可以直接訪問。
