# 模型模組使用說明

本文檔說明如何使用模型相關模組進行貨幣預測模型的訓練和預測。

## 目錄

- [PatchTST 模型](#patchtst-模型)
- [模型工廠 (ModelFactory)](#模型工廠-modelfactory)
- [基礎模型介面](#基礎模型介面)

---

## PatchTST 模型

### 概述

`PatchTST` 是一個基於 sklearn 的時間序列預測模型，使用 patch 技術和集成學習進行預測。

### 導入

```python
from src.currency_predictor.models.patchtst import PatchTST
```

### 基本用法

#### 創建模型

```python
model = PatchTST(
    seq_len=168,      # 輸入序列長度（例如：7天 x 24小時）
    pred_len=24,      # 預測長度（例如：未來24小時）
    patch_len=12,     # 每個 patch 的長度
    stride=6,         # patch 之間的步長
    n_estimators=100, # 集成模型的估計器數量
    max_depth=10,     # 決策樹最大深度
    random_state=42   # 隨機種子（用於可重現性）
)
```

### 參數說明

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `seq_len` | int | 168 | 輸入序列長度，建議使用 50-200 |
| `pred_len` | int | 24 | 預測序列長度 |
| `patch_len` | int | 12 | Patch 長度，建議為 seq_len 的 1/10 左右 |
| `stride` | int | 6 | Patch 步長，建議為 patch_len 的一半 |
| `n_estimators` | int | 100 | 隨機森林估計器數量，越大越慢但可能更準確 |
| `max_depth` | int | 10 | 決策樹最大深度，控制模型復雜度 |
| `random_state` | int | 42 | 隨機種子，設置以確保可重現性 |

### 訓練模型

```python
# 準備訓練資料（假設已經處理好）
# X: DataFrame，特徵資料
# y: Series，目標變數

model.fit(X_train, y_train)
print("模型訓練完成！")
```

#### 使用驗證資料訓練

```python
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val)
)
```

### 進行預測

```python
# X_test 需要至少包含 seq_len 筆資料
predictions = model.predict(X_test, horizon=1)

print(f"預測結果：{predictions}")
```

### 帶不確定性的預測

```python
result = model.predict_with_uncertainty(
    X_test,
    horizon=1,
    confidence_level=0.95  # 95% 信賴區間
)

print(f"預測值：{result['predictions']}")
print(f"標準差：{result['std']}")
print(f"下界：{result['lower_bound']}")
print(f"上界：{result['upper_bound']}")
```

### 完整範例

```python
from src.currency_predictor.data.storage import DataStorage
from src.currency_predictor.data_processor import DataProcessor
from src.currency_predictor.models.patchtst import PatchTST
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. 載入和處理資料
storage = DataStorage()
processor = DataProcessor()

raw_data = storage.load_raw_data('USDTWD', '1y')
cleaned = processor.clean_data(raw_data)
with_indicators = processor.create_technical_indicators(cleaned)
X, y = processor.prepare_features_target(with_indicators, 'Close', 1)

# 2. 分割資料
X_train, X_test, y_train, y_test = processor.train_test_split(
    X, y, test_size=0.2
)

# 3. 創建並訓練模型
model = PatchTST(
    seq_len=50,
    pred_len=5,
    patch_len=10,
    stride=5,
    n_estimators=100,
    max_depth=10,
    random_state=42
)

print("開始訓練模型...")
model.fit(X_train, y_train)
print("訓練完成！")

# 4. 評估模型
# 確保測試資料足夠長
test_start = len(X_train)
test_input = X[test_start:test_start+100]  # 使用100筆資料
test_target = y[test_start:test_start+100]

predictions = model.predict(test_input, horizon=1)

# 計算誤差指標
mae = mean_absolute_error(test_target[:len(predictions)], predictions)
rmse = np.sqrt(mean_squared_error(test_target[:len(predictions)], predictions))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# 5. 儲存模型
model.save_model('models/patchtst_usdtwd.joblib')
print("模型已儲存")
```

### 模型資訊

#### 取得模型資訊

```python
info = model.get_model_info()

print(f"模型名稱：{info['model_name']}")
print(f"模型類型：{info['model_type']}")
print(f"是否已訓練：{info['is_fitted']}")
print(f"參數：{info['model_params']}")
```

### 儲存和載入模型

#### 儲存模型

```python
success = model.save_model('models/my_model.joblib')
if success:
    print("模型儲存成功")
```

#### 載入模型

```python
loaded_model = PatchTST()
success = loaded_model.load_model('models/my_model.joblib')
if success:
    print("模型載入成功")
    # 現在可以使用 loaded_model 進行預測
```

---

## 模型工廠 (ModelFactory)

### 概述

`ModelFactory` 提供統一的模型創建介面，支援多種模型類型。

### 導入

```python
from src.currency_predictor.models.factory import ModelFactory, create_patchtst_model
```

### 使用工廠創建模型

#### 創建 sklearn 版本的 PatchTST

```python
model = ModelFactory.create_model(
    'patchtst_sklearn',
    seq_len=100,
    pred_len=10,
    n_estimators=50
)
```

#### 使用便捷函數創建模型

```python
model = create_patchtst_model(
    seq_len=100,
    pred_len=10,
    patch_len=20,
    stride=10
)
```

### 支援的模型類型

| 模型名稱 | 說明 |
|----------|------|
| `'patchtst_sklearn'` | 基於 sklearn 的 PatchTST（推薦） |
| `'patchtst_transformer'` | 基於 Transformer 的 PatchTST（實驗性） |

### 範例

```python
from src.currency_predictor.models.factory import ModelFactory

# 創建不同類型的模型
sklearn_model = ModelFactory.create_model(
    'patchtst_sklearn',
    seq_len=50,
    pred_len=5
)

# 如果需要 Transformer 版本（需要 GPU）
# transformer_model = ModelFactory.create_model(
#     'patchtst_transformer',
#     seq_len=50,
#     pred_len=5
# )

# 訓練模型
sklearn_model.fit(X_train, y_train)

# 預測
predictions = sklearn_model.predict(X_test)
```

---

## 基礎模型介面

### 概述

所有模型都繼承自 `BaseModel` 抽象基類，提供統一的介面。

### 必須實現的方法

所有自定義模型必須實現以下方法：

#### `fit(X, y, validation_data=None)`

訓練模型。

**參數：**
- `X`: pd.DataFrame - 訓練特徵
- `y`: pd.Series - 訓練目標
- `validation_data`: Optional[Tuple] - 驗證資料 (X_val, y_val)

**返回：** self

#### `predict(X, horizon=1)`

進行預測。

**參數：**
- `X`: pd.DataFrame - 輸入特徵
- `horizon`: int - 預測範圍

**返回：** np.ndarray - 預測結果

#### `predict_with_uncertainty(X, horizon=1, confidence_level=0.95)`

進行帶不確定性的預測。

**參數：**
- `X`: pd.DataFrame - 輸入特徵
- `horizon`: int - 預測範圍
- `confidence_level`: float - 信賴水準

**返回：** Dict[str, np.ndarray] - 包含預測值和不確定性的字典

### 創建自定義模型

```python
from src.currency_predictor.models.base import BaseModel, ModelType
import numpy as np

class MyCustomModel(BaseModel):
    """自定義預測模型"""

    def __init__(self, **kwargs):
        super().__init__(
            model_name="MyCustomModel",
            model_type=ModelType.SKLEARN_BASED
        )
        # 您的初始化邏輯

    def fit(self, X, y, validation_data=None):
        # 訓練邏輯
        self.is_fitted = True
        return self

    def predict(self, X, horizon=1):
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        # 預測邏輯
        return np.zeros(len(X))

    def predict_with_uncertainty(self, X, horizon=1, confidence_level=0.95):
        predictions = self.predict(X, horizon)
        return {
            'predictions': predictions,
            'std': np.ones_like(predictions) * 0.1,
            'lower_bound': predictions - 0.2,
            'upper_bound': predictions + 0.2
        }

# 使用自定義模型
model = MyCustomModel()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 最佳實踐

### 1. 選擇合適的參數

```python
# 對於日資料（每日匯率）
model = PatchTST(
    seq_len=60,      # 使用過去60天
    pred_len=7,      # 預測未來7天
    patch_len=10,    # 10天一個 patch
    stride=5         # 5天步長
)

# 對於小時資料
model = PatchTST(
    seq_len=168,     # 一週的小時數
    pred_len=24,     # 預測一天
    patch_len=12,    # 半天一個 patch
    stride=6         # 6小時步長
)
```

### 2. 交叉驗證

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = PatchTST(seq_len=50, pred_len=5)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val[-60:])  # 使用最後60筆
    score = mean_absolute_error(y_val[-len(val_pred):], val_pred)
    scores.append(score)

print(f"平均 MAE: {np.mean(scores):.4f}")
```

### 3. 超參數調優

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'seq_len': [50, 100, 150],
    'patch_len': [10, 15, 20],
    'n_estimators': [50, 100, 150]
}

# 注意：需要自己實現 GridSearchCV 兼容的包裝器
# 這裡僅為示意
```

---

## 常見問題

### Q: 如何選擇 seq_len 和 pred_len？

A:
- `seq_len`：應該足夠長以捕捉重要的歷史模式，但不要太長導致訓練緩慢。建議為預測目標的 10-20 倍。
- `pred_len`：根據實際需求決定，通常為 1-30 個時間步。

### Q: 模型預測效果不好怎麼辦？

A:
1. 增加訓練資料量
2. 調整 patch_len 和 stride 參數
3. 增加更多技術指標作為特徵
4. 增加 n_estimators 和 max_depth
5. 嘗試不同的預處理方法

### Q: 訓練速度太慢怎麼辦？

A:
1. 減少 n_estimators
2. 減少 max_depth
3. 使用更少的訓練資料
4. 減少特徵數量

---

## 相關文檔

- [資料模組使用說明](./data_modules.md)
- [預測管道使用說明](./prediction_pipeline.md)
- [模型架構設計](../architectures/model_design.md)
