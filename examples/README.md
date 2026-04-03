# Currency Predictor - 使用範例

本目錄包含 Currency Predictor 的各種使用範例，從基本使用到進階功能。

---

## 📚 範例清單

### 1. 基本使用範例 (basic_usage.py)

**適合：** 初學者，快速開始

**說明：** 展示如何使用 Currency Predictor 進行基本的貨幣預測

**使用的 API：**
- `ConfigManager` - 配置管理
- `PredictionPipeline` - 預測管道
- `ResultFormatter` - 結果格式化

**執行：**
```bash
uv run python examples/basic_usage.py
```

**學習要點：**
- 如何載入配置
- 如何創建預測管道
- 如何運行完整的預測流程
- 如何格式化和顯示結果

---

### 2. 自定義配置範例 (custom_config_usage.py)

**適合：** 需要自定義參數的使用者

**說明：** 展示如何使用自定義配置進行貨幣預測

**使用的 API：**
- `ConfigManager` - 配置管理
- `PredictionPipeline` - 預測管道
- 自定義配置字典

**執行：**
```bash
uv run python examples/custom_config_usage.py
```

**學習要點：**
- 如何創建自定義配置
- 如何調整模型參數
- 如何設置資料收集參數
- 如何儲存配置供後續使用

**自定義參數範例：**
```python
{
    "model_params": {
        "seq_len": 60,       # 序列長度
        "pred_len": 7,       # 預測長度
        "n_estimators": 50,  # 估計器數量
        "max_depth": 5       # 最大深度
    },
    "data_collection": {
        "period": "6mo",     # 資料期間
        "interval": "1d"     # 資料間隔
    }
}
```

---

### 3. 多貨幣預測範例 (multi_currency_prediction.py)

**適合：** 需要同時預測多個貨幣對的使用者

**說明：** 展示如何同時預測多個貨幣對並比較結果

**使用的 API：**
- `ConfigManager` - 配置管理
- `PredictionPipeline` - 預測管道
- `ResultFormatter` - 結果格式化

**執行：**
```bash
uv run python examples/multi_currency_prediction.py
```

**學習要點：**
- 如何配置多個貨幣對
- 如何批次處理預測
- 如何比較不同貨幣對的預測結果
- 如何生成綜合報告

---

### 4. 單一貨幣預測範例 (single_currency_prediction.py)

**適合：** 需要精細控制預測流程的使用者

**說明：** 展示如何使用 CurrencyPredictor 類別進行低階的貨幣預測

**使用的 API：**
- `CurrencyPredictor` - 低階預測 API

**執行：**
```bash
uv run python examples/single_currency_prediction.py
```

**學習要點：**
- 如何使用低階 API
- 如何分步驟執行：資料收集 → 訓練 → 預測 → 儲存
- 如何訪問中間結果和指標
- 如何儲存和載入模型

**流程：**
```python
# 1. 創建預測器
predictor = CurrencyPredictor(model_name="patchtst_sklearn", model_params={...})

# 2. 收集資料
data_result = predictor.collect_and_store_data([symbol], period="6mo")

# 3. 訓練模型
training_result = predictor.train_model(symbol, period="6mo")

# 4. 進行預測
prediction_result = predictor.predict(symbol, horizon=7)

# 5. 儲存模型
predictor.save_model(model_path)
```

---

### 5. 批次預測範例 (batch_prediction.py)

**適合：** 已有訓練好的模型，需要快速預測的使用者

**說明：** 展示如何使用已訓練的模型進行批次預測

**使用的 API：**
- `PredictionPipeline` - 預測管道（批次模式）

**執行：**
```bash
uv run python examples/batch_prediction.py
```

**前置條件：**
- 需要已訓練的模型文件（可以先執行 `single_currency_prediction.py`）

**學習要點：**
- 如何重用已訓練的模型
- 如何批次處理多個貨幣對
- 如何處理模型文件缺失的情況
- 如何快速獲取預測結果

**使用場景：**
- 定期預測（使用相同模型）
- 快速測試不同貨幣對
- 生產環境部署

---

### 6. 視覺化範例 (visualization_example.py)

**適合：** 需要視覺化分析貨幣資料的使用者

**說明：** 展示如何使用 CurrencyVisualizer 生成各種分析圖表

**使用的 API：**
- `CurrencyVisualizer` - 視覺化類別
- `YahooFinanceCollector` - 資料收集器

**執行：**
```bash
uv run python examples/visualization_example.py
```

**學習要點：**
- 如何設定中文字型
- 如何繪製價格歷史圖
- 如何繪製 K 線圖
- 如何繪製移動平均線
- 如何繪製成交量圖
- 如何繪製報酬率分布
- 如何比較多個貨幣對
- 如何創建綜合儀表板

**生成的圖表：**
- 價格歷史圖
- 價格與成交量組合圖
- 移動平均線圖
- 報酬率分布圖
- K 線圖
- 貨幣對比圖
- 預測結果圖
- 綜合分析儀表板

**使用場景：**
- 探索性資料分析
- 生成分析報告
- 視覺化預測結果
- 貨幣趨勢分析

**主要功能：**
```python
from currency_predictor.visualization import CurrencyVisualizer

# 創建視覺化器
visualizer = CurrencyVisualizer(
    data_path="data",
    output_dir="results/figures"
)

# 載入資料
df = visualizer.load_data("USDTWD=X")

# 創建儀表板
visualizer.create_dashboard(
    df=df,
    symbol="USDTWD=X",
    save_path="dashboard.png"
)
```

---

## 🎯 學習路徑

### 初學者路徑

1. **basic_usage.py** - 了解基本流程和概念
2. **custom_config_usage.py** - 學習如何調整參數
3. **multi_currency_prediction.py** - 處理多個貨幣對

### 進階路徑

1. **single_currency_prediction.py** - 深入了解低階 API
2. **batch_prediction.py** - 學習模型重用和批次處理

### 實際應用路徑

1. 使用 **single_currency_prediction.py** 訓練模型
2. 使用 **batch_prediction.py** 進行日常預測
3. 根據需求使用 **custom_config_usage.py** 調整參數

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 使用 uv
uv sync

# 或使用 pip
pip install -e .
```

### 2. 執行第一個範例

```bash
uv run python examples/basic_usage.py
```

### 3. 查看結果

預測結果會儲存在以下目錄：
- `data/` - 收集的歷史資料
- `models/` - 訓練好的模型
- `results/` - 預測結果

---

## 📝 範例對比

| 範例 | API 層級 | 複雜度 | 靈活性 | 執行時間 | 適用場景 |
|------|---------|--------|--------|---------|----------|
| basic_usage.py | 高階 | ⭐ | ⭐⭐ | 長 | 快速開始 |
| custom_config_usage.py | 高階 | ⭐⭐ | ⭐⭐⭐ | 長 | 參數調整 |
| multi_currency_prediction.py | 高階 | ⭐⭐ | ⭐⭐ | 很長 | 多貨幣分析 |
| single_currency_prediction.py | 低階 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 長 | 深度控制 |
| batch_prediction.py | 高階 | ⭐ | ⭐⭐⭐ | 短 | 快速預測 |

**圖示說明：**
- ⭐ = 簡單/低
- ⭐⭐ = 中等
- ⭐⭐⭐ = 複雜/高
- ⭐⭐⭐⭐ = 非常高

---

## 🔧 配置說明

### 使用 ConfigManager（推薦）

```python
from currency_predictor.config.manager import ConfigManager

# 使用默認配置
config_manager = ConfigManager()
config = config_manager.get_config()

# 使用自定義配置文件
config_manager = ConfigManager(config_path="my_config.json")
```

### 使用環境變數

設置環境變數可以覆蓋配置：

```bash
# Linux/Mac
export CURRENCY_PRED_MODEL_NAME="CustomModel"
export CURRENCY_PRED_MODEL_PARAMS__SEQ_LEN=200

# Windows
set CURRENCY_PRED_MODEL_NAME=CustomModel
set CURRENCY_PRED_MODEL_PARAMS__SEQ_LEN=200
```

### 使用 .env 文件

創建 `.env` 文件：

```env
CURRENCY_PRED_MODEL_NAME=PatchTST
CURRENCY_PRED_MODEL_PARAMS__SEQ_LEN=168
CURRENCY_PRED_LOG_LEVEL=INFO
```

---

## 💡 常見問題

### Q: 執行範例時出現 ModuleNotFoundError

**A:** 確保已安裝專案：

```bash
uv pip install -e .
```

### Q: 資料收集失敗（Yahoo Finance API 錯誤）

**A:** 這通常是網路問題或 API 限制。可以：
- 檢查網路連線
- 稍後重試
- 使用較短的資料期間（例如 "3mo" 而不是 "2y"）

### Q: 訓練時間太長

**A:** 可以調整參數以加快訓練：
- 減少 `seq_len` 和 `pred_len`
- 減少 `n_estimators`
- 使用較短的資料期間

### Q: 如何使用自己的貨幣對？

**A:** 在配置中修改 `symbols` 列表：

```python
config = {
    "symbols": ["JPYTWD=X", "AUDTWD=X"]  # 使用你想要的貨幣對
}
```

### Q: batch_prediction.py 提示模型不存在

**A:** 先執行 `single_currency_prediction.py` 訓練模型：

```bash
uv run python examples/single_currency_prediction.py
```

---

## 📖 進階使用

### 自定義模型參數

```python
model_params = {
    'seq_len': 168,      # 輸入序列長度（小時）
    'pred_len': 24,      # 預測長度（小時）
    'patch_len': 16,     # Patch 長度
    'stride': 8,         # Patch 步長
    'n_estimators': 50,  # 隨機森林估計器數量
    'max_depth': 8,      # 決策樹最大深度
    'random_state': 42   # 隨機種子
}
```

### 調整資料收集參數

```python
data_collection = {
    "period": "1y",      # 資料期間：1mo, 3mo, 6mo, 1y, 2y, max
    "interval": "1d",    # 資料間隔：1m, 5m, 15m, 1h, 1d, 1wk, 1mo
    "force_update": False # 是否強制更新資料
}
```

### 使用不確定性估計

```python
prediction_result = predictor.predict(
    symbol=symbol,
    horizon=7,
    return_uncertainty=True  # 返回預測的不確定性
)

# 訪問不確定性
if 'uncertainty' in prediction_result:
    uncertainty = prediction_result['uncertainty']
    print(f"Prediction uncertainty: {uncertainty}")
```

---

## 🛠️ 疑難排解

### 常見錯誤和解決方案

1. **Import Error**
   ```
   解決方案：uv pip install -e .
   ```

2. **Data Collection Failed**
   ```
   解決方案：檢查網路，減少 period，稍後重試
   ```

3. **Model Training Failed**
   ```
   解決方案：檢查資料是否足夠，調整模型參數
   ```

4. **Prediction Failed**
   ```
   解決方案：確保模型已訓練，檢查資料是否最新
   ```

---

## 📚 相關文檔

- [配置管理使用說明](../docs/usage_descriptions/config_management.md)
- [Pydantic Settings 遷移報告](../docs/testing_reports/pydantic_settings_migration_report.md)
- [專案開發規範](../.claude/project_guidelines.md)
- [example_usage.py 重構報告](../docs/development/example_usage_refactoring.md)

---

## 💬 需要幫助？

如果遇到問題：

1. 查看相關文檔
2. 檢查常見問題
3. 查看錯誤訊息並嘗試疑難排解
4. 提交 issue 到專案 repository

---

**祝你使用愉快！Happy Predicting! 🚀📈**
