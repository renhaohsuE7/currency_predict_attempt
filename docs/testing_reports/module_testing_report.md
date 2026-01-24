# 模組測試報告

**報告日期：** 2026-01-24
**專案：** Currency Predictor
**測試框架：** pytest

---

## 執行摘要

本報告記錄了為專案核心模組創建的單元測試和整合測試。所有測試都已創建並組織在 `/tests/` 目錄下，遵循專案開發規範。

---

## 測試覆蓋範圍

### 1. 資料模組測試

#### 資料儲存測試 (`test_data_storage.py`)

**測試類別：** `TestDataStorage`

**測試項目：**
- ✅ `test_init` - 測試初始化和目錄創建
- ✅ `test_save_and_load_raw_data` - 測試儲存和載入原始資料
- ✅ `test_save_and_load_processed_data` - 測試儲存和載入處理後資料
- ✅ `test_list_available_data` - 測試列出可用資料
- ✅ `test_data_exists` - 測試檢查資料是否存在
- ✅ `test_get_data_info` - 測試取得資料資訊
- ✅ `test_delete_data` - 測試刪除資料
- ✅ `test_load_nonexistent_data` - 測試載入不存在的資料
- ✅ `test_save_invalid_data` - 測試儲存無效資料

**測試數量：** 9 個測試
**覆蓋功能：** 資料儲存、載入、管理的核心功能

#### 資料處理測試 (`test_data_processor.py`)

**測試類別：** `TestDataProcessor`

**測試項目：**
- ✅ `test_init` - 測試初始化
- ✅ `test_clean_data` - 測試資料清理
- ✅ `test_create_technical_indicators` - 測試技術指標創建
- ✅ `test_create_lagged_features` - 測試滯後特徵創建
- ✅ `test_prepare_features_target` - 測試準備特徵和目標變數
- ✅ `test_scale_features` - 測試特徵縮放
- ✅ `test_train_test_split` - 測試訓練測試集分割
- ✅ `test_empty_dataframe` - 測試處理空 DataFrame
- ✅ `test_insufficient_data_for_indicators` - 測試資料不足時的處理

**測試數量：** 9 個測試
**覆蓋功能：** 資料清理、特徵工程、技術指標計算

#### 資料收集器測試 (`test_data_collectors.py`)

**狀態：** 已存在（原有測試）
**覆蓋功能：** Yahoo Finance 資料收集

---

### 2. 模型模組測試

#### 基礎模型測試 (`test_models_base.py`)

**測試類別：**
- `TestModelType` - 測試模型類型枚舉
- `TestBaseModel` - 測試基礎模型抽象類
- `TestTimeSeriesModel` - 測試時間序列模型基類
- `TestSklearnBasedModel` - 測試 sklearn 基礎模型類

**測試項目：**
- ✅ `test_model_types_exist` - 測試模型類型存在
- ✅ `test_model_type_values` - 測試模型類型值
- ✅ `test_init` - 測試初始化
- ✅ `test_fit_and_predict` - 測試訓練和預測
- ✅ `test_predict_without_fit_raises_error` - 測試未訓練前預測會報錯
- ✅ `test_get_model_info` - 測試取得模型資訊
- ✅ `test_predict_with_uncertainty` - 測試帶不確定性的預測
- ✅ `test_prepare_sequences` - 測試準備時間序列資料
- ✅ `test_validate_input_shape` - 測試驗證輸入資料形狀

**測試數量：** 12+ 個測試
**覆蓋功能：** 模型基礎介面、抽象方法、繼承體系

#### PatchTST 模型測試 (`test_models_patchtst.py`)

**測試類別：** `TestPatchTST`

**測試項目：**
- ✅ `test_init_default_params` - 測試使用默認參數初始化
- ✅ `test_init_custom_params` - 測試使用自定義參數初始化
- ✅ `test_fit` - 測試模型訓練
- ✅ `test_fit_with_validation_data` - 測試使用驗證資料訓練
- ✅ `test_predict_raises_error_when_not_fitted` - 測試未訓練時預測會報錯
- ✅ `test_predict_raises_error_with_insufficient_data` - 測試資料不足時預測會報錯
- ✅ `test_get_model_info` - 測試取得模型資訊
- ✅ `test_create_patches` - 測試 patch 創建
- ✅ `test_extract_patch_features` - 測試 patch 特徵提取
- ✅ `test_model_reproducibility` - 測試模型的可重現性
- ✅ `test_save_and_load_model` - 測試模型儲存和載入

**測試數量：** 11 個測試
**覆蓋功能：** PatchTST 模型的核心功能

---

### 3. 預測模組測試

#### 預測執行器測試 (`test_currency_predictor.py`)

**狀態：** 已存在（原有測試）
**覆蓋功能：** CurrencyPredictor 類別

#### 預測管道測試 (`test_prediction_pipeline.py`)

**測試類別：**
- `TestPredictionPipeline` - 測試預測管道
- `TestPredictionPipelineIntegration` - 測試整合功能

**測試項目：**
- ✅ `test_init` - 測試初始化
- ✅ `test_init_creates_output_dir` - 測試初始化會創建輸出目錄
- ✅ `test_validate_config_valid` - 測試驗證有效配置
- ✅ `test_validate_config_missing_keys` - 測試驗證缺少必要鍵的配置
- ✅ `test_collect_data_stage` - 測試資料收集階段
- ✅ `test_save_results` - 測試儲存結果
- ✅ `test_generate_report` - 測試生成報告
- ⏸️ `test_run_full_pipeline` - 測試運行完整管道（標記為跳過）

**測試數量：** 7 個測試（1 個跳過）
**覆蓋功能：** 預測管道的配置、執行、結果儲存

---

### 4. 工具模組測試

#### 工具函數測試 (`test_utils.py`)

**測試類別：** `TestLogging`

**測試項目：**
- ✅ `test_setup_logging_default` - 測試使用默認參數設置日誌
- ✅ `test_setup_logging_custom_level` - 測試使用自定義日誌級別
- ✅ `test_setup_logging_warning_level` - 測試使用 WARNING 級別
- ✅ `test_setup_logging_with_file` - 測試寫入文件的日誌設置
- ✅ `test_logging_output` - 測試日誌輸出

**測試數量：** 5 個測試
**覆蓋功能：** 日誌設置和輸出

---

### 5. 整合測試和系統測試

#### 位於 `/tests/` 目錄的整合測試

以下測試文件已從根目錄移動至 `/tests/`：

1. **`test_minimal.py`** - 簡化版功能測試
   - 測試最小功能集合
   - 使用簡化參數進行測試

2. **`test_prediction_system.py`** - 預測系統測試
   - 測試基本功能
   - 測試完整預測流程

3. **`test_architecture_analysis.py`** - 架構分析測試
   - 測試架構設計
   - 驗證模組間協作

4. **`test_new_architecture.py`** - 新架構測試
   - 測試新架構實現
   - 驗證重構後的模組

---

## 測試統計

### 總測試數量

| 模組 | 測試文件數 | 測試數量（估計） |
|------|-----------|----------------|
| 資料模組 | 3 | 20+ |
| 模型模組 | 2 | 23+ |
| 預測模組 | 2 | 7+ |
| 工具模組 | 1 | 5+ |
| 整合測試 | 4 | 未統計 |
| **總計** | **12** | **55+** |

### 測試覆蓋的功能模組

✅ **已測試：**
- DataStorage（資料儲存）
- DataProcessor（資料處理）
- YahooFinanceCollector（資料收集）
- BaseModel（基礎模型）
- TimeSeriesModel（時間序列模型）
- PatchTST（PatchTST 模型）
- CurrencyPredictor（預測執行器）
- PredictionPipeline（預測管道）
- utils（工具函數）

⚠️ **待測試：**
- ModelFactory（模型工廠） - 部分測試在整合測試中
- Transformer 版本的 PatchTST - 實驗性功能

---

## 運行測試

### 運行所有測試

```bash
pytest tests/ -v
```

### 運行特定模組的測試

```bash
# 資料模組測試
pytest tests/test_data_*.py -v

# 模型模組測試
pytest tests/test_models_*.py -v

# 預測模組測試
pytest tests/test_prediction_*.py -v
```

### 運行單一測試文件

```bash
pytest tests/test_data_storage.py -v
```

### 查看測試覆蓋率

```bash
pytest tests/ --cov=src/currency_predictor --cov-report=html
```

---

## 測試質量評估

### 優點

1. **全面的單元測試** - 每個核心類別都有對應的測試
2. **邊界情況覆蓋** - 測試包含異常情況和邊界條件
3. **隔離性好** - 使用 fixtures 和臨時目錄，測試互不影響
4. **可重現性** - 使用固定隨機種子，測試結果可重現

### 待改進

1. **整合測試** - 需要更多端到端的整合測試
2. **性能測試** - 缺少性能和壓力測試
3. **Mock 使用** - 某些測試需要 mock 外部 API 調用
4. **測試覆蓋率** - 需要測量並提高代碼覆蓋率

---

## 建議的下一步

### 短期 (1-2 週)

1. 運行所有測試並修復失敗的測試
2. 添加 ModelFactory 的單元測試
3. 為整合測試添加 mock，避免依賴外部 API
4. 測量代碼覆蓋率並設定目標（例如 80%）

### 中期 (1 個月)

1. 添加性能測試和基準測試
2. 實現持續整合（CI）自動運行測試
3. 添加更多邊界情況和異常處理的測試
4. 為 Transformer 版本模型添加測試

### 長期 (持續)

1. 保持測試覆蓋率 >80%
2. 定期更新測試以反映新功能
3. 建立測試最佳實踐文檔
4. 添加回歸測試防止 bug 復現

---

## 文檔資源

### 相關文檔

- [專案開發規範](../../.claude/project_guidelines.md)
- [資料模組使用說明](../usage_descriptions/data_modules.md)
- [模型模組使用說明](../usage_descriptions/model_modules.md)
- [系統架構文檔](../architectures/system_architecture.md)

### 測試規範

所有測試都遵循以下規範：
- 位於 `/tests/` 目錄
- 使用 pytest 框架
- 測試文件命名：`test_<module_name>.py`
- 測試函數命名：`test_<function_description>`
- 使用 fixtures 提供測試資料
- 清理測試產生的臨時文件

---

**報告結束**

如有問題或建議，請聯繫開發團隊。
