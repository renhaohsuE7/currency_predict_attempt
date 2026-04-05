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

### 6. Use Case 測試（真實 fixture data）

#### Save/Load/Predict Cycle (`test_use_case_save_load_predict.py`)

**Marker**: `@pytest.mark.slow`
**測試類別**: `TestSaveLoadPredictCycle`

**測試項目：**

- ✅ `test_save_load_predict_consistency` - fit → predict → save → load → predict 結果一致
- ✅ `test_load_model_predict_with_extra_columns` - load 後 predict 自動過濾多餘欄位
- ✅ `test_save_load_preserves_training_history` - save/load 保留 training_history
- ✅ `test_save_load_preserves_feature_columns` - save/load 保留 _feature_columns
- ✅ `test_load_nonexistent_file_returns_false` - 載入不存在路徑回傳 False

**測試數量：** 5 個測試

#### CurrencyPredictor Full Flow (`test_use_case_currency_predictor.py`)

**Marker**: `@pytest.mark.slow`
**測試類別**: `TestCurrencyPredictorFlow`

**測試項目：**

- ✅ `test_prepare_training_data_shapes` - prepare_training_data 回傳正確 shape
- ✅ `test_train_model_returns_metrics` - train_model 回傳 training_completed=True
- ✅ `test_train_then_predict` - train → predict 產出 finite predictions
- ✅ `test_predict_without_training_returns_error` - 未訓練就 predict 回傳 error
- ✅ `test_evaluate_model_produces_metrics` - evaluate 回傳非負 metrics
- ✅ `test_save_and_reload_model` - train → save → load → predict 一致

**測試數量：** 6 個測試

#### Model Comparison (`test_use_case_model_comparison.py`)

**Marker**: `@pytest.mark.slow`
**測試類別**: `TestModelComparison`

**測試項目：**

- ✅ `test_two_sklearn_configs_comparison` - 兩個 sklearn config 比較產出有效結構
- ✅ `test_comparison_returns_valid_ranking` - overall_ranking 非空且 RMSE 遞增
- ✅ `test_comparison_best_model_per_symbol` - 每個 symbol 有 best_model
- ✅ `test_comparison_all_models_have_metrics` - 每個 model 有 test_metrics
- ✅ `test_comparison_report_generation` - 產出有效 Markdown 報告

**測試數量：** 5 個測試

---

### 7. E2E 測試（CLI 和 Error Recovery）

#### CLI Entry Point (`test_e2e_cli.py`)

**測試類別**: `TestArgParsing`, `TestModeRouting`

**測試項目：**

- ✅ `test_default_args` - argparse 預設值正確
- ✅ `test_visualize_flag` - `-v` 設為 True
- ✅ `test_compare_flag` - `--compare` 設為 True
- ✅ `test_models_parsing` - `--models sklearn,huggingface` 正確解析
- ✅ `test_symbols_parsing` - `--symbols USDTWD=X,AAPL` 正確解析
- ✅ `test_single_mode_called` - 無 compare → _run_single_mode 被呼叫
- ✅ `test_compare_mode_called` - --compare → _run_compare_mode 被呼叫

**測試數量：** 7 個測試

#### Error Recovery (`test_e2e_error_recovery.py`)

**測試類別**: `TestCollectErrorRecovery`, `TestTrainPredictErrorRecovery`, `TestComparerErrorRecovery`

**測試項目：**

- ✅ `test_collect_partial_failure` - 2 symbols, 1 fails → 另一個仍成功
- ✅ `test_invalid_symbol_graceful` - 不存在 symbol → 不 crash
- ✅ `test_train_failure_returns_error_dict` - 訓練失敗 → error dict
- ✅ `test_predict_unfitted_model_error` - 未訓練 → predict 回傳 error
- ✅ `test_comparer_one_model_fails` - 2 models, 1 fails → 另一個仍有結果

**測試數量：** 5 個測試

---

## 測試統計

### 總測試數量

| 層級 | 測試文件 | 測試數量 | Marker |
| ---- | ------- | ------- | ------ |
| Unit | test_config_manager, test_data_storage, test_data_processor, test_data_collectors, test_models_base, test_models_patchtst, test_model_factory, test_currency_predictor, test_prediction_pipeline, test_model_comparer, test_utils, test_formatter, test_visualizer | ~280 | 無 |
| Integration (slow) | test_e2e_offline, test_patchtst_huggingface, test_patchtst_lightning | ~14 | `@slow` |
| Use Case (slow) | test_use_case_save_load_predict, test_use_case_currency_predictor, test_use_case_model_comparison | 16 | `@slow` |
| E2E (fast) | test_e2e_cli, test_e2e_error_recovery | 12 | 無 |
| Online E2E | test_e2e_online | 4 | `@e2e` |
| **總計** | **~20 files** | **407** | |

### 測試覆蓋的功能模組

✅ **已測試：**

- DataStorage（資料儲存）
- DataProcessor（資料處理）
- YahooFinanceCollector（資料收集）
- BaseModel（基礎模型）
- TimeSeriesModel（時間序列模型）
- PatchTST sklearn（sklearn 版模型）
- PatchTST HuggingFace（Transformers 版模型）
- PatchTST Lightning（PyTorch Lightning 版模型）
- ModelFactory（模型工廠）
- CurrencyPredictor（預測執行器）— 含完整 train/predict/save/load 流程
- PredictionPipeline（預測管道）
- ModelComparer（模型比較器）— 含真實多模型比較
- ResultFormatter（報告格式化）
- CurrencyVisualizer（視覺化）
- ConfigManager（設定管理）
- CLI Entry Point（main.py argparse + mode routing）
- Error Recovery（partial failure / graceful degradation）
- utils（工具函數）

---

## 運行測試

### 快速測試（預設，339 tests）

```bash
uv run pytest
```

### Offline E2E + Use Case（@slow，30 tests）

```bash
uv run pytest -m slow -v
```

### Online E2E（需要網路，4 tests）

```bash
uv run pytest -m e2e -v
```

### 全部測試（排除 HuggingFace，373 tests）

```bash
uv run pytest --override-ini="addopts=" --ignore=tests/test_patchtst_huggingface.py
```

### 全部測試含 HuggingFace（407 tests，需 GPU + 網路）

```bash
uv run pytest --override-ini="addopts="
```

### 特定層級

```bash
# 只跑 use case tests
uv run pytest tests/test_use_case_*.py -v

# 只跑 e2e tests（CLI + error recovery）
uv run pytest tests/test_e2e_cli.py tests/test_e2e_error_recovery.py -v

# 資料模組測試
uv run pytest tests/test_data_*.py -v

# 模型模組測試
uv run pytest tests/test_models_*.py -v
```

### 查看測試覆蓋率

```bash
uv run pytest --cov=currency_predictor --cov-report=html
```

---

## 測試質量評估

### 優點

1. **全面的單元測試** - 每個核心類別都有對應的測試
2. **邊界情況覆蓋** - 測試包含異常情況和邊界條件
3. **隔離性好** - 使用 fixtures 和臨時目錄，測試互不影響
4. **可重現性** - 使用固定隨機種子，測試結果可重現
5. **Use Case 覆蓋** - 完整業務流程（train → predict → save → load）使用真實 fixture data 驗證
6. **Error Recovery** - partial failure 和 graceful degradation 測試確保穩健性
7. **CLI 覆蓋** - main.py entry point 有完整的 argparse + mode routing 測試

---

## 文檔資源

### 相關文檔

- [測試策略計畫](../plans/2026-04-03-0080-testing-validation-strategy.md)
- [Use Case / E2E Test Plans](../plans/testing/)
- [資料模組使用說明](../usage_descriptions/data_modules.md)
- [模型模組使用說明](../usage_descriptions/model_modules.md)
- [系統架構文檔](../architectures/system_architecture.md)

### 測試規範

所有測試都遵循以下規範：

- 位於 `/tests/` 目錄
- 使用 pytest 框架
- 測試文件命名：`test_<module_name>.py`
- 測試函數命名：`test_<function_description>`
- 使用 fixtures 提供測試資料（`conftest.py` 共用 fixtures）
- Marker 分層：`@slow`（需真實訓練）、`@e2e`（需網路）
- 預設 `uv run pytest` 只跑快速測試

---

**最後更新：** 2026-04-03
