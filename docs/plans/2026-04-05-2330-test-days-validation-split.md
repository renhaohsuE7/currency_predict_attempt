# test_days 固定天數 + validation_split 統一

- **Date**: 2026-04-05 23:30
- **Status**: completed
- **Module**: predictor, comparer, pipeline, settings, HF model, Lightning model

## 目標

1. 將 train/test split 從百分比 (`test_size=0.2`) 改為固定天數 (`test_days`)
2. 統一 HF/Lightning 的 validation_split 讀 config 而非寫死
3. 清理死設定 `preprocessing.test_size`

## Context

目前 `prepare_training_data()` 使用百分比 `test_size=0.2` 切分 train/test，且 `config.json` 的 `preprocessing.test_size` 從未被讀取（死設定）。百分比切分在不同資料量下表現差異大（3y→150天太多、10y→500天荒謬）。

另外 HuggingFace 和 Lightning 模型的 validation split 寫死 `0.2`，沒讀 `TrainingConfig.validation_split`。

## 設計

### 1. `test_size` (百分比) → `test_days` (固定天數)

- `prepare_training_data(test_days: Optional[int] = None)` 取代 `test_size: float = 0.2`
- 預設公式：`max(2 * prediction_horizon, 30)`
- 安全上限：不超過資料總量的 50%
- Config 新增 `model_training.test_days`（`Optional[int]`，`None` = 用預設公式）

### 2. validation_split 統一

HF `model.py:587` 和 Lightning `wrapper.py:438` 的 hardcoded `0.2` 改讀 `effective.validation_split`（`effective` 變數已存在於兩者的 `fit()` 中）。

### 3. 清理死 config

移除 `config.json` 的 `preprocessing` 區塊，更新引用它的 `helpers.py` 和測試。

## 影響範圍

| 檔案 | 修改 |
| --- | --- |
| `src/currency_predictor/config/settings.py` | `ModelTrainingConfig` 加 `test_days: Optional[int]` |
| `src/currency_predictor/prediction/predictor.py` | `prepare_training_data()` 改 `test_days`；`train_model()` 加 `test_days` 參數 |
| `src/currency_predictor/prediction/comparer.py` | `compare()`、`train()`、`predict()` 計算 effective_test_days 並傳遞 |
| `src/currency_predictor/prediction/pipeline.py` | `_training_phase()` 計算 effective_test_days 並傳遞 |
| `src/currency_predictor/models/patchtst/huggingface/model.py` | `fit()` line 587: `0.2` → `effective.validation_split` |
| `src/currency_predictor/models/patchtst/lightning/wrapper.py` | `fit()` line 438: `0.2` → `effective.validation_split` |
| `config.json` | 加 `model_training.test_days: 30`，移除 `preprocessing` |
| `config_example.json` | 加 `model_training.test_days: null` |
| `tw2330_config.json` | 加 `model_training.test_days: 30` |
| `src/currency_predictor/utils/helpers.py` | `get_default_config()` 移除 `preprocessing.test_size` |
| `tests/test_utils.py` | 更新 `preprocessing` 相關斷言 |
| `tests/test_e2e_pipeline_output.py` | 移除 test config 中的 `preprocessing` |

## 實作步驟

1. **settings.py** — `ModelTrainingConfig` 加 `test_days: Optional[int] = Field(None, ge=1)`
2. **predictor.py** — `prepare_training_data(test_days=None)` 取代 `test_size=0.2`；`train_model()` 新增 `test_days` 參數並轉傳
3. **comparer.py** — 三個方法（`compare`/`train`/`predict`）加 `effective_test_days` 計算：
   ```python
   configured = training_config.get('test_days')
   effective_test_days = configured or max(2 * prediction_horizon, 30)
   ```
4. **pipeline.py** — `_training_phase()` 同理
5. **huggingface/model.py** — line 587: `int(0.2 * len(train_past))` → `int(effective.validation_split * len(train_past))`
6. **lightning/wrapper.py** — line 438: 同上
7. Config 檔案更新
8. 死 code 清理（`helpers.py` 移除 `test_size`、tests 更新）
9. 跑全部測試

## 完成標準

- 全部測試通過
- `prepare_training_data(test_days=30)` 確實產生 30 筆 test data
- HF/Lightning validation split 讀 config 不再寫死
- `preprocessing` 區塊已從 config.json 移除
