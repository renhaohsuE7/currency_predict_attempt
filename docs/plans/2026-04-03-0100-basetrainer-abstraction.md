# Plan #0100: Unified TrainingConfig + Training History

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: models/, prediction/
- **Priority**: P1

## 目標

統一三種 PatchTST 實作的訓練參數介面，使所有模型一致接受 `TrainingConfig`，並簡化 `CurrencyPredictor.train_model()` 中的驗證分割邏輯。

## 設計決策

| 問題 | 決策 |
| --- | --- |
| 是否建立 BaseTrainer 類別 | **不建立** — sklearn/HF/Lightning 訓練機制完全不同，額外包裝只增加複雜度 |
| TrainingConfig 放哪 | 複用已存在的 `models/patchtst/config.py` 中的 `TrainingConfig` |
| 向後相容 | 新參數 `training_config=None` 為 optional，既有 `model.fit(X, y)` 不受影響 |
| 參數優先順序 | explicit kwarg > training_config > model defaults (fine_tune_mode) |

## 已完成工作

### Part 1: sklearn training history

- 新增 `self.training_history = {'train_loss': [], 'eval_loss': []}` 到 `__init__`
- 提取 `_extract_features_from_data(X, y)` helper（序列建立 + patch 特徵提取）
- `fit()` 訓練後計算 train MSE → `training_history['train_loss']`
- 若有 validation_data，計算 eval MSE → `training_history['eval_loss']`
- `save_model()` / `load_model()` 包含 training_history

### Part 2: 統一 `training_config` 參數

- `BaseModel.fit()` 簽名加入 `training_config: Optional[Any] = None`
- `PatchTSTSklearn.fit()`: 接受 `training_config`，用 `validation_split` 自動切分
- `PatchTSTHuggingFace.fit()`: 接受 `training_config`，取代 hardcode 的 `weight_decay=0.01` 和 `warmup_ratio=0.1`
- `PatchTSTLightningWrapper.fit()`: 接受 `training_config`，用作 effective config 來源

### Part 3: 簡化 CurrencyPredictor.train_model()

- 25 行 `validation_split → validation_data` 手動轉換邏輯 → 5 行，委託模型處理分割

### 測試

新增 7 tests：

| 檔案 | Tests | 驗證內容 |
| --- | --- | --- |
| `test_models_patchtst.py` | 5 | training_config, training_history, auto validation_split, backward compat |
| `test_patchtst_huggingface.py` | 1 | training_config acceptance |
| `test_patchtst_lightning.py` | 1 | training_config acceptance |

### 測試結果

- **280 passed**（不含 pre-existing HuggingFace PermissionError）
- **25 passed**（Lightning tests 含 1 新增）
- Pre-existing: HuggingFace checkpoint PermissionError（非本次引入）

## 完成標準

- [x] TrainingConfig 已可用（複用 `models/patchtst/config.py` 中既有的）
- [x] 所有 model 的 fit() 接受 TrainingConfig（向後相容）
- [x] sklearn training_history 有值（train_loss + eval_loss）
- [x] CurrencyPredictor.train_model() 簡化（5 行取代 25 行）
- [x] 全部既有測試通過（280+）
- [x] Plan doc 更新
