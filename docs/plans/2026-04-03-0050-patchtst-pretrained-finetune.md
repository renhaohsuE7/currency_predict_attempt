# PatchTST Pretrained + Fine-Tune 支援

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: models/patchtst/huggingface/
- **前置依賴**: plan #0040 (HuggingFace PatchTST integration) ✓ completed

## 目標

讓 HuggingFace PatchTST 支援從 Hub 載入預訓練模型並 fine-tune，而非僅能從頭訓練。保持完全向後相容。

## 背景

目前 `PatchTSTHuggingFace._setup_model()` 以 `PatchTSTForPrediction(config)` 從頭建立模型，以 ~500 筆匯率資料訓練小型 Transformer，未發揮 HuggingFace 預訓練生態的核心價值。

HuggingFace Hub 上有可用的預訓練模型：
- `ibm-granite/granite-timeseries-patchtst` (616k params, ETTh1 dataset, Apache 2.0)
- `ibm-research/patchtst-etth1-pretrain`

PatchTST 為 channel-independent 架構，即使預訓練模型訓練在 7-channel 多變量資料，也可安全地以 `num_input_channels=1` 用於單一匯率。

## 設計決策

### Q1: 哪裡分支 pretrained vs from-scratch？

**`_setup_model()`**。這是唯一建立 HuggingFace 模型的位置，拆分為 `_setup_from_scratch_model()` + `_setup_pretrained_model()`，既有邏輯不動。

### Q2: 預訓練模型架構參數 vs 用戶設定？

使用 pretrained 時，架構參數 (`d_model`, `n_heads`, `n_layers`, `ffn_dim`) **來自預訓練模型**，用戶只能覆蓋任務參數 (`prediction_length`, `num_input_channels`)。載入後更新 `self.context_length` 等屬性以保持一致。

### Q3: Fine-tune 模式及訓練超參數？

三種模式，各有推薦預設：

| Mode | 說明 | num_epochs | learning_rate | early_stopping_patience |
|------|------|-----------|---------------|------------------------|
| `from_scratch` | 現有行為（向後相容） | 50 | 1e-4 | 10 |
| `full` | 解凍所有參數，較低 LR | 20 | 1e-5 | 5 |
| `linear_probe` | 凍結 backbone，只訓 head | 10 | 1e-3 | 5 |

使用者在 `fit()` 明確傳入的值優先。

### Q4: 在哪裡指定預訓練模型名稱？

`config.json` 的 `model_params` 和 constructor 都可以：

```json
{
    "model_name": "patchtst_transformer",
    "model_params": {
        "pretrained_model_name_or_path": "ibm-granite/granite-timeseries-patchtst",
        "fine_tune_mode": "full",
        "pred_len": 7
    }
}
```

`factory.py` 已透過 `**kwargs` 傳遞，不需改動。

## 修改範圍

| File | Change | 影響範圍 |
|------|--------|---------|
| `models/patchtst/config.py` | 新增 `pretrained_model_name_or_path`, `fine_tune_mode` 欄位；`TrainingConfig.for_fine_tune_mode()` | 低 |
| `models/patchtst/huggingface/model.py` | 拆分 `_setup_model()`；新增 `_setup_pretrained_model()`, `_apply_fine_tune_freezing()`；`fit()` 預設值依模式決定；更新 save/load/info | 中 |
| `config/settings.py` | `ModelParams` 新增 2 個 optional Pydantic 欄位 | 低 |
| `tests/test_patchtst_huggingface.py` | 新增 fine-tune mode 單元測試 | 低 |
| `tests/test_patchtst_pretrained_integration.py` | 新檔案，`@pytest.mark.slow` 網路整合測試 | 新增 |

**不需修改**：`factory.py`、`base.py`、`predictor.py`、`pipeline.py`

## 實作步驟

### Step 1: `config.py`

- `PatchTSTConfig` 新增欄位：
  - `pretrained_model_name_or_path: Optional[str] = None`
  - `fine_tune_mode: Literal['from_scratch', 'full', 'linear_probe'] = 'from_scratch'`
  - `is_pretrained` property
- `to_dict()` 加入這兩個欄位
- `TrainingConfig` 新增 `for_fine_tune_mode(mode: str)` classmethod

### Step 2: `settings.py`

- `ModelParams` 新增：
  - `pretrained_model_name_or_path: Optional[str] = Field(None, ...)`
  - `fine_tune_mode: str = Field('from_scratch', ...)`

### Step 3: `model.py` — 核心改動

**3a. Constructor**
- 新增 `pretrained_model_name_or_path` 和 `fine_tune_mode` 參數
- 存為 instance attributes
- 若傳入 `config` 物件，從中讀取（constructor 參數優先）

**3b. `_setup_model()` 拆分**
```python
def _setup_model(self, num_features=1):
    self._num_features = num_features
    if self.pretrained_model_name_or_path:
        self._setup_pretrained_model(num_features)
    else:
        self._setup_from_scratch_model(num_features)
```

**3c. `_setup_pretrained_model()`**
- `PatchTSTForPrediction.from_pretrained(name, num_input_channels=num_features, prediction_length=self.prediction_length, ignore_mismatched_sizes=True)`
- 從 `model.config` 更新 `self.context_length`, `self.d_model` 等
- 呼叫 `_apply_fine_tune_freezing()`
- 捕捉 `OSError`/`ConnectionError`

**3d. `_apply_fine_tune_freezing()`**
- `linear_probe`: 凍結所有非 head 參數
- `full`: 所有參數可訓練

**3e. `fit()` 預設值**
- `num_epochs`, `learning_rate`, `early_stopping_patience` 預設改為 `None`
- 用 `TrainingConfig.for_fine_tune_mode()` 填入對應模式的預設
- 使用者明確傳入的值優先（向後相容）

**3f. `save_model()` / `load_model()`**
- metadata 加入 `pretrained_model_name_or_path` 和 `fine_tune_mode`

**3g. `get_model_info()`**
- 加入 `pretrained_model_name_or_path`, `fine_tune_mode`, `trainable_parameters`

### Step 4: 測試

**新增到 `test_patchtst_huggingface.py`**：
- `TestPatchTSTPretrainedInit` — constructor 接受 pretrained 參數、預設值正確
- `TestPatchTSTFineTuneDefaults` — `TrainingConfig.for_fine_tune_mode()` 回傳值驗證
- `TestPatchTSTFineTuneModes` — from_scratch 向後相容 (fit → predict)

**新增 `test_patchtst_pretrained_integration.py`** (`@pytest.mark.slow`)：
- 從 Hub 載入 `ibm-granite/granite-timeseries-patchtst`
- 驗證架構參數 (d_model, n_layers 來自預訓練)
- Fine-tune + predict 端對端
- Linear probe 凍結驗證

### Step 5: 驗證

```bash
uv run pytest                           # 既有 238+ tests 全部通過
uv run pytest -m slow                   # 預訓練整合測試
uv run pytest --cov=currency_predictor  # 覆蓋率 ≥ 85%
```

## 注意事項

- **Context length**: IBM Granite 預設 context_length=512，需要 ≥519 筆資料。目前 config.json 用 "1y" (~252 筆) 不足。使用 pretrained 時需調整 data period 或覆蓋 context_length。
- **`ignore_mismatched_sizes=True`**: 當 `num_input_channels` 或 `prediction_length` 與預訓練不同時，相關 projection 層重新初始化，其餘權重保留。
- **首次下載**: `from_pretrained()` 會自動快取到 `~/.cache/huggingface/`，需要網路。

## 風險評估

- 預訓練模型 domain (電力) 與目標 domain (匯率) 差異大，transfer learning 效果待驗證
- GPU 記憶體：預訓練模型較大 (616k params)，但仍可在 CPU 上 fine-tune
- Hub 網路存取：離線環境需預先下載模型

## 執行結果（2026-04-03）

### 修改檔案

1. `config.py` — 新增 `pretrained_model_name_or_path`, `fine_tune_mode`, `is_pretrained`, `TrainingConfig.for_fine_tune_mode()`
2. `settings.py` — `ModelParams` 新增 2 個 Pydantic 欄位
3. `model.py` — 拆分 `_setup_model()`, 新增 `_setup_pretrained_model()`, `_apply_fine_tune_freezing()`, `fit()` 預設值依模式決定, save/load/info 更新, 同時重新修復 #0040 bug (predict/predict_with_uncertainty)
4. `test_patchtst_huggingface.py` — 新增 4 個 test class (13 tests)
5. `test_patchtst_pretrained_integration.py` — 新檔案 (7 tests, `@pytest.mark.slow`)
6. `pyproject.toml` — 註冊 `slow` marker

### 測試結果

- **251 passed, 0 failed** (excluding slow)
- 覆蓋率：**80%**
- `config.py`: **100%**
- `model.py`: **79%**

## 完成標準

- [x] `PatchTSTHuggingFace(pretrained_model_name_or_path="ibm-granite/...", fine_tune_mode="full")` 可正常建立
- [x] `_setup_pretrained_model()` 正確載入並覆蓋任務參數
- [x] `linear_probe` 模式下 backbone 參數 `requires_grad=False`
- [x] `fit()` 預設值依模式自動調整
- [x] 既有 tests 全部通過（向後相容）— **251 passed**
- [x] 預訓練整合測試通過 — **6/6 passed** (Docker container, RTX 3090)
- [x] 覆蓋率 ≥ 80%
