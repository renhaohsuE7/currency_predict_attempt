# HuggingFace PatchTST 整合

- **Date**: 2026-04-03 00:40
- **Status**: completed
- **Module**: models/patchtst/huggingface/
- **前置依賴**: plan #0030 (P0 selection → HuggingFace)

## 目標

讓 HuggingFace PatchTST 完整可用：修復已知 bug、寫 integration test、確認端對端 pipeline 可運行。

## 現況分析

### 依賴狀態

- `transformers 4.56.2` ✓ 已安裝
- `torch 2.8.0+cu128` ✓ 已安裝，CUDA 可用
- `PatchTSTHuggingFace` ✓ 可 import 並 instantiate

### 已知 Bug

1. **predict() output shape 錯誤**
   - HuggingFace 輸出 shape: `(batch, prediction_length, num_channels)` = `(1, 7, 1)`
   - 現有 code `mean(dim=1)` 錯誤地 average 掉 prediction_length → 回傳 `(1,)` 而非 `(7,)`
   - 修正：squeeze channel dim 而非 mean across predictions

2. **predict_with_uncertainty() 可能有相同問題**
   - 需確認 output shape 處理邏輯

### 已實作（不需改動）

- `__init__`, `_setup_model`, `_create_sequences`, `_prepare_data`
- `fit()` with HuggingFace Trainer + early stopping
- `save_model()` / `load_model()`
- `evaluate()`
- `get_model_info()`

## 實作步驟

1. **修復 `predict()` output shape** — squeeze channel dim instead of mean(dim=1)
2. **修復 `predict_with_uncertainty()` shape** — 同上邏輯
3. **寫 `test_patchtst_huggingface.py`** — fit/predict/save/load/uncertainty
4. **跑 `uv run pytest`** 確認 0 failures
5. **更新 plan #0030 status**

## 風險評估

- GPU 記憶體不足時 HuggingFace Trainer 可能 OOM → fit() 有 try/except
- CPU fallback 會很慢但功能正常

## 執行結果（2026-04-03）

### Bug 修復

1. **predict()**: `predictions.mean(dim=1)` → `predictions.squeeze(0).squeeze(-1)`
   - HuggingFace 輸出 `(batch, pred_len, channels)` 非 `(batch, samples, pred_len)`
   - 修正後 shape: `(1,)` → `(7,)` ✓

2. **predict_with_uncertainty()**: 改用 Monte Carlo dropout sampling (30 samples)
   - 原邏輯假設輸出有多個 samples dim，實際只有 1 channel
   - 改為 `model.train()` 啟用 dropout → 多次 forward → 計算 mean/std

### Integration Test

新增 `test_patchtst_huggingface.py` — 19 tests:

- Init (4), Fit (2), Predict (5), Uncertainty (4), Save/Load (2), Evaluate (1), ModelInfo (1)

### 測試結果

- **238 passed, 0 failed**
- 覆蓋率：83% → **85%**
- `huggingface/model.py`: 74% → **87%**

## 完成標準

- [x] predict() 回傳 shape `(prediction_length,)` 正確
- [x] predict_with_uncertainty() 回傳正確的 upper/lower bounds
- [x] Integration test 通過：fit → predict → save → load → predict
- [x] `uv run pytest` 0 failures (**238 passed**)
