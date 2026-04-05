# Plan #0090: PyTorch Lightning PatchTST

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: models/patchtst/lightning/, models/factory.py
- **Priority**: P0

## 目標

完成 PyTorch Lightning 版 PatchTST，使其可透過 `ModelFactory.create_model('patchtst_lightning')` 使用，並通過整合測試。

## 已完成工作

### Bug 修復（8 個）

#### wrapper.py

| Bug | 嚴重度 | 說明 |
| --- | --- | --- |
| A | Critical | `predict()` 缺少 scaler transform/inverse_transform → 重寫為 extract → scale → infer → inverse_transform |
| B | Critical | `evaluate()` 傳入 `horizon=len(y_true)` → 改為 `min(len(y_true), prediction_length)` |
| C | Medium | `get_model_info()` 缺少 context_length/prediction_length/trainable_parameters 欄位 + None guard |
| D | Medium | Training history 只抓最後 epoch → 新增 `MetricsHistoryCallback` 累積每 epoch loss |
| E | Low | Device handling 不一致 → 訓練後 `model.cpu(); model.eval()` 確保推論在 CPU |
| F | Low | save/load 硬寫 `num_features=1` → metadata 存取 num_features |

#### lightning_module.py

| Bug | 嚴重度 | 說明 |
| --- | --- | --- |
| G | Low | Warmup scheduler 未組合 → `SequentialLR(LinearLR + CosineAnnealingLR)` |
| H | Low | `test_step` 用 val_ prefix → 抽出 `_eval_step(batch, prefix)` helper |

### 測試（24 tests 全部通過）

新增 `tests/test_patchtst_lightning.py`，對齊 HuggingFace 測試結構：

| Class | Tests | 驗證內容 |
| --- | --- | --- |
| `TestPatchTSTLightningInit` | 4 | model_name, model_type, is_fitted, device |
| `TestPatchTSTLightningFit` | 3 | is_fitted, returns self, training_history |
| `TestPatchTSTLightningPredict` | 5 | shape, horizon, finite values, not fitted, insufficient data |
| `TestPatchTSTLightningUncertainty` | 4 | keys, shapes, bounds order, not fitted |
| `TestPatchTSTLightningSaveLoad` | 2 | save/load cycle, nonexistent path |
| `TestPatchTSTLightningEvaluate` | 1 | mse/mae/rmse >= 0 |
| `TestPatchTSTLightningModelInfo` | 3 | info fields, trainable_params, unfitted |
| `TestPatchTSTLightningFactory` | 2 | ModelFactory create + with params |

### 測試結果

- **274 passed**（含 24 新增 Lightning tests），0 failed
- Pre-existing: HuggingFace test PermissionError（checkpoint 寫入權限，非本次引入）

## 完成標準

- [x] `ModelFactory.create_model('patchtst_lightning')` 回傳可用模型
- [x] `model.fit(X_train, y_train)` + `model.predict(X_test)` 正常運作
- [x] Unit tests 通過（24 passed, non-slow）
- [x] predict() 輸出在原始 scale（inverse_transform 已修復）
- [x] save/load cycle 驗證通過
- [x] `uv run main.py --models lightning` CLI 驗證（訓練 + 預測成功，test set evaluation 因資料長度 < context_length skip）
- [ ] GPU integration tests（`@pytest.mark.slow`，需 Docker 環境 — 歸入 #0100 Docker/CI 計畫）
- [x] README 更新 Lightning 標示（移除「開發中」/ WIP）
