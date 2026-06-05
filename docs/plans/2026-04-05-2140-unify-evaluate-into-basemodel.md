# Unify evaluate() into BaseModel

- **Date**: 2026-04-05 21:40
- **Status**: completed
- **Module**: models/base.py, models/patchtst/*, prediction/predictor.py

## 目標

將三個 PatchTST 模型重複的 `evaluate()` 實作統一搬進 BaseModel，使所有模型自動獲得 MASE/MDA metrics。

## 背景

跑預設 pipeline 後發現只有 NaiveModel 有 MASE/MDA。根本原因：`_evaluate_model()` 的 if/else 分支讓 PatchTST 走 `model.evaluate()` 路徑（只回傳 MSE/MAE/RMSE），NaiveModel 走 fallback 路徑（有 MASE/MDA）。

三個 PatchTST 的 `evaluate()` 是幾乎相同的程式碼（predict + 基本 metrics），適合 Pull Up Method 重構。

## 影響範圍

| 檔案 | 修改 |
| --- | --- |
| `src/currency_predictor/models/base.py` | BaseModel 新增 evaluate() |
| `src/currency_predictor/models/patchtst/sklearn/model.py` | 刪除 evaluate() |
| `src/currency_predictor/models/patchtst/huggingface/model.py` | 刪除 evaluate() |
| `src/currency_predictor/models/patchtst/lightning/wrapper.py` | 刪除 evaluate() |
| `src/currency_predictor/prediction/predictor.py` | _evaluate_model() 簡化 |
| `tests/test_e2e_offline.py` | evaluate 測試加 MASE/MDA 斷言 |
| `tests/test_patchtst_lightning.py` | evaluate 測試加 MASE/MDA 斷言 |
| `tests/test_patchtst_huggingface.py` | evaluate 測試加 MASE/MDA 斷言 |

## 實作步驟

1. BaseModel 新增 evaluate(X, y_true, y_train=None) — predict() + 統一 metrics
2. 刪除三個 PatchTST 子類的 evaluate()
3. _evaluate_model() 簡化為 self.model.evaluate(X, y_true, y_train=y_train)
4. 更新 evaluate 相關測試

## 風險評估

- 低風險：predict() 不帶 horizon 已確認各模型安全 fallback
- evaluate() production 呼叫者只有 _evaluate_model() 一處
- NaiveModel 自動繼承，不需修改

## 完成標準

- 全部測試通過
- 預設 pipeline 跑完後所有模型都顯示 MASE/MDA
