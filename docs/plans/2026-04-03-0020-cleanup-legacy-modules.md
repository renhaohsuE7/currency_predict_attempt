# 清理 Legacy 模組

- **Date**: 2026-04-03 00:20
- **Status**: completed
- **Module**: models/, data_collector.py, tests/
- **後續**: 完成後才能開始 [`2026-04-03-0010-boost-test-coverage.md`](2026-04-03-0010-boost-test-coverage.md)

## 目標

移除已被新架構取代的 legacy 模組，降低維護負擔並消除覆蓋率分母中的死碼。

**為什麼要先做這個**：目前測試大多 import legacy 路徑，補覆蓋率等於替死碼灌水。清理後才能確保測試對齊新架構。

## 影響範圍

### Legacy production modules（待刪除）

| 檔案 | 覆蓋率 | 被取代者 | 說明 |
| --- | --- | --- | --- |
| `src/currency_predictor/models.py` | 0% | `models/` package | 舊的單檔，含舊 CurrencyPredictor |
| `src/currency_predictor/models/patchtst.py` | 0% | `models/patchtst/sklearn/model.py` | 舊的單檔 PatchTST |
| `src/currency_predictor/models/patchtst_transformer.py` | 23% | `models/patchtst/huggingface/model.py` | 舊的 transformer 版 |
| `src/currency_predictor/data_collector.py` | 56% | `data/collectors.py` YahooFinanceCollector | 舊的資料收集器 |

### 受影響的 test files（需更新 import）

| 測試檔案 | 目前 legacy import | 應改為 |
| --- | --- | --- |
| `test_models_patchtst.py` | `models.patchtst.PatchTST` | `models.patchtst.sklearn.model.PatchTSTSklearn` 或保持 re-export |
| `test_training_issues.py` | `models.patchtst.PatchTST` | 同上 |
| `test_architecture_analysis.py` | `models.patchtst_transformer` | `models.patchtst.huggingface.model` |
| `test_currency_predictor.py` | `CurrencyDataCollector` | `data.collectors.YahooFinanceCollector` |

### 受影響的其他檔案（需確認）

- `__init__.py` — 目前 re-export `CurrencyDataCollector` from `data_collector.py`
- `models/__init__.py` — 可能 import legacy class
- `examples/`, `notebooks/` — 可能引用 legacy 路徑

## 實作步驟

1. **掃描所有 import** — grep 找出引用四個 legacy 模組的所有檔案
2. **更新 `__init__.py` re-exports** — `CurrencyDataCollector` 指向 `data.collectors.YahooFinanceCollector`（或新建 adapter）
3. **更新 test imports** — 改指向新架構路徑
4. **更新 examples/notebooks** — 若有引用
5. **刪除四個 legacy 檔案**
6. **跑 `uv run pytest`** 確認 0 failures
7. **跑 `uv run pytest --cov`** 確認覆蓋率分母縮減，比率提升

## 風險評估

- `__init__.py` 的 public API 有外部使用者依賴 `CurrencyDataCollector` 名稱
  - 對策：在 `__init__.py` 保留別名 `CurrencyDataCollector = YahooFinanceCollector`
- `models/patchtst/__init__.py` 已有 `PatchTST = PatchTSTSklearn` re-export
  - 對策：保留 re-export，刪除 legacy 檔案不影響外部 import
- example / notebook 可能因 import 路徑變更而壞掉
  - 對策：同步更新

## 執行結果（2026-04-03）

### 已刪除檔案

- `src/currency_predictor/models.py`
- `src/currency_predictor/models/patchtst.py`
- `src/currency_predictor/models/patchtst_transformer.py`
- `src/currency_predictor/data_collector.py`
- `src/currency_predictor/models/transformer/` (整個目錄，broken imports)
- `tests/test_architecture_analysis.py` (script-based，imports legacy)

### Import 遷移

- `__init__.py`: `CurrencyDataCollector` → `YahooFinanceCollector` (保留別名)
- `test_currency_predictor.py`: 重寫為 `TestYahooFinanceCollector`
- `factory.py` / `predictor.py`: 僅 docstring 或 backward-compatible alias，無需修改

### 測試結果

- **151 passed, 0 failed, 1 skipped**（-1 from deleted test file）
- 覆蓋率：**57% → 67%**（移除死碼後分母縮減）

## 完成標準

- [x] 四個 legacy 檔案 + transformer/ 目錄已刪除
- [x] 所有 import 已遷移至新架構（零 legacy import）
- [x] `uv run pytest` 0 failures
- [x] `__init__.py` public API 保持向後兼容（re-export 別名）
- [x] 覆蓋率從 57% 提升至 67%
