# Plan: Rename evaluate → evaluate_single_shot + 評估方法研究

- **Date**: 2026-04-05 23:45
- **Status**: completed
- **Module**: base model, predictor, comparer, tests

## Context

目前 `BaseModel.evaluate()` 是 single-shot 評估：模型對整個 X_test 只呼叫一次 `predict()`，產出固定 `pred_len` 個預測值，再跟 `y_true[-pred_len:]` 比較。這有兩個問題：

1. **評估數據點太少** — 不管 `test_days` 設多大，metrics 只根據 `pred_len` (15) 個值計算
2. **時間對齊問題** — 模型預測的是 X_test 尾端之後的未來值，但比較的是 y_test 的最後 pred_len 天

需要先將舊方法改名以釐清語意，再研究合適的新評估方法。

---

## Phase 1: Rename `evaluate()` → `evaluate_single_shot()`

### 影響範圍

| 檔案 | 行號 | 修改 |
| --- | --- | --- |
| `src/currency_predictor/models/base.py` | 101 | 方法名 `evaluate` → `evaluate_single_shot`，更新 docstring 標明 single-shot 語意 |
| `src/currency_predictor/prediction/predictor.py` | 357, 359 | `_evaluate_model()` 內呼叫改為 `.evaluate_single_shot()` |
| `src/currency_predictor/prediction/comparer.py` | 497 | `predictor.model.evaluate(` → `.evaluate_single_shot(` |
| `tests/test_patchtst_huggingface.py` | 170 | `trained_model.evaluate(` → `.evaluate_single_shot(` |
| `tests/test_patchtst_lightning.py` | 232 | `trained_model.evaluate(` → `.evaluate_single_shot(` |
| `tests/test_e2e_offline.py` | 107, 113 | `model.evaluate(` → `.evaluate_single_shot(`，更新 docstring |
| `notebooks/currency_prediction_pipeline.ipynb` | cell ~2068 | `model.evaluate(` → `.evaluate_single_shot(` |

### 不受影響

- **backtesting/runner.py** — 使用 `ModelComparer.compute_unified_metrics()`，不呼叫 `model.evaluate()`
- **所有 model 子類別** (naive, sklearn, HF, lightning) — 沒有 override `evaluate()`，自動繼承新名稱
- **`_evaluate_model()` 本身不改名** — 它是 predictor 的 private helper，語意是「評估模型」，名稱仍合理

### 實作步驟

1. `base.py` — rename method + 更新 docstring 加上 single-shot 說明
2. `predictor.py` — 更新 call site + docstring
3. `comparer.py` — 更新 call site
4. Tests (3 files) — 更新 call sites
5. Notebook — 更新 call site
6. 跑全部測試確認

---

## Phase 2: 評估方法研究 — 適合本專案的評估策略

### 現有問題分析

**Single-shot evaluate 的問題：**
```
X_test (60 天) → model.predict(X_test) → 產出 15 個值 → 跟 y_test[-15:] 比
```
- 只用 15 點算 metrics，統計上不可靠
- predictions 是「X_test 末尾之後的 15 天」，但 y_test[-15:] 是「test period 的最後 15 天」

### 業界標準做法 (文獻摘要)

根據 [Hyndman FPP3 §5.10](https://otexts.com/fpp3/tscv.html) 和 [PMC forecast evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/)：

1. **Rolling Origin Evaluation (Time Series Cross-Validation)**
   - 滑動 origin，每次用 origin 之前的資料預測之後 h 步
   - 產生多組 (predicted, actual) 配對，metrics 更穩健
   - FPP3: "forecast error increases as horizon increases"

2. **Multi-step 評估要分 horizon 報告**
   - 不是把所有 h 步的 error 平均成一個數字
   - 而是報告 h=1, h=2, ..., h=15 各自的 error
   - 可以看出模型在不同 horizon 的衰退程度

3. **必須跟 naive baseline 比較**
   - MASE < 1.0 才算有意義（我們已有 MASE，但基於 single-shot 結果）
   - 匯率預測中，naive 通常是理論天花板

4. **避免的陷阱**
   - 不可對整個 series 做 preprocessing 後才 split（data leakage）
   - 不可只看圖，要看 metrics（visual illusion with rolling origins）
   - 不可用單一 test window，需要多個 origin 點

### 建議的新方法：`evaluate_rolling()`

**概念：** 在 test set 上做 rolling origin evaluation

```python
def evaluate_rolling(
    self,
    X: pd.DataFrame,
    y_true: pd.Series,
    seq_len: int,
    pred_len: int,
    y_train: Optional[pd.Series] = None,
    step: int = 1,
) -> Dict[str, Any]:
    """Rolling origin evaluation over test set.

    從 test set 的第 seq_len 個位置開始，每次取 seq_len 個值做 context，
    預測下一個 pred_len 步，跟實際值比較。origin 每次前進 step 步。

    Returns:
        {
            'per_horizon': {1: {rmse, mae, ...}, 2: {...}, ...},
            'aggregate': {rmse, mae, mase, mda, ...},
            'n_origins': int,
        }
    """
```

**滑動邏輯：**
```
test set: [t0, t1, ..., t59]  (60 天)
seq_len=32, pred_len=15, step=1

origin 0: context=[t0..t31]  → predict [t32..t46] → compare with actual [t32..t46]
origin 1: context=[t1..t32]  → predict [t33..t47] → compare with actual [t33..t47]
...
origin 13: context=[t13..t44] → predict [t45..t59] → compare with actual [t45..t59]

共 14 個 origin，每個 horizon 有 14 個 error sample
```

**關鍵限制：** 目前 PatchTST 的 `predict()` 只接受整個 DataFrame，取最後 seq_len 做 one-shot。支援 rolling 的方式：

- **Option A: 外部切 window** — `evaluate_rolling()` 每次切一段 X 丟給 `predict()`。不需改 model，但每次 predict 都有 model overhead
- **Option B: 新增 model.predict_rolling()** — 模型內部做 batch rolling prediction，效率高但需改每個 model 子類別

### 建議

**Phase 1（本次實作）：** 只做 rename，不加新功能

**Phase 2（完成後討論）：**
- 是否新增 `evaluate_rolling()` 用 Option A（外部切 window，不改 model）
- 是否加 per-horizon metrics 分析
- 是否跟現有 backtest runner 整合（runner 已有 rolling fold 但也有 alignment 問題）

---

## 完成標準

### Phase 1
- [x] `evaluate()` 改名為 `evaluate_single_shot()`
- [x] 所有 7 個 call sites 更新
- [x] 全部測試通過
- [x] notebook 更新

### Phase 2
- [x] 討論後決定是否實作 `evaluate_rolling()`
