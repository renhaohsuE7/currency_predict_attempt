# Walk-Forward Backtesting

- **Date**: 2026-04-05 20:33
- **Status**: draft
- **Module**: prediction/
- **Reference**: [docs/reference/production-forecasting-best-practices.md](../reference/production-forecasting-best-practices.md) §2

## 目標

實作 walk-forward backtesting，在多個時間點做 out-of-sample 預測，產出 RMSE/MASE 的分布，驗證模型在不同市場環境下的穩定性。

## 背景

- 目前用單次 80/20 train/test split，只反映一個時間點的表現
- 如果那段 test period 剛好穩定，RMSE 會好看但不代表真實能力
- Walk-forward 在多個 fold 做評估，產出 mean/std/min/max，更能反映穩定性

## 影響範圍

| 檔案 | 修改內容 |
| ---- | -------- |
| `src/currency_predictor/prediction/backtester.py` | 新增 `WalkForwardBacktester` class |
| `src/currency_predictor/prediction/predictor.py` | 新增 `backtest()` method |
| `main.py` | 新增 `--backtest` CLI flag |
| `tests/test_backtester.py` | Backtester 單元測試 |

## 實作步驟

### Step 1: WalkForwardBacktester

```
class WalkForwardBacktester:
    def __init__(self, n_splits, test_size, gap=0, strategy='expanding'):
        ...

    def run(self, model_class, model_params, data, target_col='Close'):
        # 對每個 fold:
        #   1. Split train/test
        #   2. model.fit(train)
        #   3. model.predict(test)
        #   4. 計算 RMSE, MAE, MASE, MDA
        # 回傳 DataFrame of per-fold metrics
```

- 支援 `expanding` (訓練集逐漸增大) 和 `rolling` (固定大小窗口) 策略
- `gap` 參數：train/test 之間的 embargo 期（避免 autocorrelation leakage）
- 每個 fold 獨立 fit scaler（避免 data leakage）

### Step 2: 整合到 pipeline

- `CurrencyPredictor.backtest(symbol, n_splits=5)` → 回傳 per-fold metrics
- `main.py --backtest` → 對所有 model 跑 backtest，產出摘要報告

### Step 3: 報告

- Per-fold metrics table
- Summary: mean ± std RMSE/MASE
- 視覺化：每個 fold 的預測 vs 實際

## 完成標準

- Expanding window 和 rolling window 策略都可用
- Per-fold metrics 正確計算
- Gap (embargo) 參數可配置
- 所有現有 tests 通過
