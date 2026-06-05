# Retraining Strategy

- **Date**: 2026-04-05 20:39
- **Status**: draft
- **Module**: prediction/
- **Reference**: [docs/reference/production-forecasting-best-practices.md](../reference/production-forecasting-best-practices.md) §7

## 目標

定義並實作模型 retraining 策略，自動化判斷何時需要重新訓練模型。

## 背景

- 目前完全手動重跑 `main.py` 來更新模型
- 沒有自動判斷「模型是否需要重訓練」的機制
- 2025 研究（arXiv）顯示 weekly periodic retraining 是好的起點，不需要每天重訓練
- 結合 drift monitoring (#0300)，可以在 performance 退化時觸發 ad-hoc retraining

## 影響範圍

| 檔案 | 修改內容 |
| ---- | -------- |
| `src/currency_predictor/prediction/retrainer.py` | 新增 `RetrainingScheduler` |
| `config.json` | 新增 retraining 配置區塊 |
| `tests/test_retrainer.py` | Retraining 邏輯測試 |

## 實作步驟

### Step 1: RetrainingScheduler

- 檢查最後一次 training 的時間戳
- 檢查最近的 drift score
- 判斷是否需要 retrain：
  - 超過 `max_age` 天數 → periodic retrain
  - MASE > `mase_threshold` → triggered retrain

### Step 2: Config 設定

```json
"retraining": {
    "strategy": "adaptive",
    "max_age_days": 7,
    "mase_threshold": 1.2,
    "rolling_window_days": 730
}
```

### Step 3: CLI 整合

- `main.py --check-retrain` → 檢查是否需要 retrain，不實際執行
- `main.py --auto-retrain` → 檢查並在需要時自動 retrain

## 完成標準

- Periodic + trigger-based 策略都可用
- 可配置 max_age 和 MASE threshold
- Retrain 前保留前一版模型供 rollback
- 所有現有 tests 通過

## 依賴

- #0250 (MASE metric) — 觸發判斷以 MASE 為基準
- #0300 (Drift monitoring) — 提供 drift score 作為 trigger 輸入
