# Plan #0210: Dashboard Volume=0 Fix

- **Date**: 2026-04-04
- **Status**: completed
- **Module**: visualization

## 目標

Yahoo Finance 對外匯 (forex) OTC 市場回傳 Volume=0，導致 dashboard 成交量 subplot 顯示空白。偵測 Volume 是否有效，無效時改顯示 Daily Price Range (High − Low) 替代。

## 影響範圍

| 檔案 | 變更 |
|------|------|
| `src/currency_predictor/visualization/visualizer.py` | `create_dashboard()` Volume 有效性檢查 + fallback |
| `src/currency_predictor/visualization/interactive.py` | `plot_technical_analysis()` 修正 `has_volume` 判斷 |
| `tests/test_visualizer.py` | 新增 Volume=0 scenario 測試 |
| `tests/test_interactive_visualizer.py` | 新增 Volume=0 scenario 測試 |

## 實作步驟

1. 修改 `visualizer.py` — `create_dashboard()` 增加 `df['Volume'].max() > 0` 檢查，fallback 到 Daily Range
2. 修改 `interactive.py` — `plot_technical_analysis()` 修正 `has_volume` 條件
3. 新增測試：Volume=0 時 dashboard 不 crash、fallback subplot 正確
4. Docker 跑全套測試驗證

## 完成標準

- Volume=0 時 dashboard 顯示 Daily Range 替代
- InteractiveVisualizer 正確判斷 has_volume
- 新增 Volume=0 測試，全套測試通過
