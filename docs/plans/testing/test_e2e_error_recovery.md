# Test Plan: Error Recovery E2E

- **Date**: 2026-04-03
- **Status**: completed
- **Test file**: `tests/test_e2e_error_recovery.py`
- **Marker**: 無（mock-based）

## Gap

所有測試只覆蓋 happy path，沒有 partial failure / graceful degradation 測試。

## Strategy

Mock YahooFinanceCollector / DataStorage 讓指定 symbol 拋例外，驗證上層模組不 crash。

## Tests

| Test | 說明 |
| --- | --- |
| `test_collect_partial_failure` | 2 symbols, 1 fails → 另一個仍成功 |
| `test_train_failure_returns_error_dict` | train_model() 失敗 → training_completed=False |
| `test_predict_unfitted_model_error` | 未訓練 → predict 回傳 error dict |
| `test_comparer_one_model_fails` | 2 models, 1 fails → 另一個仍有結果 |
| `test_invalid_symbol_graceful` | 不存在 symbol → 不 crash |
