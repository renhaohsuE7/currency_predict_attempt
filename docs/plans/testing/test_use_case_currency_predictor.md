# Test Plan: CurrencyPredictor Use Case

- **Date**: 2026-04-03
- **Status**: completed
- **Test file**: `tests/test_use_case_currency_predictor.py`
- **Marker**: `@pytest.mark.slow`

## Gap

`test_currency_predictor.py` 只測 init + utility function，沒有呼叫 `train_model()`、`predict()`、`prepare_training_data()` 等核心方法。

## Strategy

用 fixture CSV 注入 DataStorage（mock collect），使 CurrencyPredictor 能用 real API 但不需網路。

## Tests

| Test | 說明 |
| --- | --- |
| `test_prepare_training_data_shapes` | 回傳 (X_train, y_train, X_test, y_test) shape 正確 |
| `test_train_model_returns_metrics` | training_completed=True, train_metrics 有值 |
| `test_train_then_predict` | train → predict → predictions finite |
| `test_predict_without_training_returns_error` | 未訓練 → error dict |
| `test_evaluate_model_produces_metrics` | mse/mae/rmse 非負 |
| `test_save_and_reload_model` | train → save → load → predict 一致 |
