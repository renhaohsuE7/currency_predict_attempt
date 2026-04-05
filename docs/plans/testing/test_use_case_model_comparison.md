# Test Plan: Model Comparison Use Case

- **Date**: 2026-04-03
- **Status**: completed
- **Test file**: `tests/test_use_case_model_comparison.py`
- **Marker**: `@pytest.mark.slow`

## Gap

`test_model_comparer.py` 全部使用 mock predictor，沒有真實模型訓練比較。無法驗證 compare() 在真實資料下的結構正確性。

## Strategy

用兩個不同 config 的 sklearn 模型（不同 n_estimators）比較，避免 HuggingFace/Lightning 的 GPU/時間需求。Mock `collect_and_store_data` 改用 fixture CSV 注入 DataStorage。

## Tests

| Test | 說明 |
| --- | --- |
| `test_two_sklearn_configs_comparison` | 兩個 sklearn config → compare() 回傳有效結構 |
| `test_comparison_returns_valid_ranking` | overall_ranking 非空、RMSE 遞增排序 |
| `test_comparison_best_model_per_symbol` | 每個 symbol 有 best_model |
| `test_comparison_all_models_have_metrics` | 每個 model result 有 test_metrics.rmse |
| `test_comparison_report_generation` | compare 結果 → generate_comparison_report() 產出有效 markdown |
