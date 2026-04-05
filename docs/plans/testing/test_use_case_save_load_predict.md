# Test Plan: Save/Load/Predict Use Case

- **Date**: 2026-04-03
- **Status**: completed
- **Test file**: `tests/test_use_case_save_load_predict.py`
- **Marker**: `@pytest.mark.slow`

## Gap

Train → Save → Load → Re-predict 完整 cycle 無測試。既有 `test_models_patchtst.py` 只測 save 檔案存在，沒測 load 後 predict 一致性。

## Tests

| Test | 說明 |
| --- | --- |
| `test_save_load_predict_consistency` | fit → predict → save → load → predict → 兩次結果一致 |
| `test_load_model_predict_with_extra_columns` | save → load → predict(X_with_Close) 自動過濾 |
| `test_save_load_preserves_training_history` | save → load → training_history 一致 |
| `test_save_load_preserves_feature_columns` | save → load → _feature_columns 一致 |
| `test_load_nonexistent_file_returns_false` | load_model("bad_path") 回傳 False |

## Data

- Fixture: `e2e_fixture_path` → DataProcessor pipeline → split X/y
- Model: PatchTSTSklearn with `fast_sklearn_params`
- Storage: `tmp_path` for model files
