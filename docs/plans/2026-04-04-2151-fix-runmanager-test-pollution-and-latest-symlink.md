# Fix RunManager Test Pollution & results/latest Symlink

- **Date**: 2026-04-04 21:51
- **Status**: completed
- **Module**: tests, prediction/run_manager

## 目標

修正 E2E CLI tests 汙染 `results/` 目錄的問題，確保 `results/latest` symlink 永遠指向最新的**真實** pipeline run。

## 問題

### Issue 1: tests 寫入真實 `results/` 目錄

`tests/test_e2e_cli.py::TestNewModeRouting` 中有 4 個 test 沒有 mock `RunManager`。每次 test suite 執行時：

1. 建立真實的 `RunManager(base_dir="results")`
2. `setup()` 建立 `results/runs/YYYYMMDD_HHMMSS/`
3. `start_operation()` 建立空的 `{op_id}/figures/` + `{op_id}/models/` 骨架
4. `complete_operation()` 標記為 completed
5. `update_latest_symlink()` 覆寫 `results/latest` → 指向空骨架

受影響的 test：
- `test_train_only_passes_op_type` (L198)
- `test_train_only_compare_mode` (L223)
- `test_predict_only_passes_use_op` (L277)
- `test_backtest_mode_routes_to_run_backtest_mode` (L299)

### Issue 2: 現行 output 沒有 symbol 子目錄

所有 symbol 的結果平鋪在 `{op_dir}/` 內，靠檔名區分。不影響功能但不利瀏覽和管理。

## 影響範圍

| 檔案 | 修改內容 |
|------|----------|
| `tests/test_e2e_cli.py` | 4 個 test 加 `@patch('main.RunManager')` |
| cleanup script (optional) | 清理已有的 test 骨架目錄 |

## 實作步驟

### Step 1: 修復 4 個未 mock RunManager 的 test

對 `TestNewModeRouting` 中這 4 個 test 加上 `@patch('main.RunManager')`，與現有的 `test_continue_run_uses_continue_latest` 保持一致。

每個 test 需要：
1. 加 `@patch('main.RunManager')` decorator
2. 函數簽名加 `mock_rm` 參數
3. 設定 mock RunManager instance（`op_dir = Path('/tmp/fake_op_dir')`）

範例：

```python
@patch('main._run_single_mode', return_value=0)
@patch('main.RunManager')                           # ← 新增
@patch('main.ConfigManager')
@patch('main.ensure_directories')
@patch('main.setup_logging')
def test_train_only_passes_op_type(
    self, mock_logging, mock_dirs, mock_cm, mock_rm, mock_single   # ← 加 mock_rm
):
    ...
    mock_rm_instance = MagicMock()
    mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
    mock_rm.return_value = mock_rm_instance
    ...
```

### Step 2: 清理已有的 test 骨架目錄

手動清理 `results/runs/` 中被 test 汙染的 run 目錄。辨別方式：
- `config_snapshot.json` 為 `{}` 的 run
- 所有 operation 在同一秒內完成
- op 子目錄內 `figures/` 和 `models/` 為空

候選清理目錄：
- `results/runs/20260404_133953/`
- `results/runs/20260404_211301/`

修復 `results/latest` symlink 指向正確的真實 run。

### Step 3 (scope 外，僅記錄): symbol 子目錄化

如果日後需要在同一 run 處理多 symbol 且需獨立管理結果，可考慮在 `{op_dir}/` 下新增 symbol 子目錄結構。目前按檔名區分夠用，暫不實作。

## 風險評估

- **低風險**：只修 test code，不動 production code
- 加 mock 後 test 行為與實際 main() 的 RunManager 互動一致（已有 `test_continue_run_uses_continue_latest` 做為參考）
- 清理 test 骨架目錄前確認不包含真實 pipeline 結果

## 驗證

```bash
# 修復後跑 test
uv run pytest tests/test_e2e_cli.py -v -k "TestNewModeRouting"

# 確認不再產生新的 run 目錄
ls results/runs/  # 執行前後數量不變

# 全部 test
uv run pytest

# 清理後確認 latest symlink 正確
ls -la results/latest
```

## 完成標準

- 4 個 test 加上 `@patch('main.RunManager')`
- test suite 執行後不再在 `results/runs/` 產生新目錄
- `results/latest` 不再被 test 覆寫
- 已有的 test 骨架目錄被清理
- 所有 539 個 test 繼續通過
