# Test Plan: CLI Entry Point E2E

- **Date**: 2026-04-03
- **Status**: completed
- **Test file**: `tests/test_e2e_cli.py`
- **Marker**: 無（unit test level）

## Gap

`main.py` 的 CLI argument parsing 和 mode routing 完全沒有測試。

## Strategy

Mock ConfigManager / PredictionPipeline / ModelComparer，只測 argument parsing 和 mode routing。

## Tests

| Test | 說明 |
| --- | --- |
| `test_default_args` | argparse 預設值正確 |
| `test_visualize_flag` | `-v` 設為 True |
| `test_compare_flag` | `--compare` 設為 True |
| `test_models_parsing` | `--models sklearn,huggingface` 正確解析 |
| `test_symbols_parsing` | `--symbols USDTWD=X,AAPL` 正確解析 |
| `test_single_mode_called` | 無 compare → _run_single_mode 被呼叫 |
| `test_compare_mode_called` | --compare → _run_compare_mode 被呼叫 |
