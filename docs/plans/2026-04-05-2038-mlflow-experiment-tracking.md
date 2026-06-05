# MLflow Experiment Tracking

- **Date**: 2026-04-05 20:38
- **Status**: draft
- **Module**: prediction/, models/
- **Reference**: [docs/reference/production-forecasting-best-practices.md](../reference/production-forecasting-best-practices.md) §8

## 目標

整合 MLflow，記錄每次 training run 的 hyperparameters、metrics、artifacts，支援 experiment 比較和 model registry。

## 背景

- 目前沒有 experiment tracking，每次 hyperparameter 調整的結果只存在 terminal output 和 results/ 中
- 無法有效比較不同 hyperparameter 組合的效果
- 無法追溯歷史最佳模型的配置
- MLflow 是最廣泛使用的開源 experiment tracking 工具

## 影響範圍

| 檔案 | 修改內容 |
| ---- | -------- |
| `pyproject.toml` | `uv add --optional tracking mlflow` |
| `docker-compose.yml` | 可選 mlflow server service |
| `src/currency_predictor/prediction/tracking.py` | MLflow wrapper（graceful degradation） |
| `src/currency_predictor/prediction/predictor.py` | training 結束後 log metrics/params |
| `tests/test_tracking.py` | Tracking 測試 |

## 實作步驟

### Step 1: MLflow wrapper

- `TrackingClient` class，封裝 MLflow API
- 若 MLflow 未安裝，graceful degradation（只 log warning，不中斷 pipeline）
- `log_training_run(model_name, params, metrics, model_path)`
- `log_prediction(symbol, predictions, model_version)`

### Step 2: 整合到 training pipeline

- `train_model()` 結束後自動 log
- Record: model_name, seq_len, pred_len, RMSE, MAE, MASE, training_time

### Step 3: Docker Compose MLflow server

- 新增 `mlflow` service（SQLite backend + artifact store）
- 可選啟動（`docker compose --profile tracking up`）

## 完成標準

- Training runs 自動記錄到 MLflow
- MLflow UI 可瀏覽 experiment history
- 不安裝 MLflow 時 pipeline 正常運作（graceful degradation）
- 所有現有 tests 通過
