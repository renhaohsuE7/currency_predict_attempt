# Currency Prediction Attempt

[![CI](https://github.com/renhaohsuE7/currency_predict_attempt/actions/workflows/ci.yml/badge.svg)](https://github.com/renhaohsuE7/currency_predict_attempt/actions/workflows/ci.yml)

使用 PatchTST 時間序列 Transformer 模型預測匯率及股票走勢，支援多模型比較、Walk-forward Backtesting、CAPM 特徵工程，並產出視覺化報告。

## Architecture

```
YahooFinanceCollector → DataStorage → DataProcessor → Model (train/predict) → ResultFormatter → CurrencyVisualizer
```

### Key Modules (`src/currency_predictor/`)

- **config/** — `ConfigManager` + Pydantic Settings，讀取 `config.json`
- **data/** — `YahooFinanceCollector` (OHLCV)、`DataStorage` (本地持久化)
- **models/** — `ModelFactory` (factory pattern)，三種 PatchTST 實作：
  - `patchtst_sklearn` — scikit-learn based (always available)
  - `patchtst_huggingface` — HuggingFace Transformers (支援預訓練 + fine-tune)
  - `patchtst_lightning` — PyTorch Lightning (`uv sync --extra lightning`)
- **prediction/** — `PredictionPipeline` (高層 orchestration)、`CurrencyPredictor` (低層 API)、`ModelComparer` (多模型比較)
- **backtesting/** — `BacktestRunner` (walk-forward validation)、`WalkForwardSplitter`、`FinancialMetrics`
- **reporting/** — `ResultFormatter` (JSON/Markdown reports)、`ReportGenerator` (PDF via fpdf2)
- **visualization/** — `CurrencyVisualizer` (dashboards, equity curves, drawdown)、`InteractiveVisualizer` (Plotly)
- **utils/** — `AssetType` (forex/stock/crypto/index 分類)、helpers

## HuggingFace Pretrained Models

HuggingFace 版 PatchTST 支援從 Hub 載入預訓練模型並 fine-tune，也可從頭訓練。

### 支援的預訓練模型

| Model | URL | Parameters | 說明 |
| --- | --- | --- | --- |
| granite-timeseries-patchtst | [ibm-granite/granite-timeseries-patchtst](https://huggingface.co/ibm-granite/granite-timeseries-patchtst) | 616K | 基礎版，ETTh1 dataset 訓練，context=512, Apache 2.0 |
| granite-timeseries-patchtst-fm-r1 | [ibm-granite/granite-timeseries-patchtst-fm-r1](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r1) | ~260M | Foundation model，支援 zero-shot forecasting + missing value imputation，context=8192 |

論文：[A Time Series is Worth 64 Words (ICLR 2023)](https://arxiv.org/abs/2211.14730)

### Fine-Tune 模式

| Mode | 說明 | 預設 epochs | 預設 LR |
| --- | --- | --- | --- |
| `from_scratch` | 從頭訓練（預設） | 50 | 1e-4 |
| `full` | 載入預訓練模型，全參數 fine-tune | 20 | 1e-5 |
| `linear_probe` | 凍結 backbone，只訓練 prediction head | 10 | 1e-3 |

### 使用範例

```python
from currency_predictor.models.patchtst import PatchTSTHuggingFace

# 從頭訓練（向後相容）
model = PatchTSTHuggingFace(context_length=64, prediction_length=7)

# 預訓練 + full fine-tune
model = PatchTSTHuggingFace(
    pretrained_model_name_or_path="ibm-granite/granite-timeseries-patchtst",
    fine_tune_mode="full",
    prediction_length=7,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test, horizon=7)
```

也可在 `config.json` 中設定：

```json
{
    "model_name": "patchtst_transformer",
    "model_params": {
        "pretrained_model_name_or_path": "ibm-granite/granite-timeseries-patchtst",
        "fine_tune_mode": "full",
        "pred_len": 7
    }
}
```

## Prerequisites

- **Python 3.11**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **NVIDIA GPU**（optional）— CUDA 12.x，程式碼自動 fallback 到 CPU

Docker 額外需要：

- **Docker Engine** ≥ 24.0 + **Compose V2**
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**（GPU 支援）

## Installation

### Docker Compose (recommended)

```bash
git clone <repo-url> && cd currency_predict_attempt

# 建置 image
docker compose build prod     # 生產 image（含所有模型，最小 runtime）
docker compose build dev      # 開發 image（含 pytest, black, jupyter）

# 一鍵完整流程（預設：多模型比較 + 視覺化）
docker compose run --rm prod

# 只訓練
docker compose run --rm prod uv run main.py --train

# 用已訓練模型預測 60 天（含 metrics）
docker compose run --rm prod uv run main.py --predict --days 60

# 強制重新下載 + 重新訓練
docker compose run --rm prod uv run main.py --fresh

# 指定貨幣對 / 股票
docker compose run --rm prod uv run main.py --symbols USDTWD=X,EURUSD=X
docker compose run --rm prod uv run main.py --symbols AAPL,TSLA

# Walk-forward Backtesting
docker compose run --rm prod uv run main.py --backtest --symbols USDTWD=X

# 測試
docker compose run --rm dev
```

> 沒有 NVIDIA GPU？建立 `docker-compose.override.yml` 移除 `deploy` 區塊，程式碼會自動 fallback 到 CPU。

### Local (alternative)

如果不使用 Docker，可直接用 `uv` 執行：

```bash
git clone <repo-url> && cd currency_predict_attempt
uv sync --all-extras --all-groups

uv run main.py                                       # 一鍵完整流程（多模型比較 + 視覺化）
uv run main.py --train --symbols USDTWD=X            # 只訓練
uv run main.py --predict --days 60 --symbols USDTWD=X  # 用已訓練模型預測 60 天
uv run main.py --backtest --symbols USDTWD=X         # Backtesting
uv run pytest                                        # 測試
```

## Complete Workflow

從零開始的完整操作流程：下載歷史資料 → 訓練模型 → 預測匯率 → 產出報告與視覺化圖表。

### Step 0: 設定 config.json

所有參數集中在 `config.json`，執行前先根據需求調整：

```jsonc
{
    "model_name": "patchtst_transformer",       // 模型：patchtst_sklearn / patchtst_transformer / patchtst_huggingface
    "model_params": {
        "seq_len": 64,                          // 輸入序列長度（look-back window）
        "pred_len": 22,                         // 預測長度（交易日；7=一週, 22=一個月）
        "patch_len": 8,
        "stride": 4,
        "d_model": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "dropout": 0.1
    },
    "symbols": ["USDTWD=X"],                   // 目標貨幣對（或股票 ticker 如 AAPL）
    "prediction_horizon": 22,                   // 預測天數（需與 pred_len 一致）
    "data_collection": {
        "period": "3y",                         // 歷史資料長度：2y / 3y / 5y / max
        "interval": "1d"                        // 資料間隔：1d / 1h / 5m
    },
    "model_training": {
        "period": "3y",                         // 訓練資料期間（需與 data_collection.period 一致）
        "target_column": "Close"
    },
    "prediction": {
        "period": "3y"                          // 預測資料期間（需與 data_collection.period 一致）
    },
    "backtest": {
        "enabled": false,                       // CLI --backtest 啟用
        "strategy": "rolling",                  // rolling | expanding
        "initial_train_days": 252,              // 初始訓練窗口（1年）
        "test_step_days": 30,                   // 每次前進天數
        "test_window_days": 30,                 // 測試窗口大小
        "data_period": "3y"                     // Backtest 使用的資料期間
    },
    "capm": {
        "enabled": false,                       // 啟用 CAPM 特徵（僅對股票有效）
        "market_index": "^GSPC",                // 市場基準（S&P 500）
        "rolling_window": 252
    }
}
```

> **提示**: `prediction_horizon` 和 `model_params.pred_len` 應保持一致。22 交易日 ≈ 一個月。

### Step 1: 執行預測

Pipeline 會自動完成所有步驟：下載 Yahoo Finance 歷史資料 → 建立技術指標 → 訓練模型 → 預測未來走勢 → 產出報告。

已下載的資料會自動抽樣驗證（spot-check 3 個日期），確認正確後跳過重複下載。

> **指令慣例**：以下 CLI 參數說明以 `uv run main.py` 簡寫。Docker 環境請加前綴 `docker compose run --rm prod`。
>
> **首次使用或程式碼更新後**，需先 rebuild image：`docker compose build prod`

```bash
# === 推薦：一鍵完整流程（預設：多模型比較 + 視覺化） ===

docker compose run --rm prod

# 強制重新下載資料 + 重新訓練（忽略快取）
docker compose run --rm prod uv run main.py --fresh

# === 訓練 / 預測分離 ===

# 只訓練模型（不預測）
docker compose run --rm prod uv run main.py --train --symbols 2330.TW

# 用已訓練的模型預測 60 天（含 evaluation metrics）
docker compose run --rm prod uv run main.py --predict --days 60 --symbols 2330.TW

# 指定使用某次操作的模型
docker compose run --rm prod uv run main.py --predict --days 30 --use-op 568164ca

# === 進階用法 ===

# 單一模型預測（不比較）
docker compose run --rm prod uv run main.py --single

# 多模型比較，不產出圖表
docker compose run --rm prod uv run main.py --no-viz

# 指定模型比較
docker compose run --rm prod uv run main.py --models sklearn,transformer

# 指定貨幣對
docker compose run --rm prod uv run main.py --symbols USDTWD=X,EURUSD=X

# Walk-forward Backtesting
docker compose run --rm prod uv run main.py --backtest --symbols USDTWD=X

# 使用自訂設定檔（例如台積電）
docker compose run --rm prod uv run main.py --config tw2330_config.json
```

### Step 2: 查看結果

每次執行的所有產出會儲存在獨立的時間戳目錄 `results/runs/{YYYYMMDD_HHMMSS}/`，`results/latest` 是指向最新 run 的 symlink：

```
results/
├── runs/
│   ├── 20260403_221438/
│   │   ├── manifest.json                  # 操作紀錄（op_id, type, status）
│   │   ├── config_snapshot.json           # 本次執行的 config 快照
│   │   ├── {op_id}/                       # 每次操作獨立目錄
│   │   │   ├── models/
│   │   │   │   └── {symbol}_{model}.joblib    # 已訓練的模型
│   │   │   ├── figures/
│   │   │   │   ├── {symbol}_dashboard.png     # 分析儀表板（-v）
│   │   │   │   └── {symbol}_model_comparison.png  # 多模型比較圖
│   │   │   ├── comparison_report.md           # 多模型比較報告
│   │   │   ├── predictions_{symbol}.csv       # 預測值 CSV（各模型 vs actual）
│   │   │   ├── metrics.csv                    # 評估指標 CSV（RMSE, MAE, MASE, MDA）
│   │   │   └── pipeline_results.json          # 完整執行結果
│   │   └── {op_id}/                       # --predict 產生新的 op
│   │       └── ...
│   └── 20260403_231015/
│       └── ...
└── latest -> runs/20260403_231015/        # 最新 run 的 symlink
```

> Docker 環境中 `data/`、`results/` 皆已透過 volume mount 映射到 host，直接在本機檢視即可。

### CLI 參數一覽

| 參數 | 說明 | 範例 |
| --- | --- | --- |
| `-c, --config` | 指定設定檔（預設 `config.json`） | `--config tw2330_config.json` |
| `--train` | 只執行資料收集 + 模型訓練（跳過預測） | `uv run main.py --train` |
| `--predict` | 用已訓練模型執行預測（含 evaluation metrics） | `uv run main.py --predict` |
| `--days N` | 覆蓋 config 中的 `prediction_horizon`（預測天數） | `uv run main.py --predict --days 60` |
| `--use-op ID` | 指定使用哪次操作的模型（搭配 `--predict`） | `uv run main.py --predict --use-op a3f7c1e2` |
| `--continue` | 在最新的 run 目錄內繼續（`--predict` 自動啟用） | `uv run main.py --continue` |
| `--single` | 單一模型模式（關閉多模型比較） | `uv run main.py --single` |
| `--no-viz` | 不產出視覺化圖表 | `uv run main.py --no-viz` |
| `--fresh` | 強制重新下載資料 + 重新訓練模型（忽略快取） | `uv run main.py --fresh` |
| `-v, --visualize` | 產出視覺化圖表（預設：啟用） | `uv run main.py -v` |
| `--compare` | 多模型比較模式（預設：啟用） | `uv run main.py --compare` |
| `--full` | 向後相容：等效預設行為（= `--compare -v`） | `uv run main.py --full` |
| `--models` | 指定比較的模型（逗號分隔） | `--models sklearn,transformer` |
| `--symbols` | 指定貨幣對/股票（逗號分隔，覆蓋 config） | `--symbols USDTWD=X,AAPL` |
| `--backtest` | Walk-forward backtesting 模式 | `uv run main.py --backtest` |
| `--backtest-strategy` | Backtest 策略：`rolling`（預設）或 `expanding` | `--backtest-strategy expanding` |

> `--train-only` 和 `--predict-only` 仍可使用（隱藏 alias），但建議改用 `--train` / `--predict`。

### Train → Predict 工作流程

完整 pipeline（預設）會一次完成 下載 → 訓練 → 預測。但實務上你可能想：

- 訓練一次，之後用同一組模型反覆預測不同天數
- 比較不同預測天數的 metrics（RMSE、MASE、MDA 等）

```bash
# 1. 先訓練（模型存在 results/runs/{run_dir}/{op_id}/models/）
uv run main.py --train --symbols 2330.TW

# 2. 用已訓練的模型預測 15 天（自動使用最新 train/full 操作的模型）
uv run main.py --predict --days 15 --symbols 2330.TW

# 3. 不滿意？改預測 60 天，同一組模型
uv run main.py --predict --days 60 --symbols 2330.TW

# 4. 指定使用特定操作的模型（op_id 可從 manifest.json 或 terminal 輸出取得）
uv run main.py --predict --days 30 --use-op 568164ca --symbols 2330.TW
```

**模型選擇邏輯**：`--predict` 自動啟用 `--continue`，在最新的 run 目錄中反向搜尋 manifest，找到最近一次 **completed** 的 `full` 或 `train` 操作，從該操作的 `models/` 目錄載入模型。若想指定特定操作，使用 `--use-op <op_id>`。

**`--predict` 包含 evaluation metrics**：預測完成後會自動在 test set 上計算 RMSE、MAE、MASE、MDA 等指標，並輸出 `predictions_{symbol}.csv` 和 `metrics.csv`。

### 可用模型

| 名稱 | CLI 值 | 說明 | 需求 |
| --- | --- | --- | --- |
| PatchTST sklearn | `sklearn` | sklearn-based，快速，CPU 友善 | 無額外需求 |
| PatchTST Transformer | `transformer` | 完整 Transformer 架構 | 無額外需求 |
| PatchTST HuggingFace | `huggingface` | 支援 pretrained + fine-tune | GPU + `uv sync --all-extras` |
| PatchTST Lightning | `lightning` | PyTorch Lightning 版 | `uv sync --extra lightning`（Docker prod 已含） |

### 完整範例：從零預測 USDTWD 一個月走勢

```bash
# 1. 調整 config.json: data_collection.period="2y", prediction_horizon=22, pred_len=22

# 2. 一鍵執行（自動下載 + 訓練 + 預測 + 報告 + 圖表）
docker compose build prod
docker compose run --rm prod

# 3. 查看結果（latest 指向最新 run）
cat results/latest/comparison_report.md
cat results/latest/prediction_report.md
ls results/latest/figures/
```

### 清理產出

```bash
# 清理 results/ 和 models/（保留已下載資料）
./scripts/clean.sh

# 清理全部（含已下載的 Yahoo Finance 資料）
./scripts/clean.sh --all

# Docker 環境
docker compose run --rm prod bash scripts/clean.sh
docker compose run --rm prod bash scripts/clean.sh --all
```

## Development

### Testing

專案共 **588 tests**，分為四個層級，透過 pytest marker 控制執行範圍。

#### Docker (recommended)

```bash
docker compose run --rm dev                                     # 快速測試（預設）
docker compose run --rm dev uv run pytest -m slow -v            # Offline E2E + Use Case
docker compose run --rm dev uv run pytest -m e2e -v             # Online E2E（需網路）
docker compose run --rm dev uv run pytest --cov=currency_predictor  # 覆蓋率
```

#### Local

```bash
uv run pytest                           # 快速測試（排除 @slow 和 @e2e）
uv run pytest -m slow -v               # Offline E2E + Use Case
uv run pytest -m e2e -v                # Online E2E（需網路）
uv run pytest --cov=currency_predictor --cov-report=html  # 覆蓋率
```

#### 特定類別

```bash
uv run pytest tests/test_use_case_*.py -v                       # Use Case tests
uv run pytest tests/test_e2e_cli.py -v                          # CLI routing tests
uv run pytest tests/test_e2e_stock.py -v                        # Stock E2E tests
uv run pytest tests/test_backtest_*.py -v                       # Backtesting tests
uv run pytest tests/test_e2e_pipeline_output.py -v              # Pipeline output tests
uv run pytest tests/test_data_*.py -v                           # 資料模組
uv run pytest tests/test_models_*.py -v                         # 模型模組
uv run pytest tests/test_patchtst_huggingface.py -v             # HuggingFace 整合
```

| 層級 | Tests | Marker | 說明 |
| --- | --- | --- | --- |
| Unit / Integration | 539 | 無 | 快速單元和整合測試（含 backtesting、CLI routing） |
| Slow (Offline E2E) | 33 | `@slow` | Use Case + sklearn/Lightning 離線端到端 + HuggingFace |
| Online E2E | 16 | `@e2e` | yfinance 真實資料（forex + 股票 ticker） |

### Code Quality

```bash
# Docker
docker compose run --rm dev uv run black .
docker compose run --rm dev uv run flake8

# Local
uv run black .
uv run flake8
```

### Jupyter & Shell (Docker)

```bash
docker compose up jupyter      # http://localhost:8888
docker compose run --rm dev bash
```

### Docker 備註

- HuggingFace 預訓練模型快取在 `.cache/huggingface/`（bind mount），不會污染 host `~/.cache/`
- dev service mount `src/` 和 `tests/`，修改程式碼後不需重新 build image

## Documentation

| 文件 | 說明 |
| --- | --- |
| [docs/architectures/system_architecture.md](docs/architectures/system_architecture.md) | 系統架構總覽：模組結構、類別階層、資料流程、設計模式 |
| [docs/usage_descriptions/model_modules.md](docs/usage_descriptions/model_modules.md) | 模型模組使用說明：PatchTST 各版本、ModelFactory、pretrained + fine-tune |
| [docs/usage_descriptions/data_modules.md](docs/usage_descriptions/data_modules.md) | 資料模組使用說明：YahooFinanceCollector、DataStorage、DataProcessor |
| [docs/usage_descriptions/config_management.md](docs/usage_descriptions/config_management.md) | 配置管理說明：Pydantic Settings、config.json |
| [docs/plans/](docs/plans/) | 開發計畫文件（#0000 ~ #0240） |
| [docs/plans/testing/](docs/plans/testing/) | Use Case / E2E 測試計畫 |
| [docs/testing_reports/](docs/testing_reports/) | 模組測試報告 |

## Project Structure

```
currency_predict_attempt/
├── src/currency_predictor/
│   ├── config/            # ConfigManager, Pydantic Settings
│   ├── data/              # YahooFinanceCollector, DataStorage
│   ├── models/
│   │   └── patchtst/
│   │       ├── sklearn/   # sklearn-based PatchTST
│   │       ├── huggingface/  # HuggingFace Transformers PatchTST
│   │       └── lightning/    # PyTorch Lightning PatchTST
│   ├── prediction/        # Pipeline, Predictor, ModelComparer
│   ├── backtesting/       # BacktestRunner, WalkForwardSplitter, FinancialMetrics
│   ├── reporting/         # ResultFormatter, ReportGenerator (PDF)
│   ├── visualization/     # CurrencyVisualizer, InteractiveVisualizer
│   └── utils/             # AssetType, helpers
├── tests/                 # 588 tests, 80%+ coverage
├── docs/
│   ├── architectures/     # 系統架構文件
│   ├── usage_descriptions/ # 模組使用說明
│   ├── plans/             # 開發計畫文件
│   └── development/       # 開發報告與分析
├── notebooks/             # Jupyter notebooks
├── scripts/               # clean.sh (清理產出)
├── Dockerfile             # Multi-stage build (base/prod/dev)
├── docker-compose.yml     # prod / dev / jupyter services
├── .cache/huggingface/    # HF 模型快取 (Docker volume, gitignored)
├── config.json            # 主要設定檔
├── main.py                # Entry point
└── pyproject.toml         # uv / pytest / coverage 設定
```
