# Currency Prediction Attempt

A machine learning project for predicting currency (FX) exchange rates with
PatchTST-style time-series models. Data comes from Yahoo Finance via `yfinance`.

> **Honest status:** in a leakage-free walk-forward evaluation on USDTWD, **none
> of the three PatchTST implementations beats a naive random-walk baseline** —
> all are substantially worse. See `docs/development/patchtst_evaluation.md`.
> Treat this repo as a learning/scaffolding project, not a production forecaster.

## Architecture

`CurrencyPredictor` is a thin **Facade** that delegates to three collaborators:

| Component | File | Responsibility |
|---|---|---|
| `DataManager` | `data/manager.py` | collect → store → clean → feature-engineer → split |
| `ModelTrainer` | `prediction/trainer.py` | create / train (with validation split) / evaluate / save / load |
| `PredictionEngine` | `prediction/engine.py` | produce predictions from the trained model |

The model layer is built around a `ModelFactory` over a shared `BaseModel`
interface (`fit(X, y, validation_data=None, **kwargs)` / `predict`), with three
interchangeable PatchTST backends.

## Project Structure

```
src/currency_predictor/
├── config/                 # ConfigManager + settings (pydantic-settings)
├── data/
│   ├── collectors.py       # YahooFinanceCollector (yfinance)
│   ├── storage.py          # DataStorage (CSV files under data/raw, data/processed)
│   └── manager.py          # DataManager (collect/store/process/split)
├── data_processor.py       # DataProcessor (indicators, lags, scaling, split)
├── models/
│   ├── base.py             # BaseModel / ModelType / TimeSeries bases
│   ├── factory.py          # ModelFactory + create_patchtst_model
│   └── patchtst/
│       ├── config.py       # PatchTSTConfig / TrainingConfig
│       ├── sklearn/        # PatchTSTSklearn (GradientBoosting, fast)
│       ├── huggingface/    # PatchTSTHuggingFace (real Transformer)
│       └── lightning/      # PatchTSTLightningWrapper (PyTorch Lightning)
├── prediction/
│   ├── predictor.py        # CurrencyPredictor (Facade)
│   ├── trainer.py          # ModelTrainer
│   ├── engine.py           # PredictionEngine
│   └── pipeline.py         # PredictionPipeline (multi-symbol orchestration)
├── reporting/              # ResultFormatter (ASCII status tags, cp950-safe)
└── visualization/          # CurrencyVisualizer (matplotlib dashboards, Chinese font)
main.py                     # CLI entry point
scripts/evaluate_patchtst.py# 3-model vs naive walk-forward evaluation
docs/development/           # architecture analysis, refactor spec/plan, evaluation
```

## Setup

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra lightning      # full deps incl torch + transformers + pytorch-lightning
```

## Usage

```bash
uv run python main.py                 # run the prediction pipeline
uv run python main.py --visualize     # also render dashboards
uv run python scripts/evaluate_patchtst.py   # reproduce the model-vs-naive evaluation
```

Minimal API (Facade):

```python
from currency_predictor.prediction import CurrencyPredictor

p = CurrencyPredictor(model_name="patchtst_huggingface")
p.collect_and_store_data(["USDTWD=X"], period="1y")
p.train_model("USDTWD=X", period="1y")
print(p.predict("USDTWD=X", horizon=7))
```

## Data flow & a known limitation

`collect → DataStorage → DataProcessor`: each collection writes a **new
timestamped CSV** (`data/raw/{symbol}_{period}_{YYYYmmdd_HHMMSS}.csv`); loads
pick the newest file by mtime. This accumulates duplicate files, has no single
source of truth, and offers no schema/query. A relational store (e.g. the parent
project's PostgreSQL, upserting by `(symbol, date)`) would be idempotent,
queryable, and de-duplicated. `DataManager` takes its storage backend via the
constructor, so a `PostgresDataStore` implementing the same
`load_raw_data`/`save_raw_data` interface could be dropped in later without
touching the upper layers (not built here — YAGNI). See
`docs/development/code-map-and-data-flow.md`.

## Development

```bash
uv run pytest -q                              # full suite
uv run pytest -q --ignore=tests/test_visualizer.py   # skip a flaky matplotlib/GUI test file
uv run black .                                # format
```

Refactor & evaluation notes live in `docs/development/`:
`2026-06-21-refactor-and-evaluation-design.md` (spec),
`...-plan.md` (plan), `baseline-2026-06-21.md`,
`code-map-and-data-flow.md`, `patchtst_evaluation.md`.
