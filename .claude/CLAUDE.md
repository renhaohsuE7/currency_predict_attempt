# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Always respond in **English or Traditional Chinese (繁體中文) only**. Never use Simplified Chinese.

## Commands

All commands run inside Docker containers.

```bash
# Install dependencies (inside container)
uv sync

# Run main prediction pipeline
docker compose run --rm prod
docker compose run --rm prod uv run main.py --visualize

# Tests — one-off (creates & removes container)
docker compose run --rm test                                                    # all tests
docker compose run --rm test uv run pytest tests/test_config_manager.py -v      # single file
docker compose run --rm test uv run pytest tests/test_config_manager.py::test_fn -v  # single test

# Tests — interactive dev (container must be running)
docker exec -it currency-pred-dev uv run pytest -v                              # all tests
docker exec -it currency-pred-dev uv run pytest tests/test_config_manager.py -v  # single file

# Code quality
docker compose run --rm test uv run black .    # format
docker compose run --rm test uv run flake8     # lint
docker compose run --rm test uv run mypy       # type check
```

## Architecture

Currency exchange rate prediction system using PatchTST time series models. Python 3.11, managed with `uv`.

### Core Data Flow

```
YahooFinanceCollector → DataStorage → DataProcessor → Model (train/predict) → ResultFormatter → CurrencyVisualizer
```

### Key Modules (`src/currency_predictor/`)

- **config/** — `ConfigManager` loads `config.json`, validates via Pydantic Settings (`settings.py`)
- **data/** — `YahooFinanceCollector` fetches OHLCV data; `DataStorage` handles local persistence
- **models/** — `ModelFactory` (factory pattern) creates model instances. Three PatchTST implementations:
  - `patchtst_sklearn` — scikit-learn based (always available)
  - `patchtst_huggingface` — HuggingFace Transformers (optional)
  - `patchtst_lightning` — PyTorch Lightning (optional)
  - All implement `BaseModel` ABC from `models/base.py`
- **prediction/** — `PredictionPipeline` (high-level orchestration: collect → train → predict → report) and `CurrencyPredictor` (low-level step-by-step API)
- **reporting/** — `ResultFormatter` (JSON/Markdown), `ReportGenerator` (PDF via fpdf2, optional)
- **visualization/** — `CurrencyVisualizer` (matplotlib), `InteractiveVisualizer` (Plotly, optional), `ThemeManager` (light/dark)

### Entry Point

`main.py` → `ConfigManager` → `PredictionPipeline.run_full_pipeline()` → optional `CurrencyVisualizer`

### Configuration

`config.json` defines active model (`patchtst_transformer`), currency symbols, data collection params, preprocessing settings, and file paths. `config_example.json` is the template.

## Critical: docker compose + uv Only

**Dev environment: docker compose + Dockerfile + uv.** All commands run inside the container.

**Never use:** `pip`, `pip install`, `uv pip install`, `conda`, or bare `python`.

```bash
# Add dependencies (NEVER manually edit pyproject.toml dependency sections)
uv add <package>                           # runtime
uv add --optional <group> <package>        # optional (e.g. --optional interactive plotly)
uv add --dev <package>                     # dev

# Install / sync
uv sync --all-extras --all-groups

# Run
uv run <cmd>
```

## Development Rules

Detailed rules are in `.claude/rules/`:

- `rules/code-style.md` — Python style, naming, type annotations, Pydantic v2
- `rules/testing.md` — pytest conventions, test structure
- `rules/uv.md` — uv package manager usage (enforced)
- `rules/planning.md` — Planning before non-trivial changes
- `rules/fail-loud.md` — never silently swallow evaluation/metrics errors into empty results

Additional project conventions in `.claude/project_guidelines.md` (docs/test directory organization).
