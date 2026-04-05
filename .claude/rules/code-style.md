---
paths:
  - "src/**/*.py"
  - "main.py"
---

# Code Style

## Python General

- **Python 3.11+** — 可自由使用 `match/case`、`tomllib`、新型 typing features
- **Type annotations** on all public functions and class attributes — 避免 `Any` 除非不得已
- **Line length**: 88 characters (Black default)
- **Formatter**: `black`
- **Linter**: `flake8`
- **Type checker**: `mypy`

## Naming Conventions

| Entity | Convention | Example |
| --- | --- | --- |
| Variables / functions | `snake_case` | `process_data` |
| Classes | `PascalCase` | `PredictionPipeline` |
| Constants | `UPPER_SNAKE` | `MAX_LAG_FEATURES` |
| Pydantic models | `PascalCase` | `DataCollectionConfig` |
| Private helpers | leading underscore | `_validate_input` |

## Pydantic v2 Patterns

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict

class MyConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    period: str = Field(default="1y", description="Data collection period")
```

- Use `model_config = ConfigDict(...)` — **not** `class Config`
- Use `@field_validator` — **not** `@validator`
- Use `Annotated[type, Field(...)]` for reusable field definitions

## Docstrings

- Google-style docstrings on public functions and classes
- Inline comments only when logic is non-obvious

```python
def calculate_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Calculate lag features for time series data.

    Args:
        df: Input DataFrame with price data.
        lags: List of lag periods to compute.

    Returns:
        DataFrame with added lag columns.

    Raises:
        ValueError: When lags contain non-positive values.
    """
```

## Module Organization

- One concern per module; avoid catch-all "utils.py" growing unbounded
- ML models implement `BaseModel` ABC from `models/base.py`
- New model variants register in `ModelFactory` (`models/factory.py`)
