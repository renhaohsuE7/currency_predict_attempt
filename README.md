# Currency Prediction Attempt

A machine learning project for predicting currency exchange rates.

## Project Structure

```
currency_predict_attempt/
├── src/
│   ├── currency_predictor/
│   │   ├── __init__.py
│   │   ├── data_collector.py      # Data collection from APIs
│   │   ├── data_processor.py      # Data cleaning and preprocessing
│   │   ├── models.py              # ML models for prediction
│   │   └── utils.py               # Utility functions
├── notebooks/                     # Jupyter notebooks for analysis
├── data/                         # Raw and processed data
├── tests/                        # Unit tests
├── main.py                       # Main entry point
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Setup

1. Make sure you have `uv` installed
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Activate the virtual environment:
   ```bash
   uv shell
   ```

## Usage

Run the main script:
```bash
uv run main.py
```

## Development

Install development dependencies:
```bash
uv sync --extra dev
```

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black .
```

## Features

- [ ] Currency data collection from APIs
- [ ] Data preprocessing and feature engineering
- [ ] Multiple ML models for prediction
- [ ] Model evaluation and comparison
- [ ] Visualization of results
- [ ] Real-time prediction capabilities
