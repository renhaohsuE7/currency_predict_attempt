---
paths:
  - "tests/**/*.py"
---

# Testing Conventions

## Stack

- **pytest** as test runner
- `unittest.mock` / `pytest-mock` for mocking

## Directory Layout

```text
tests/
├── conftest.py              # Global fixtures
├── test_config_manager.py   # Unit tests per module
├── test_data_collectors.py
├── test_models_patchtst.py
└── ...
```

- All tests go in `/tests/` — never in root or `src/`
- File naming: `test_<module_name>.py`
- Function naming: `test_<scenario>_<expected_outcome>`
  - e.g. `test_predict_with_empty_data_raises_error`
  - No `test_1`, `test_it_works` vagueness

## Bug Fix Workflow

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Confirm the test passes

## Test Scope

- Unit tests: test individual modules in isolation, mock external I/O (Yahoo Finance, file system)
- Integration tests: test `PredictionPipeline` end-to-end with mock data
- Test edge cases: empty data, missing columns, invalid config values

## Docker Execution

Tests must run inside Docker containers, not on the host.

```bash
# One-off full test suite (creates & removes container)
docker compose run --rm test

# Single file / single test
docker compose run --rm test uv run pytest tests/test_config_manager.py -v
docker compose run --rm test uv run pytest tests/test_config_manager.py::test_fn -v

# Interactive dev (keep container running, faster iteration)
docker exec -it currency-pred-dev uv run pytest -v
docker exec -it currency-pred-dev uv run pytest tests/test_config_manager.py -v
```
