# Plan #0120: mypy Type Checking

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: 全專案
- **Priority**: P1

## 目標

配置 mypy 靜態型別檢查，修復 type errors，並整合到開發流程。

## 實作摘要

### 配置

`pyproject.toml` 新增：

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
plugins = ["pydantic.mypy"]
exclude = ["models/patchtst/lightning/"]

[[tool.mypy.overrides]]
module = "currency_predictor.models.patchtst.huggingface.model"
disable_error_code = ["attr-defined", "assignment", "arg-type", "misc", "call-overload", "no-any-return"]

[tool.pydantic-mypy]
init_forbid_extra = false
init_typed = false
warn_required_dynamic_aliases = false
```

### Baseline → 0 errors

| 階段 | Error count |
| --- | --- |
| Baseline | 111 errors in 14 files |
| After fixes | **0 errors in 26 files** |

### 修復分類

| 類型 | 修復方式 | 影響檔案 |
| --- | --- | --- |
| Implicit Optional (`str = None`) | 改為 `Optional[str] = None` | storage.py, data_processor.py, sklearn/model.py, factory.py, visualizer.py, huggingface/model.py |
| Missing annotations (`var-annotated`) | 加入明確型別 | base.py, sklearn/model.py |
| `no-any-return` | `dict(x)` / 明確 type hint | utils.py, factory.py, predictor.py |
| List→ndarray reassignment | 改用不同變數名 | sklearn/model.py |
| Import fallback (`X = None`) | `# type: ignore[assignment,misc]` | 5 個 `__init__.py` + factory.py |
| Pydantic Field issues | pydantic-mypy plugin | settings.py |
| ConfigManager._settings | 改型別為 `AppSettings` (非 None) | manager.py |
| HuggingFace `self.model` lifecycle | per-module override | huggingface/model.py |
| `logging.Handler` list | 明確 `list[logging.Handler]` | utils.py |

### 設計決策

| 問題 | 決策 |
| --- | --- |
| HuggingFace model 37 errors | per-module override — `self.model` 在 `_setup_model()` 才初始化，mypy 無法追蹤此 lifecycle |
| Lightning module | exclude — optional dependency |
| Pydantic compatibility | 使用 pydantic-mypy plugin |
| Import fallback patterns | `# type: ignore` — 常見 Python optional dependency pattern |

### 修改檔案

| 檔案 | 改動 |
| --- | --- |
| `pyproject.toml` | `[tool.mypy]` + `[[tool.mypy.overrides]]` + `[tool.pydantic-mypy]` |
| `data/storage.py` | `period: Optional[str]` |
| `data_processor.py` | `lags: Optional[list]` |
| `utils.py` | `dict(config)` return, `list[logging.Handler]` |
| `models/base.py` | `model_params: Dict[str, Any]`, `sequence_length: Optional[int]`, `feature_columns: list[str]` |
| `models/factory.py` | `# type: ignore` imports, `Optional[bool/str]`, explicit `model: BaseModel` |
| `models/__init__.py` | `# type: ignore` imports |
| `models/patchtst/__init__.py` | `# type: ignore` imports |
| `models/patchtst/huggingface/__init__.py` | `# type: ignore` imports |
| `models/patchtst/huggingface/model.py` | `Optional[int]` for horizon params |
| `models/patchtst/sklearn/model.py` | Type annotations for training_history, patches, flattened_features, scaler; `Optional[int]` for horizon |
| `config/manager.py` | `self._settings: AppSettings`, `# type: ignore` for DEFAULT_CONFIG |
| `prediction/predictor.py` | `dict(metrics)`, `success: bool` annotations |
| `visualization/visualizer.py` | `Optional[List[str/int]]`, narrowed None check |

## 測試結果

- **292 passed, 0 failed**
- **mypy: 0 errors in 26 source files**

## 完成標準

- [x] `uv run mypy src/currency_predictor/` 零 errors
- [x] pyproject.toml 有 `[tool.mypy]` 配置
- [ ] CI 包含 mypy step（依 #0130 完成時程）
