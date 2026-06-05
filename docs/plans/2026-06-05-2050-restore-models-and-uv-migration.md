# Plan: 重實作遺失的 models/ 評估層 + 全面收斂到 docker compose + uv

- **Date**: 2026-06-05 20:50
- **Status**: completed
- **Module**: `models/`(base、naive、patchtst 子類)、CI、文件
- **Related**: `docs/issues/2026-06-05-models-package-accidental-deletion.md`(P0)

## 目標

1. **重實作 stage (b)**:補回因誤刪而遺失、且 git 無法復原的模型端程式碼
   (`BaseModel.evaluate_single_shot` / `evaluate_rolling` + `models/naive.py`),
   讓工作目錄既有的消費端(predictor / runner / formatter / tests)恢復可運作。
2. **收斂工具鏈**:全面確保只用 docker compose + Dockerfile + uv(`uv add` / `uv sync`);
   掃除殘留的 `pip install` / `uv pip install` 參照;確認無 `requirements.txt` 並建立防迴歸。

## 前置事實(已調查)

- `models/` 已用 `git checkout HEAD -- src/currency_predictor/models/` 恢復,import 全通(stage (a) 完成)。
- 恢復的是**舊版**:`BaseModel` 只有 `fit`/`predict`/`predict_with_uncertainty`(abstract)、
  `get_model_info`/`save_model`/`load_model`;**無** `evaluate*`。三個 PatchTST 子類各自有舊的 `evaluate()`。
- **消費端與測試已存在於工作目錄**(modified / untracked),已按新 API 撰寫:
  - `prediction/predictor.py:371/382` 呼叫 `model.evaluate_rolling(...)` / `evaluate_single_shot(...)`
  - `backtesting/result.py` 已有 `per_horizon_metrics` / `n_origins` / `avg_per_horizon_metrics` 欄位
  - `backtesting/runner.py`、`reporting/formatter.py` 已按 rolling/per-horizon 改寫
  - `prediction/metrics.py`(untracked)已提供 `mase()` / `mda()`
  - 驗收測試(untracked,已是規格):`tests/test_naive_model.py`、`tests/test_evaluate_rolling.py`、`tests/test_metrics.py`
- 規格來源(completed 但 untracked 的 plan):
  - `2026-04-05-2032-naive-baseline-and-mase`、`2026-04-05-2140-unify-evaluate-into-basemodel`
  - `2026-04-07-0000-rename-evaluate-single-shot`、`2026-04-08-evaluate-rolling-per-horizon`
- 工具鏈現況:**無 requirements.txt / setup.py / Pipfile**。pyproject + uv 已就位。殘留 pip 參照:
  - `.github/workflows/ci.yml:46` → `uv sync ... && uv pip install mypy`(**真正違規**)
  - 文件:`notebooks/README.md`、`examples/README.md`、`docs/development/notebook_organization.md` 多處 `pip install` / `uv pip install -e .`(文案,需改)

## 驗收測試已 pin 住的 API(不可偏離)

**`NaiveModel`**(`tests/test_naive_model.py`):
- `NaiveModel(pred_len=24)` 預設 `pred_len == 24`;`model_name == "NaiveModel"`;初始 `is_fitted == False`
- `fit(X, y)` 為 no-op:回傳 `self`、設 `is_fitted = True`、`training_history["epochs"] == 0`
- `predict(X)` 回傳「最後一個 `Close` 值」重複 `pred_len` 次
- 可由 `ModelFactory` 建立(註冊名 `naive`)

**`evaluate_rolling`**(`tests/test_evaluate_rolling.py`):
- 簽名 `evaluate_rolling(X, y, seq_len, pred_len, y_train=None, step=1)`
- 回傳鍵:`aggregate`、`per_horizon`、`h1_actual`、`h1_predicted`
- `aggregate` 含標準 metric 鍵;有 `y_train` 時含 `mase`,無則不含;含 `n_origins`
- `n_origins == len(X) - seq_len - pred_len + 1`(step=1)
- `per_horizon` 為 1-indexed、長度 == `pred_len`、每項含 `rmse`/`mae`
- 資料不足(`len(X) < seq_len + pred_len`)時 `raise ValueError`

## 影響範圍

| 檔案 | 動作 |
| --- | --- |
| `src/currency_predictor/models/base.py` | **新增** `evaluate_single_shot()` 與 `evaluate_rolling()`(Pull Up + reuse metrics) |
| `src/currency_predictor/models/patchtst/sklearn/model.py` | **刪除** 子類 `evaluate()`(改繼承) |
| `src/currency_predictor/models/patchtst/huggingface/model.py` | **刪除** 子類 `evaluate()` |
| `src/currency_predictor/models/patchtst/lightning/wrapper.py` | **刪除** 子類 `evaluate()` |
| `src/currency_predictor/models/naive.py` | **新檔** `NaiveModel(BaseModel)` |
| `src/currency_predictor/models/factory.py` | 註冊 `naive`;`get_available_models` 加入 naive |
| `src/currency_predictor/models/__init__.py` | export `NaiveModel` |
| `.github/workflows/ci.yml` | `uv pip install mypy` → 改為 dev dependency(`uv add --dev mypy` 後 `uv sync --all-groups`) |
| `notebooks/README.md`, `examples/README.md`, `docs/development/notebook_organization.md` | `pip install` / `uv pip install -e .` 文案改為 `uv add` / `uv sync` |
| `.env`(選擇性) | `APP_UID/APP_GID` 與 host 不符的處理(見風險) |
| `pyproject.toml`(選擇性) | build-backend `setuptools`→`hatchling`,根除 egg-info 權限問題 |

> 既有但**不需新寫**(已在 WT):`predictor.py`、`runner.py`、`result.py`、`formatter.py`、
> `prediction/metrics.py` 及三個測試 — 它們是消費端/驗收,只要模型端補齊即應轉綠。

## 實作步驟

### Part A — 重實作模型端(stage b)

0. **建分支**:`git checkout -b fix/restore-models-and-uv`(挾帶現有工作目錄變更)。
1. `prediction/metrics.py` 確認 `mase`/`mda` 簽名符合(已存在),必要時補一個內部 `_compute_metrics(actual, pred, y_train)` 共用聚合(rmse/mae/mape/mda/(+mase))。
2. `base.py` 新增 `evaluate_single_shot(X, y_true, y_train=None)`:把三個 PatchTST 子類重複的 `evaluate()` 邏輯 Pull Up(predict + 統一 metrics,reuse step 1),語意為 single-shot。
3. `base.py` 新增 `evaluate_rolling(X, y_true, seq_len, pred_len, y_train=None, step=1)`:
   依 `2026-04-08` plan 的 reference 實作(外部切 window 餵 `predict()`,Option A;per-horizon column-wise;
   `aggregate` flatten 後算;回 `h1_actual`/`h1_predicted`)。
4. 刪除 sklearn / HF / lightning 子類的 `evaluate()`(改繼承 base)。
5. `models/naive.py` 實作 `NaiveModel`:persistence(重複最後 Close);`fit` no-op;`predict_with_uncertainty` 簡單回傳。
6. `factory.py` 註冊 `naive`(`get_available_models` + `create_model`);`models/__init__.py` export。

### Part B — 工具鏈收斂(docker + uv)

7. 在容器內把 mypy 收為 dev 依賴:`uv add --dev mypy`(更新 pyproject + uv.lock),改 `ci.yml` 用 `uv sync --all-extras --all-groups`,移除 `uv pip install mypy`。
8. 文件文案替換:`pip install <pkgs>` → `uv sync --all-extras`;`uv pip install -e .` → `uv sync`(editable 由 uv 自動處理)。
9. 防迴歸:確認 repo 無 `requirements.txt`(本次已確認);於 `.claude/rules/uv.md` 既有禁則下,新增一條 grep-able 檢查(可選:在 CI 加一步 `! test -f requirements.txt`)。
10.(選擇性,建議)`pyproject.toml` build-backend 改 `hatchling`,根除 nonroot 容器 editable build 的 egg-info 權限失敗;或在文件記錄 `APP_UID=$(id -u) APP_GID=$(id -g)` 的正確跑法。

### Part C — 驗證

11. 容器內逐步測試(由小到大):
    ```bash
    APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test \
      uv run pytest tests/test_metrics.py tests/test_naive_model.py tests/test_evaluate_rolling.py -q
    ```
12. 擴大:`tests/test_backtest_runner.py`、`tests/test_model_comparer.py`、`tests/test_prediction_pipeline.py`。
13. 全測:`APP_UID=... docker compose run --rm test`(預期排除既有 HF sandbox PermissionError)。
14. `uv run black .` / `flake8` / `mypy`。

## 風險評估

- **API 偏離測試** → 嚴格以三個 untracked 測試為準,先跑這三檔當紅綠燈。
- **evaluate_rolling 與 runner 既有期望不一致**(runner 已用 `h1_actual`/`h1_predicted`/`aggregate`)→ 回傳結構嚴格照 `2026-04-08` plan。
- **`.env` uid 不符**:本機 host uid=1003,`.env`=1006/1007 → build egg-info 權限失敗。對策:跑指令帶 `APP_UID=$(id -u) APP_GID=$(id -g)`,或改 build-backend 為 hatchling(step 10)。不直接改 `.env` 數字(可能對應其他主機)。
- **HF/Lightning optional 未安裝** → 模型端改動須在 sklearn-only 環境也能 import(維持現有 try/except optional 匯入)。

## 完成標準

- [ ] `tests/test_naive_model.py` / `test_evaluate_rolling.py` / `test_metrics.py` 全綠
- [ ] `predictor._evaluate_model` 走 rolling 路徑不再 `AttributeError`
- [ ] 全測通過(排除既有 HF sandbox PermissionError)
- [ ] CI 無 `uv pip install`;repo 無 `requirements.txt`;文件無 `pip install` 安裝指引
- [ ] `docs/issues/2026-06-05-models-package-accidental-deletion.md` Status → completed
