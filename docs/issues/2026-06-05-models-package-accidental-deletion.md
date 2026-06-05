# Issue: `models/` 套件遭誤刪導致整個套件無法 import (P0)

- **Date**: 2026-06-05
- **Status**: completed（stage (a) import 已修復;stage (b) 方法/模組已重實作並測試通過）
- **Severity**: P0 — blocker (套件完全無法 import,主流程全毀)
- **Module**: `models/`(被刪)、`prediction/predictor`、`prediction/comparer`、`backtesting/runner`

## 症狀

工作目錄中 `src/currency_predictor/models/` 整個套件(13 檔)被刪除,但仍有多處 import 它:

```
src/currency_predictor/prediction/predictor.py:16  from ..models.factory import ModelFactory, create_patchtst_model
src/currency_predictor/prediction/predictor.py:17  from ..models.patchtst.config import TrainingConfig
src/currency_predictor/prediction/comparer.py:18   from ..models.factory import ModelFactory
src/currency_predictor/backtesting/runner.py:16    from ..models.factory import ModelFactory
```

任何 `import currency_predictor.prediction.predictor` 立即 `ModuleNotFoundError`。預設 `python main.py`
(`compare=True` / `visualize=True`,config `model_names: null` 需呼叫 `ModelFactory.get_available_models()`)整條主流程死掉。

## 診斷:確認為誤刪(刪到仍被依賴的目錄)

| 檢查 | 結果 |
| --- | --- |
| `models/` 在 HEAD (`a4dfebf`) | ✅ 完整存在、import 一致、可運作 |
| 刪除範圍 | 僅工作目錄、**未 staged**（`git status` 全為 `D`）;無 stash;reflog 僅 clone + 1 commit |
| importer 是否同步改寫 | ❌ `predictor/comparer/runner` 仍 `from ..models.factory import ...`(有意移除重構會一併改掉) |
| 替代 model 檔 | ❌ 全樹無 untracked model 實作 |
| WT 是否依賴 models/ | ✅ import `ModelFactory`/`create_patchtst_model`/`TrainingConfig`;呼叫 `ModelFactory.create_model`、`model.evaluate_rolling`、`model.evaluate_single_shot` |

結論:這是把「仍被工作目錄依賴」的目錄誤刪,而非協調好的重構。

## 嚴重轉折:部分內容 git 無法復原

- HEAD 版 `models/` 具備 `ModelFactory` / `create_patchtst_model` / `TrainingConfig` / `create_model` /
  `get_available_models` → `git restore` 可修好 **import**。
- 但 HEAD 版 `models/` **不含** `evaluate_rolling` / `evaluate_single_shot`(全樹皆無定義),而工作目錄的
  `predictor.py:371/382` 會呼叫它們。描述新增這兩者的計畫
  `docs/plans/2026-04-08-evaluate-rolling-per-horizon.md` 與
  `docs/plans/2026-04-05-2140-unify-evaluate-into-basemodel.md` 皆標 **completed**,但兩者都是
  **untracked(未 commit)**。
- 代表「較新、含 `evaluate_rolling`/`evaluate_single_shot` 的 `models/` 實作」是**未 commit 的工作,隨刪除一起消失**,
  因無 commit / stash / reflog 而**無法用 git 復原**。

## 修復(兩段)

1. **(a) 立即解 import** — 恢復 HEAD 版 models/:

   ```bash
   git restore --source=HEAD --staged --worktree -- src/currency_predictor/models/
   # 或： git checkout HEAD -- src/currency_predictor/models/
   ```

   驗證:`docker compose run --rm test uv run python -c "import currency_predictor.prediction.predictor"`

2. **(b) 重新實作遺失的方法** — 依下列兩份 completed plan,在恢復後的 `models/`(`BaseModel` 及各 PatchTST 子類)
   補回 `evaluate_rolling()` 與 `evaluate_single_shot()`:
   - `docs/plans/2026-04-05-2140-unify-evaluate-into-basemodel.md`
   - `docs/plans/2026-04-08-evaluate-rolling-per-horizon.md`

   驗證:`tests/test_evaluate_rolling.py`(untracked,已存在)應通過。

## 實測結果（2026-06-05）

### stage (a) — 完成 ✅

```bash
git checkout HEAD -- src/currency_predictor/models/   # 恢復 13 檔
```

容器內 import 全部通過(`models.factory`、`prediction.predictor`、`prediction.comparer`、
`backtesting.runner`、`verification.verifier`)。

> **環境陷阱**:`.env` 設 `APP_UID=1006/APP_GID=1007`,但本機 host uid/gid 為 **1003**。
> 因 dev/test 以可寫方式 bind-mount `./src`,容器使用者 uid 不符會導致 editable build 在 `src/` 建
> `egg-info` 時 `Permission denied`。需以 `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run ...`
> 執行(或修正 `.env`)。另需先移除 stale 的 `src/currency_predict_attempt.egg-info`(未追蹤,已刪)。

### stage (b) — 待重實作（git 無法復原的遺失工作）

恢復後的 HEAD 版 models/ 僅提供 `base / factory / patchtst.{config,huggingface,lightning,sklearn}`,
各模型只有 `evaluate()`。WT 程式碼 / 測試仍引用以下**不存在**的符號:

| 缺失 | 引用處 | 規格來源(completed 但 untracked 的 plan) |
| --- | --- | --- |
| `models/naive.py`（`NaiveModel`) | `tests/test_evaluate_rolling.py`、factory 註冊 | `docs/plans/2026-04-05-2032-naive-baseline-and-mase.md` |
| `evaluate_rolling()`（BaseModel + 各子類) | `predictor.py:371`、`tests/test_evaluate_rolling.py` | `docs/plans/2026-04-08-evaluate-rolling-per-horizon.md` |
| `evaluate_single_shot()`（BaseModel + 各子類) | `predictor.py:382` | `docs/plans/2026-04-07-0000-rename-evaluate-single-shot.md`、`docs/plans/2026-04-05-2140-unify-evaluate-into-basemodel.md` |

在 stage (b) 完成前,任何走訓練/評估的執行期路徑(`_evaluate_model` → `evaluate_rolling`)會丟
`AttributeError`(被 except 吞 → 空 metrics),且 `tests/test_evaluate_rolling.py` 等測試 import 失敗。

### 驗證結果（2026-06-05）

- 全測:**633 passed, 1 failed, 49 deselected**（容器內 `docker compose run --rm test`）。
- 唯一失敗 `test_models_patchtst.py::...test_predict_with_subset_columns_uses_close_or_fallback`
  為**既有、與本 issue 無關**的矛盾測試(WT 新增測試期望 fallback,但 sklearn `predict()` 依
  issue `2026-04-03-sklearn-predict-feature-mismatch` 的刻意設計會 raise)。待另行決定方向。
- 環境修正:`results/`、`models/`、`.cache/` 原為 Docker daemon 以 root 自動建立 → 已 chown 至
  host uid(1003),解除 visualizer/HF/lightning 的 `PermissionError`。

## 預防

- 將兩份 untracked plan 與相關工作 commit 進版本控制,避免「completed 卻只存在於工作目錄」。
- 重要重構前先建分支並 commit baseline(符合專案 branching 規則)。

## 後續

- import 修好後,`docs/issues/2026-06-05-code-review-findings.md` 列出的其餘 bug 才會真正浮現,接續處理。
